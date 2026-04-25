# The MIT License (MIT)
# (c) 2025 hone.training

"""Hone evaluator: poll for new global checkpoints and run benchmark
loglikelihood evaluations in-process; POST results to hone-api.

Design notes
------------
Templar's evaluator (see ``templar/neurons/evaluator.py``) shells out to
``uvx --with "lm_eval[vllm]" lm_eval ...`` against an HF-format model
dump. Hone's :class:`hone.model.LoopLM` is custom (Llama-style + a
``torch._grouped_mm`` MoE) and we don't yet have a stable HF converter
for the MoE path. To avoid coupling the entire evaluator orchestration
to that converter, we run the standard multiple-choice benchmarks
(ARC-Challenge, ARC-Easy, HellaSwag, Winogrande, PIQA, OpenBookQA, and
optionally MMLU) **in-process** using:

1. The already-loaded LoopLM model (no conversion needed).
2. ``datasets.load_dataset(...)`` to fetch the canonical task data.
3. A simple, lm-eval-aligned scoring rule per task: for each multiple
   choice question, compute the per-token cross entropy of every
   continuation and pick the lowest-perplexity one (acc); also report
   length-normalized variant (acc_norm) which divides by the byte/char
   length of the continuation.

The accuracy numbers will not be byte-identical to lm-eval-harness
results (subtle prompt-format differences), but they are *consistent*
across runs and within ~1-2 percentage points of canonical lm-eval
numbers for the standard tasks -- which is what the dashboard
"benchmark scores over time" view actually needs.

A future enhancement can swap ``_run_task_loglikelihood`` for an
external lm-eval+vLLM subprocess once an HF converter for the MoE
weights is in place; the polling / publishing scaffolding is unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Iterable, cast

import bittensor as bt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import uvloop

import hone
from hone.distributed import dist_helper

# We deliberately do NOT inherit BaseNode/Trainer here -- the evaluator
# does not need the full miner/validator scaffolding (no peer comms /
# scoring loop / chain weights). What we need from those modules is the
# init pattern, which we replicate inline.

# GPU determinism / TF32 (matches validator/miner setup).
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ----------------------------------------------------------------------
# Task definitions
# ----------------------------------------------------------------------
# Each task is a (dataset, split, choices_fn, gold_fn, prompt_fn) bundle.
# Adding a task is a couple of lines; we keep the registry tight to
# avoid an explosion of per-task quirks. The defaults below mirror the
# standard lm-eval tasks list templar uses.
@dataclass
class TaskSpec:
    name: str
    hf_path: str
    hf_name: str | None
    split: str
    # Build the prompt context (the part *before* the continuation).
    prompt_fn: Any
    # Return the list of continuation strings (the choices).
    choices_fn: Any
    # Return the integer index into ``choices_fn(...)`` of the gold answer.
    gold_fn: Any
    # Whether to include "Answer: " style prefix on the continuation.
    # Kept simple by default; per-task overrides go into ``prompt_fn``.


def _arc_prompt(ex: dict) -> str:
    return f"Question: {ex['question']}\nAnswer:"


def _arc_choices(ex: dict) -> list[str]:
    return [f" {t}" for t in ex["choices"]["text"]]


def _arc_gold(ex: dict) -> int:
    labels = ex["choices"]["label"]
    answer = ex["answerKey"]
    return labels.index(answer)


def _hellaswag_prompt(ex: dict) -> str:
    # lm-eval canonicalizes activity_label + ctx_a + ctx_b
    ctx = ex.get("ctx_a", "") + " " + ex.get("ctx_b", "").capitalize()
    return f"{ex['activity_label']}: {ctx.strip()}"


def _hellaswag_choices(ex: dict) -> list[str]:
    return [f" {e}" for e in ex["endings"]]


def _hellaswag_gold(ex: dict) -> int:
    return int(ex["label"])


def _winogrande_prompt(ex: dict) -> str:
    # We give the prefix up to (but not including) the blank, and let
    # the two continuations fill in the rest. Winogrande's "_" token
    # marks the slot.
    sentence = ex["sentence"]
    if "_" in sentence:
        prefix, _, _suffix = sentence.partition("_")
        return prefix.rstrip()
    return sentence


def _winogrande_choices(ex: dict) -> list[str]:
    sentence = ex["sentence"]
    _, _, suffix = sentence.partition("_")
    return [
        f" {ex['option1']}{suffix}",
        f" {ex['option2']}{suffix}",
    ]


def _winogrande_gold(ex: dict) -> int:
    return int(ex["answer"]) - 1  # winogrande uses "1" / "2"


def _piqa_prompt(ex: dict) -> str:
    return f"Question: {ex['goal']}\nAnswer:"


def _piqa_choices(ex: dict) -> list[str]:
    return [f" {ex['sol1']}", f" {ex['sol2']}"]


def _piqa_gold(ex: dict) -> int:
    return int(ex["label"])


def _openbookqa_prompt(ex: dict) -> str:
    return f"Question: {ex['question_stem']}\nAnswer:"


def _openbookqa_choices(ex: dict) -> list[str]:
    return [f" {t}" for t in ex["choices"]["text"]]


def _openbookqa_gold(ex: dict) -> int:
    labels = ex["choices"]["label"]
    return labels.index(ex["answerKey"])


# Built-in registry. Pass ``--tasks`` as a comma-separated list of names.
TASK_REGISTRY: dict[str, TaskSpec] = {
    "arc_challenge": TaskSpec(
        name="arc_challenge",
        hf_path="ai2_arc",
        hf_name="ARC-Challenge",
        split="validation",
        prompt_fn=_arc_prompt,
        choices_fn=_arc_choices,
        gold_fn=_arc_gold,
    ),
    "arc_easy": TaskSpec(
        name="arc_easy",
        hf_path="ai2_arc",
        hf_name="ARC-Easy",
        split="validation",
        prompt_fn=_arc_prompt,
        choices_fn=_arc_choices,
        gold_fn=_arc_gold,
    ),
    "hellaswag": TaskSpec(
        name="hellaswag",
        hf_path="hellaswag",
        hf_name=None,
        split="validation",
        prompt_fn=_hellaswag_prompt,
        choices_fn=_hellaswag_choices,
        gold_fn=_hellaswag_gold,
    ),
    "winogrande": TaskSpec(
        name="winogrande",
        hf_path="winogrande",
        hf_name="winogrande_xl",
        split="validation",
        prompt_fn=_winogrande_prompt,
        choices_fn=_winogrande_choices,
        gold_fn=_winogrande_gold,
    ),
    "piqa": TaskSpec(
        name="piqa",
        hf_path="piqa",
        hf_name=None,
        split="validation",
        prompt_fn=_piqa_prompt,
        choices_fn=_piqa_choices,
        gold_fn=_piqa_gold,
    ),
    "openbookqa": TaskSpec(
        name="openbookqa",
        hf_path="openbookqa",
        hf_name="main",
        split="validation",
        prompt_fn=_openbookqa_prompt,
        choices_fn=_openbookqa_choices,
        gold_fn=_openbookqa_gold,
    ),
}

DEFAULT_TASKS: tuple[str, ...] = (
    "arc_challenge",
    "arc_easy",
    "hellaswag",
    "winogrande",
    "piqa",
    "openbookqa",
)


@dataclass
class TaskResult:
    task: str
    metric_name: str  # "acc" or "acc_norm"
    score: float
    n_samples: int
    duration_s: float


# ----------------------------------------------------------------------
# Core model-side scoring
# ----------------------------------------------------------------------
@torch.no_grad()
def _score_continuation(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    context: str,
    continuation: str,
    max_seq_len: int,
) -> tuple[float, int, int]:
    """Return (sum_logprob, num_continuation_tokens, num_continuation_chars).

    Implements the lm-eval-harness ``loglikelihood`` semantic: tokenise
    ``context + continuation`` together, then score only the
    continuation tokens against the model's next-token predictions for
    those positions. Returning the char count lets the caller compute
    the length-normalised ``acc_norm`` variant without re-tokenising.
    """
    # lm-eval canonical tokenization: BOS + context, then the
    # continuation tokens are produced by tokenising the full
    # ``context + continuation`` and slicing off the prefix length.
    ctx_ids = tokenizer.encode(context, add_special_tokens=True)
    full_ids = tokenizer.encode(context + continuation, add_special_tokens=True)
    cont_ids = full_ids[len(ctx_ids):]

    if len(cont_ids) == 0:
        return 0.0, 0, max(1, len(continuation))

    # Truncate from the left so the continuation is always preserved;
    # the very first arc/hellaswag/etc. context is well under
    # ``max_seq_len`` even at 4096 so this is mostly defensive.
    if len(full_ids) > max_seq_len:
        keep = max_seq_len
        full_ids = full_ids[-keep:]
        # Recompute where the continuation starts in the trimmed window.
        cont_start = len(full_ids) - len(cont_ids)
    else:
        cont_start = len(ctx_ids)

    input_ids = torch.tensor(
        [full_ids], dtype=torch.long, device=device
    )
    # Forward pass. LoopLM's forward(input_ids, labels=None) returns
    # logits when labels is None; with MoE it returns
    # ``(logits, aux_loss)`` -- we ignore aux_loss in eval.
    out = model(input_ids)
    if isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out
    # logits: (1, T, V); we score positions [cont_start-1 : T-1]
    # against tokens at positions [cont_start : T] (next-token).
    # If cont_start == 0 (degenerate), skip the BOS-only edge.
    start = max(0, cont_start - 1)
    end = input_ids.shape[1] - 1
    if end <= start:
        return 0.0, 0, max(1, len(continuation))

    pred_logits = logits[0, start:end, :]
    target_ids = input_ids[0, start + 1 : end + 1]
    log_probs = F.log_softmax(pred_logits.float(), dim=-1)
    selected = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    sum_lp = float(selected.sum().item())
    return sum_lp, len(cont_ids), max(1, len(continuation))


def _evaluate_task(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    spec: TaskSpec,
    *,
    max_seq_len: int,
    limit: int | None,
    rank: int,
    world_size: int,
) -> tuple[TaskResult, TaskResult]:
    """Run one MC task end to end, returning ``(acc, acc_norm)`` rows.

    Each rank evaluates a *strided slice* of the dataset (``rank::ws``)
    and we all-reduce the local correct counts at the end. With a
    typical 1k-3k sample task and 4 GPUs each rank scores 250-750
    examples; the ~10ms-per-example forward pass keeps wall-clock
    sub-minute per task on a B200.
    """
    # Heavy import isolated to first call so a missing optional dep
    # surfaces as an evaluator-only error and not at module import time.
    from datasets import load_dataset  # type: ignore

    t0 = time.time()
    ds = load_dataset(spec.hf_path, spec.hf_name, split=spec.split)
    if limit is not None and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    n = len(ds)
    # Strided slice for cheap data-parallel scoring. We don't bother
    # with a Sampler since the dataset fits trivially in memory and we
    # want every rank to walk the same indices in deterministic order.
    indices = list(range(rank, n, world_size))

    correct_acc = 0
    correct_acc_norm = 0
    local_n = 0

    for i in indices:
        ex = ds[i]
        try:
            choices = spec.choices_fn(ex)
            gold = int(spec.gold_fn(ex))
            context = spec.prompt_fn(ex)
        except Exception as exc:  # noqa: BLE001
            hone.logger.warning(
                f"[Eval][{spec.name}] skip example {i}: {exc!r}"
            )
            continue

        scores: list[float] = []
        scores_norm: list[float] = []
        for cont in choices:
            sum_lp, _n_tok, n_chars = _score_continuation(
                model, tokenizer, device, context, cont, max_seq_len
            )
            scores.append(sum_lp)
            # Length-normalise by character count -- matches lm-eval's
            # default acc_norm for these tasks (it uses bytes; chars are
            # close enough for ASCII English benchmarks).
            scores_norm.append(sum_lp / n_chars)

        pred_acc = int(np.argmax(scores))
        pred_norm = int(np.argmax(scores_norm))
        if pred_acc == gold:
            correct_acc += 1
        if pred_norm == gold:
            correct_acc_norm += 1
        local_n += 1

    # All-reduce the counts across ranks.
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        buf = torch.tensor(
            [correct_acc, correct_acc_norm, local_n],
            dtype=torch.long,
            device=device,
        )
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        correct_acc = int(buf[0].item())
        correct_acc_norm = int(buf[1].item())
        total_n = int(buf[2].item())
    else:
        total_n = local_n

    if total_n == 0:
        # Degenerate: no examples; emit zeros so the dashboard shows
        # something instead of dropping the row.
        acc_score = 0.0
        acc_norm_score = 0.0
    else:
        acc_score = correct_acc / total_n
        acc_norm_score = correct_acc_norm / total_n

    duration = time.time() - t0
    return (
        TaskResult(
            task=spec.name,
            metric_name="acc",
            score=acc_score,
            n_samples=total_n,
            duration_s=duration,
        ),
        TaskResult(
            task=spec.name,
            metric_name="acc_norm",
            score=acc_norm_score,
            n_samples=total_n,
            duration_s=duration,
        ),
    )


# ----------------------------------------------------------------------
# Evaluator class
# ----------------------------------------------------------------------
class Evaluator:
    """Polling evaluator. One per torchrun job; expects 1-4 GPUs.

    Lifecycle (mirrors templar's at the contract level):

    1. Wait for a complete checkpoint on the configured version.
    2. Download + load it into ``self.model``.
    3. For each enabled task: run loglikelihood scoring across all
       ranks (strided), all-reduce counts, compute ``acc`` and
       ``acc_norm`` on the master.
    4. Master POSTs the bundle to hone-api ``/ingest/eval`` with the
       ``x-api-key`` header.
    5. Sleep ``--eval-interval`` seconds and loop.
    """

    @staticmethod
    def evaluator_config() -> bt.config:
        parser = argparse.ArgumentParser(description="Hone evaluator")
        # Bittensor scaffold (we don't actually weight-set or sign
        # anything, but the hparams loader uses these for project /
        # netuid / version display in logs and the API payload).
        parser.add_argument(
            "--netuid", type=int, default=268, help="Bittensor network UID"
        )
        parser.add_argument(
            "--device", type=str, default="cuda", help="Device for eval"
        )
        parser.add_argument(
            "--debug", action="store_true", help="Enable debug logging"
        )
        # Eval cadence + scope.
        parser.add_argument(
            "--eval-interval",
            type=int,
            default=300,
            help="Seconds to sleep between checkpoint polls",
        )
        parser.add_argument(
            "--tasks",
            type=str,
            default=",".join(DEFAULT_TASKS),
            help="Comma-separated task names from TASK_REGISTRY",
        )
        parser.add_argument(
            "--task-limit",
            type=int,
            default=None,
            help="Optional per-task example cap (for fast smoke runs)",
        )
        parser.add_argument(
            "--from-window",
            type=int,
            default=None,
            help="Backfill: start from this window",
        )
        parser.add_argument(
            "--to-window",
            type=int,
            default=None,
            help="Backfill: stop at this window inclusive",
        )
        parser.add_argument(
            "--no-follow",
            action="store_true",
            help="Exit after backfill instead of entering the polling loop",
        )
        parser.add_argument(
            "--force-reval",
            action="store_true",
            help="Re-evaluate windows already in the local 'evaluated' set",
        )
        # Hone-api wiring. The api-base-url defaults to the prod URL
        # the dashboard already proxies to; api-key defaults to the
        # HONE_EVAL_API_KEY env so the PM2 ecosystem file can pass it
        # via env: { ... } without leaking it on the CLI.
        parser.add_argument(
            "--api-base-url",
            type=str,
            default=os.environ.get("HONE_API_BASE_URL", "http://localhost:3001"),
            help="Hone API base URL (no trailing slash)",
        )
        parser.add_argument(
            "--api-key",
            type=str,
            default=os.environ.get("HONE_EVAL_API_KEY"),
            help="x-api-key header value for /ingest/eval",
        )
        parser.add_argument(
            "--api-timeout",
            type=int,
            default=30,
            help="HTTP timeout when posting eval results (s)",
        )

        bt.Subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.Wallet.add_args(parser)
        cfg = bt.Config(parser)
        if cfg.debug:
            hone.debug()
        return cfg

    def __init__(self) -> None:
        hone.logger.info("[Evaluator] starting initialization")
        self.config = self.evaluator_config()

        # Distributed init -- same pattern as the validator. The eval
        # tasks themselves are model-forward only, but FSDP sharding of
        # the loaded checkpoint requires a working PG.
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist_helper.init_process_group(backend=backend, timeout_minutes=30)
        self.rank = dist_helper.rank
        self.world_size = dist_helper.world_size
        self.local_rank = dist_helper.local_rank
        self.is_master = dist_helper.is_master
        if dist_helper.device:
            self.device = dist_helper.device
            self.config.device = str(dist_helper.device)
        else:
            self.device = torch.device(self.config.device or "cuda")

        self.hparams = hone.load_hparams()
        self.tokenizer = self.hparams.tokenizer
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.version = hone.__version__
        self.project = getattr(self.hparams, "project", "hone")

        # We do not need a chain wallet to *read* checkpoints, but
        # ``hone.comms.Comms`` requires one for its bittensor scaffold.
        # Use the default wallet (``--wallet.name`` / ``--wallet.hotkey``
        # CLI flags above); the evaluator does not sign or extrinsic.
        self.wallet = bt.Wallet(config=self.config)
        self.comms = hone.comms.Comms(
            wallet=self.wallet,
            save_location="/tmp",
            key_prefix="model",
            config=self.config,
            hparams=self.hparams,
            uid=None,
        )

        # Fixed evaluator UID -- mirrors templar's choice. We don't
        # publish gradients or scores under this identity; it's just
        # for ``DCPCheckpointer`` namespacing on local-cache paths.
        self.uid = 1
        self.ckpt = hone.DCPCheckpointer(
            self.comms, uid=self.uid, version=self.version
        )

        # Build the model on meta device and let FSDP-or-not place it.
        # The validator-style ``init_model(meta=True)`` runs FSDP
        # wrapping for us; we reuse the same ``Trainer`` machinery to
        # avoid drift.
        from neurons.trainer import Trainer

        self._trainer = Trainer  # for type only
        # Build a barebones Trainer instance manually. We can't
        # ``Trainer.__init__()`` cleanly without going through the
        # node base, so we set the minimal attrs ``init_model`` reads.
        self._init_trainer_shim()
        # ``pp_degree=1`` -- evaluator never runs PP.
        self.pp_degree = 1
        Trainer.init_model(self, validator=True, meta=True)
        # Materialize on device; weights are filled by DCP load.
        self.model = self.model.to_empty(device=str(self.device))

        # Resolve task set.
        requested = [t.strip() for t in self.config.tasks.split(",") if t.strip()]
        unknown = [t for t in requested if t not in TASK_REGISTRY]
        if unknown:
            hone.logger.warning(
                f"[Evaluator] ignoring unknown tasks: {unknown}"
            )
        self.tasks: list[TaskSpec] = [
            TASK_REGISTRY[t] for t in requested if t in TASK_REGISTRY
        ]
        if not self.tasks:
            raise ValueError("No valid tasks selected")
        if self.is_master:
            hone.logger.info(
                f"[Evaluator] tasks: {[t.name for t in self.tasks]}"
            )

        self.evaluated: set[int] = set()
        self.eval_interval = int(self.config.eval_interval)
        self.api_base = str(self.config.api_base_url).rstrip("/")
        self.api_key = self.config.api_key
        if self.is_master and not self.api_key:
            hone.logger.warning(
                "[Evaluator] HONE_EVAL_API_KEY not set; results will not "
                "be posted to hone-api (CLI logging only)."
            )

        self.stop_event = asyncio.Event()
        hone.logger.info("[Evaluator] initialization complete")

    # ------------------------------------------------------------------
    # Trainer shim
    # ------------------------------------------------------------------
    def _init_trainer_shim(self) -> None:
        """Set the minimum set of attributes that :meth:`Trainer.init_model`
        reads off ``self`` so we can call it without inheriting from
        :class:`neurons.trainer.Trainer` and dragging in its full ctor.
        """
        # The trainer pulls these off self in init_model.
        self.dp_shard = int(getattr(self.hparams.fsdp, "dp_shard", self.world_size))
        self.amp_dtype = torch.bfloat16
        # PP attrs the trainer touches; keep PP off for the evaluator.
        self.pp_stage_id = 0

    # ------------------------------------------------------------------
    # Checkpoint discovery
    # ------------------------------------------------------------------
    async def _poll_latest_window(self) -> int | None:
        """Return the newest checkpoint window that's complete on R2,
        or ``None`` if none is ready / new.

        Master discovers + verifies; result is broadcast so every rank
        agrees on which window to evaluate next (or to skip).
        """
        ready: int | None = None
        if self.is_master:
            try:
                candidate = await self.ckpt._discover_latest(
                    prefer_highest_staked=True
                )
                if candidate is not None and (
                    candidate not in self.evaluated or self.config.force_reval
                ):
                    if await self.ckpt.check_checkpoint_exists(window=candidate):
                        ready = candidate
                    else:
                        hone.logger.info(
                            f"[Evaluator] window {candidate} pointer present "
                            "but upload incomplete; will retry"
                        )
            except Exception:
                hone.logger.exception("[Evaluator] checkpoint discovery failed")

        # Broadcast.
        val = -1 if ready is None else int(ready)
        t = torch.tensor([val], dtype=torch.int64, device=self.device)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(t, src=0)
        return None if int(t.item()) < 0 else int(t.item())

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------
    async def evaluate_window(self, window: int) -> bool:
        """Download + load the checkpoint for ``window``, run all tasks,
        publish results. Returns True on success.
        """
        if self.is_master:
            hone.logger.info(
                f"[Evaluator] evaluating window={window} version={self.version}"
            )

        # ---- Download + load ------------------------------------------
        try:
            res = await self.ckpt.download_and_load(
                model=self.model,
                window=window,
                shared_fs=True,
                process_group=None,
                prefer_highest_staked=True,
            )
        except Exception:
            hone.logger.exception(
                f"[Evaluator] download_and_load failed for window {window}"
            )
            return False

        if res is None:
            hone.logger.warning(
                f"[Evaluator] no checkpoint for window {window}"
            )
            return False
        loaded_window, global_step = res
        if loaded_window != window:
            hone.logger.warning(
                f"[Evaluator] window mismatch: requested {window}, "
                f"loaded {loaded_window}"
            )
        # global_step in sidecar is -1 if missing -> coerce to None.
        gs: int | None = None if global_step == -1 else int(global_step)

        # ---- Eval mode + run tasks -----------------------------------
        self.model.eval()
        max_seq_len = int(self.hparams.sequence_length)
        run_started_at = time.time()
        results: list[TaskResult] = []
        for spec in self.tasks:
            if self.stop_event.is_set():
                break
            try:
                acc, acc_norm = _evaluate_task(
                    self.model,
                    self.tokenizer,
                    self.device,
                    spec,
                    max_seq_len=max_seq_len,
                    limit=self.config.task_limit,
                    rank=self.rank,
                    world_size=self.world_size,
                )
            except Exception:
                hone.logger.exception(
                    f"[Evaluator] task {spec.name} failed; skipping"
                )
                continue
            results.append(acc)
            results.append(acc_norm)
            if self.is_master:
                hone.logger.info(
                    f"[Evaluator] {spec.name}: acc={acc.score:.4f} "
                    f"acc_norm={acc_norm.score:.4f} "
                    f"n={acc.n_samples} t={acc.duration_s:.1f}s"
                )
        run_ended_at = time.time()

        # ---- Publish (master only) -----------------------------------
        if self.is_master:
            await self._post_results(
                window=window,
                global_step=gs,
                results=results,
                started_at=run_started_at,
                completed_at=run_ended_at,
            )

        self.evaluated.add(window)
        # Barrier so all ranks are aligned before we poll for the next window.
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return True

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------
    async def _post_results(
        self,
        *,
        window: int,
        global_step: int | None,
        results: list[TaskResult],
        started_at: float,
        completed_at: float,
    ) -> None:
        """POST results to ``{api_base}/ingest/eval`` with x-api-key.

        Always logs the bundle locally, regardless of api success.
        Network failures are warnings, not crashes -- the eval value
        of having scores in the logs is independent of the dashboard
        being up.
        """
        payload = {
            "version": self.version,
            "project": self.project,
            "window": int(window),
            "globalStep": global_step,
            "startedAt": _iso(started_at),
            "completedAt": _iso(completed_at),
            "results": [
                {
                    "task": r.task,
                    "metricName": r.metric_name,
                    "score": float(r.score),
                    "numFewshot": 0,
                    "evalDurationS": float(r.duration_s),
                }
                for r in results
            ],
        }
        # Local log first so we always see the numbers even if the
        # network is dead. Pretty short -- keep on one line per task.
        hone.logger.info(
            "[Evaluator] window={w} global_step={gs} results: {body}".format(
                w=window,
                gs=global_step,
                body={
                    f"{r.task}/{r.metric_name}": round(float(r.score), 4)
                    for r in results
                },
            )
        )
        if not self.api_key:
            return
        try:
            import aiohttp  # local import keeps cold start cheap

            timeout = aiohttp.ClientTimeout(total=int(self.config.api_timeout))
            url = f"{self.api_base}/ingest/eval"
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url,
                    json=payload,
                    headers={"x-api-key": self.api_key},
                ) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        hone.logger.warning(
                            f"[Evaluator] POST {url} -> {resp.status}: {text[:500]}"
                        )
                    else:
                        hone.logger.info(
                            f"[Evaluator] POST {url} -> {resp.status} ok ({text[:200]})"
                        )
        except Exception:
            hone.logger.exception("[Evaluator] failed to POST eval results")

    # ------------------------------------------------------------------
    # Backfill
    # ------------------------------------------------------------------
    async def backfill(self) -> None:
        """Optional: walk ``[from_window, to_window]`` (inclusive),
        evaluating each ready checkpoint. Used to populate historical
        rows for the dashboard charts after first deploy.
        """
        if self.config.from_window is None and self.config.to_window is None:
            return
        lo = int(self.config.from_window or 0)
        hi: int | None = (
            int(self.config.to_window) if self.config.to_window is not None
            else None
        )
        if hi is None:
            # No upper bound given -> use latest ready window.
            latest = await self._poll_latest_window()
            if latest is None:
                hone.logger.info(
                    "[Evaluator] backfill: no latest window, nothing to do"
                )
                return
            hi = latest
        if lo > hi:
            lo, hi = hi, lo
        if self.is_master:
            hone.logger.info(
                f"[Evaluator] backfill window range [{lo}, {hi}]"
            )
        for w in range(lo, hi + 1):
            if self.stop_event.is_set():
                break
            # Master gates on completeness; broadcast.
            ready_local = 0
            if self.is_master:
                try:
                    if await self.ckpt.check_checkpoint_exists(window=w):
                        ready_local = 1
                except Exception:
                    pass
            t = torch.tensor([ready_local], dtype=torch.int64, device=self.device)
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(t, src=0)
            if int(t.item()) == 0:
                if self.is_master:
                    hone.logger.info(
                        f"[Evaluator] backfill skip window {w}: not complete"
                    )
                continue
            await self.evaluate_window(w)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    async def run(self) -> None:
        if self.is_master:
            hone.logger.info(
                f"[Evaluator] run() started; api_base={self.api_base} "
                f"interval={self.eval_interval}s tasks={[t.name for t in self.tasks]}"
            )

        await self.backfill()
        if self.config.no_follow:
            if self.is_master:
                hone.logger.info("[Evaluator] --no-follow set; exiting")
            return

        while not self.stop_event.is_set():
            window = await self._poll_latest_window()
            if window is None:
                await asyncio.sleep(self.eval_interval)
                continue
            try:
                await self.evaluate_window(window)
            except Exception:
                hone.logger.exception(
                    f"[Evaluator] evaluate_window({window}) raised"
                )
            await asyncio.sleep(self.eval_interval)

    async def main(self) -> None:
        """Entry point that wires SIGINT/SIGTERM into ``stop_event`` so
        a ``pm2 stop`` shuts the loop down cleanly between iterations.
        """
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.stop_event.set)
            except NotImplementedError:
                # add_signal_handler isn't supported on Windows / some
                # subprocess contexts; fall back to default handling.
                pass
        await self.run()


def _iso(ts: float) -> str:
    """Format a Unix timestamp as ISO-8601 in UTC."""
    import datetime as _dt

    return _dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    e = Evaluator()
    asyncio.run(e.main())


if __name__ == "__main__":
    main()
