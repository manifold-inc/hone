"""Trainer — handles local training with standard cross-entropy loss."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import time
from contextlib import nullcontext
from typing import Iterable

import torch
import torch.nn as nn
from torch import autocast
from torch.distributed.tensor import DTensor as DT
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader

import hone
from hone.distributed import dist_helper
from hone.loss import compute_loss
from hone.model import LoopLM, LoopLMConfig
from hone.muon import Muon, SingleDeviceMuonWithAuxAdam
from neurons.base_node import CPU_COUNT


class Trainer:
    """Manages model creation, optimizers, and the inner training loop."""

    def __init__(self):
        self.inner_scheduler_step_count = 0

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    def set_dataloader(self, validator: bool = False) -> None:
        self.dataset = self.dataset_manager.active_dataset

        max_steps = getattr(self.hparams, "max_inner_steps", None) or self.hparams.inner_steps
        # Miners draw from a deterministic max_inner_steps-sized pool and
        # may stop early for window headroom. Validators must sample/evaluate
        # from the same pool; using inner_steps here makes the sample digest
        # and "own data" eval silently cover a different distribution.
        pool_steps = max_steps

        shared_args = dict(
            dataset=self.dataset,
            uid=self.uid,
            window=self.current_window,
            steps_per_window=pool_steps,
            micro_bs=self.hparams.micro_batch_size,
            rank=self.rank,
            world_size=self.world_size,
        )

        if validator:
            SamplerClass = hone.EvalSampler
            kwargs = shared_args | dict(
                batch_size=self.hparams.target_batch_size,
                validation_bs=self.hparams.validator_sample_micro_bs
                * self.hparams.micro_batch_size,
            )
        else:
            SamplerClass = hone.MinerSampler
            kwargs = shared_args | dict(
                micro_bs=self.hparams.micro_batch_size,
                batch_size=self.hparams.batch_size,
                target_batch_size=self.hparams.target_batch_size,
            )

        self.sampler = SamplerClass(**kwargs)

        # Read DataLoader knobs from hparams.dataloader (dict or
        # SimpleNamespace), falling back to the legacy single-thread
        # defaults when the section is absent. With ``num_workers > 0``
        # plus ``pin_memory=True`` plus ``prefetch_factor`` the loader
        # tokenizes microbatches in worker processes and keeps a queue
        # of pinned-CPU tensors ready for H2D, removing the synchronous
        # ``next(loader_iter)`` bounce on the inner loop's main thread
        # (see ``_fetch_microbatch`` ~ trainer.py:1292).
        dl_cfg_raw = getattr(self.hparams, "dataloader", None) or {}
        if hasattr(dl_cfg_raw, "__dict__"):
            dl_cfg = vars(dl_cfg_raw)
        elif isinstance(dl_cfg_raw, dict):
            dl_cfg = dl_cfg_raw
        else:
            dl_cfg = {}
        num_workers = int(dl_cfg.get("num_workers", 0))
        pin_memory = bool(dl_cfg.get("pin_memory", False))
        persistent_workers = bool(
            dl_cfg.get("persistent_workers", False)
        ) and num_workers > 0
        prefetch_factor_cfg = dl_cfg.get("prefetch_factor", None)
        # ``prefetch_factor`` is only valid when num_workers > 0.
        loader_kwargs: dict = dict(
            dataset=self.dataset,
            sampler=self.sampler,
            batch_size=self.hparams.micro_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        if num_workers > 0 and prefetch_factor_cfg is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor_cfg)

        self.loader = DataLoader(**loader_kwargs)
        hone.logger.info(
            f"[Run] dataset + sampler ready "
            f"(num_workers={num_workers}, pin_memory={pin_memory}, "
            f"persistent_workers={persistent_workers}, "
            f"prefetch_factor={prefetch_factor_cfg if num_workers > 0 else 'n/a'})"
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def get_expected_params(self) -> set[str]:
        """Set of compressed-payload keys we expect each peer to upload.

        Keyed by canonical (wrapper-stripped) param names so peers and
        validator agree regardless of compile / FSDP / AC wrapping order.
        """
        canon = hone.canonical_param_names(self.model)
        expected = set()
        for name, _ in self.model.named_parameters():
            cname = canon.get(name, name)
            expected.add(cname + "idxs")
            expected.add(cname + "vals")
            expected.add(cname + "quant_params")
        return expected

    def init_model(self, validator=False, meta=False):
        config: LoopLMConfig = self.hparams.model_config

        if meta:
            # Meta-init path: build LoopLM on meta -> wrap with FSDP2
            # (returns DTensor on meta). Caller (Miner/Validator) is
            # responsible for ``to_empty(device)`` and then
            # ``init_weights()`` once tensors have real storage.
            with torch.device("meta"):
                self.model = LoopLM(config)

            self._apply_activation_checkpointing()
            if self.world_size > 1 and dist_helper.is_distributed():
                self._apply_fsdp()
            self._apply_torch_compile()
        else:
            # Eager-init path: real weights allocated immediately on `self.device`.
            self.model = LoopLM(config)
            self.model.init_weights()
            self.model.to(self.device)

            self._apply_activation_checkpointing()
            if self.world_size > 1 and dist_helper.is_distributed():
                self._apply_fsdp()
            self._apply_torch_compile()

        self.expected_compressed_params = self.get_expected_params()
        self.tokenizer = self.hparams.tokenizer

    def _apply_torch_compile(self) -> None:
        """Compile the model with ``torch.compile``.

        Reads ``hparams.fsdp.compile`` for the master switch (kept on
        the existing fsdp sub-namespace for back-compat).

        ``HONE_DISABLE_TORCH_COMPILE=1`` env var (also honoured by
        Muon's Newton-Schulz, see hone/src/hone/muon/muon_fsdp2.py)
        forces this off regardless of the hparam. The intended use
        is the validator process, whose ``evaluate_model`` forward
        path triggers Inductor codegen on the first call: if gcc /
        Triton's runtime build can't link (broken toolchain, missing
        headers, ...) the eval crashes with an unhandled ``InductorError``
        and there's no fallback. Validator gets ~no perf benefit from
        compile (forward only, runs N times per window per peer), so
        it's the safe one to opt out.
        """
        if os.environ.get("HONE_DISABLE_TORCH_COMPILE", "0") == "1":
            hone.logger.info(
                "[Model] torch.compile skipped (HONE_DISABLE_TORCH_COMPILE=1)"
            )
            return
        fsdp_cfg = getattr(self.hparams, "fsdp", None) or {}
        if isinstance(fsdp_cfg, dict):
            do_compile = fsdp_cfg.get("compile", False)
        else:
            do_compile = getattr(fsdp_cfg, "compile", False)
        if not do_compile:
            return

        self.model = torch.compile(self.model)
        hone.logger.info("[Model] torch.compile applied (full model)")

    def _apply_activation_checkpointing(self) -> None:
        """Wrap each transformer ``Block`` in a checkpoint wrapper.

        Activation checkpointing (AC) trades compute for memory: each
        wrapped module's forward pass is re-run during backward instead
        of stashing all intermediates. For the 8B-A1B MoE config the
        ``torch._grouped_mm`` activations dominate per-layer memory and
        AC frees enough room to grow ``micro_batch_size`` / ``batch_size``.

        Modes (``hparams.fsdp.activation_checkpoint``):
        - ``"none"`` / falsy / missing: no AC (legacy behaviour).
        - ``"selective"``: wrap each transformer block. Recommended.
        - ``"full"``: same wrap path here; reserved for a future
          finer-grained policy that also checkpoints inside the block.

        Must run *before* ``_apply_fsdp`` so AC sees plain ``nn.Module``
        children, not the FSDP-composable variant which intercepts
        forward via its own hooks.
        """
        mode = (
            getattr(self.hparams.fsdp, "activation_checkpoint", None)
            if hasattr(self.hparams, "fsdp")
            else None
        )
        if not mode or str(mode).lower() in ("none", "false", "off"):
            return

        try:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointImpl,
                checkpoint_wrapper,
            )
        except ImportError as e:
            hone.logger.warning(
                f"[Model] activation_checkpoint={mode} requested but the "
                f"checkpoint_wrapper API is unavailable ({e!r}); skipping."
            )
            return

        def _wrap(layer):
            return checkpoint_wrapper(
                layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            )

        wrapped = 0
        for i, layer in enumerate(self.model.layers):
            self.model.layers[i] = _wrap(layer)
            wrapped += 1

        hone.logger.info(
            f"[Model] activation checkpointing applied to {wrapped} "
            f"transformer blocks (mode={mode})"
        )

    def _apply_fsdp(self):
        """Apply FSDP2 wrapping at the TransformerBlock level.

        ``fully_shard`` uses its default world mesh, equivalent to a
        1-D mesh of all ranks.
        """
        try:
            from torch.distributed.fsdp import fully_shard
        except ImportError:
            from torch.distributed._composable.fsdp import fully_shard

        fsdp_kwargs: dict = {}

        # Mixed-precision policy: keep params + computation in
        # ``param_dtype`` (default bf16) but reduce gradients in fp32
        # for stability. Halves the per-layer all-gather wire bandwidth
        # versus the previous accidental fp32 default and removes a
        # redundant fp32->bf16 cast inside autocast. Reads
        # ``hparams.fsdp.mixed_precision`` (set in hparams.json); falls
        # back to no policy when the hparam is missing or null so older
        # configs keep their existing behaviour.
        mp_name = (
            getattr(self.hparams.fsdp, "mixed_precision", None)
            if hasattr(self.hparams, "fsdp")
            else None
        )
        if mp_name:
            try:
                from torch.distributed.fsdp import MixedPrecisionPolicy
            except ImportError:
                from torch.distributed._composable.fsdp import (
                    MixedPrecisionPolicy,
                )
            param_dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            param_dtype = param_dtype_map.get(str(mp_name).lower())
            if param_dtype is None:
                hone.logger.warning(
                    f"[Model] fsdp.mixed_precision={mp_name!r} not "
                    "recognised; skipping MixedPrecisionPolicy."
                )
            else:
                fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                    param_dtype=param_dtype,
                    reduce_dtype=torch.float32,
                )
                hone.logger.info(
                    f"[Model] FSDP MixedPrecisionPolicy: "
                    f"param_dtype={param_dtype}, reduce_dtype=float32"
                )

        for layer in self.model.layers:
            fully_shard(layer, **fsdp_kwargs)
        fully_shard(self.model, **fsdp_kwargs)

        hone.logger.info(
            f"[Model] FSDP2 applied, world_size={self.world_size}"
        )

    # ------------------------------------------------------------------
    # Optimizers & Schedulers
    # ------------------------------------------------------------------
    def init_optimizers_schedulers(self, validator=False):
        self.lr = float(self.hparams.outer_learning_rate)
        # ``nesterov=True`` with ``momentum=0.9`` was the original
        # DiLoCo/DeMo recipe but it amplifies compression-noise in the
        # outer-grad velocity buffer (we observed outer_grad L2 growing
        # 1.3 -> 5.9 -> 11.7 -> 19.6 -> 26.7 across consecutive outer
        # steps with the old defaults, with corresponding loss spikes
        # at every window boundary). The Nesterov look-ahead step adds
        # another ``+momentum * grad`` on top of the regular
        # momentum-corrected step, ~1.5x larger updates than plain SGD
        # momentum. Make it a hparam so we can flip back if needed.
        # ``foreach=True`` collapses the per-tensor SGD update into a
        # single ``torch._foreach_*`` kernel pass over the param list,
        # which on MoE 8B-A1B (~7100 trainable tensors) eliminates
        # 5-15s of Python-loop overhead per outer step. Works with
        # DTensor shards in PyTorch >= 2.7 (the foreach ops dispatch
        # through DTensor's __torch_dispatch__ to per-shard kernels);
        # ``hone/pyproject.toml`` pins ``torch>=2.7.1``.
        self.outer_optimizer = SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.hparams.outer_momentum,
            nesterov=bool(getattr(self.hparams, "outer_nesterov", True)),
            foreach=True,
        )
        self.outer_scheduler = self._build_outer_scheduler()
        self.inner_optimizer = self._build_inner_optimizer(validator)
        self.inner_scheduler = self._build_inner_scheduler()
        self.inner_scheduler_step_count = 0

        optimizer_config = getattr(self.hparams, "optimizer", {})
        opt_type = optimizer_config.get("type", "adamw").lower()
        opt_cfg = optimizer_config.get(opt_type, {})
        sched_cfg = opt_cfg.get("scheduler", {})
        self.warmup_inner_steps = sched_cfg.get("warmup_inner_steps", 0)
        self.warmup_steps_taken = 0

        hone.logger.info("[Init] optimizers & schedulers constructed")

    def _build_outer_scheduler(self) -> lr_scheduler.LRScheduler | None:
        """Optional cosine schedule on the outer SGD learning rate.

        Default is ``"constant"`` (no scheduler) for back-compat, so any
        existing config that doesn't set ``outer_lr_schedule`` keeps its
        previous behaviour. With ``"cosine"`` the outer LR anneals from
        the configured starting ``outer_learning_rate`` down to
        ``outer_lr * outer_lr_min_factor`` over ``outer_lr_t_max`` outer
        steps -- one outer step per successful gather/window. This pairs
        with outer-grad clipping to slow the late-run "spike-and-recover"
        pattern as the model gets closer to convergence.
        """
        kind = str(getattr(self.hparams, "outer_lr_schedule", "constant"))
        if kind == "constant":
            return None
        if kind != "cosine":
            hone.logger.warning(
                f"[Init] unknown outer_lr_schedule={kind!r}; "
                "falling back to constant LR"
            )
            return None

        t_max = int(getattr(self.hparams, "outer_lr_t_max", 1000))
        min_factor = float(
            getattr(self.hparams, "outer_lr_min_factor", 0.1)
        )
        eta_min = float(self.lr) * max(0.0, min_factor)
        sched = lr_scheduler.CosineAnnealingLR(
            self.outer_optimizer, T_max=t_max, eta_min=eta_min
        )
        hone.logger.info(
            f"[Init] outer LR schedule: cosine "
            f"lr={self.lr} -> eta_min={eta_min:.5g} over {t_max} outer steps"
        )
        return sched

    def _build_inner_scheduler(self):
        optimizer_config = getattr(self.hparams, "optimizer", {})
        opt_type = optimizer_config.get("type", "adamw").lower()
        opt_cfg = optimizer_config.get(opt_type, {})
        sched_cfg = opt_cfg.get("scheduler", {})

        default_lr = 2e-4 if opt_type == "adamw" else 0.02
        effective_lr = opt_cfg.get("learning_rate", default_lr)

        warmup_steps = sched_cfg.get("warmup_steps", 750)
        t_max = sched_cfg.get("t_max", 20000)
        eta_min_factor = sched_cfg.get("eta_min_factor", 0.1)

        warmup = lr_scheduler.LinearLR(
            self.inner_optimizer, start_factor=1e-6, end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = lr_scheduler.CosineAnnealingLR(
            self.inner_optimizer, T_max=t_max,
            eta_min=effective_lr * eta_min_factor,
        )
        scheduler = lr_scheduler.SequentialLR(
            self.inner_optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
        hone.logger.info(
            f"[Init] {opt_type} scheduler: lr={effective_lr}, "
            f"warmup={warmup_steps}, t_max={t_max}"
        )
        return scheduler

    def should_skip_scheduler_step(self) -> bool:
        """Check if we're in the LR flatten window and should skip scheduler stepping.

        flatten_start_step and flatten_duration are in outer steps (windows),
        but we check against inner_scheduler_step_count which tracks individual
        scheduler steps.
        """
        optimizer_config = getattr(self.hparams, "optimizer", {})
        opt_type = optimizer_config.get("type", "adamw").lower()
        opt_cfg = optimizer_config.get(opt_type, {})
        sched_cfg = opt_cfg.get("scheduler", {})

        flatten_start_window = sched_cfg.get("flatten_start_step", None)
        if flatten_start_window is None or flatten_start_window <= 0:
            return False

        flatten_duration_windows = sched_cfg.get("flatten_duration", 0)
        if flatten_duration_windows <= 0:
            return False

        inner_steps_per_window = self.hparams.inner_steps
        flatten_start_inner = flatten_start_window * inner_steps_per_window
        flatten_end_inner = (
            flatten_start_window + flatten_duration_windows
        ) * inner_steps_per_window

        return (
            flatten_start_inner <= self.inner_scheduler_step_count < flatten_end_inner
        )

    def _build_inner_optimizer(self, validator: bool):
        optimizer_config = getattr(self.hparams, "optimizer", {})
        opt_type = optimizer_config.get("type", "adamw").lower()

        if validator:
            dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
            opt_cfg = optimizer_config.get(opt_type, {})
            default_lr = 2e-4 if opt_type == "adamw" else 0.02
            return torch.optim.SGD([dummy], lr=opt_cfg.get("learning_rate", default_lr))

        if opt_type == "adamw":
            cfg = optimizer_config.get("adamw", {})
            base_lr = cfg.get("learning_rate", 2e-4)
            base_wd = cfg.get("weight_decay", 0.1)
            betas = tuple(cfg.get("betas", [0.9, 0.95]))
            eps = cfg.get("eps", 1e-8)

            params = [p for p in self.model.parameters() if p.requires_grad]
            return torch.optim.AdamW(
                params,
                lr=base_lr,
                weight_decay=base_wd,
                betas=betas,
                eps=eps,
                fused=True,
                foreach=False,
            )

        if opt_type == "muon":
            cfg = optimizer_config.get("muon", {})
            muon_lr = cfg.get("learning_rate", 0.02)

            is_fsdp = any(isinstance(p, DT) for p in self.model.parameters())

            hidden_2d, embed, scalar, head = [], [], [], []
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.ndim >= 2 and "embed" not in name and "lm_head" not in name:
                    hidden_2d.append(p)
                elif "embed" in name:
                    embed.append(p)
                elif "lm_head" in name:
                    head.append(p)
                else:
                    scalar.append(p)

            wd = self.hparams.weight_decay
            adam_groups = []
            if head:
                adam_groups.append(dict(
                    params=head, lr=muon_lr * cfg.get("head_lr_scale", 0.5),
                    weight_decay=wd,
                ))
            if embed:
                adam_groups.append(dict(
                    params=embed, lr=muon_lr * cfg.get("embed_lr_scale", 0.5),
                    weight_decay=wd,
                ))
            if scalar:
                adam_groups.append(dict(
                    params=scalar, lr=muon_lr * cfg.get("scalar_lr_scale", 0.2),
                    weight_decay=wd,
                ))
            adam_groups = [
                {**g, "betas": (0.9, 0.95), "eps": 1e-8, "use_muon": False}
                for g in adam_groups
            ]

            muon_group = dict(
                params=hidden_2d,
                lr=muon_lr,
                momentum=cfg.get("momentum", 0.95),
                weight_decay=cfg.get("weight_decay", 0.01),
                use_muon=True,
            )
            if is_fsdp:
                muon_group["rms_scale"] = cfg.get("rms_scale", True)
                muon_group["nesterov"] = cfg.get("nesterov", True)
                return Muon(adam_groups + [muon_group])
            else:
                return SingleDeviceMuonWithAuxAdam(adam_groups + [muon_group])

        raise ValueError(f"Unknown optimizer type: {opt_type}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    async def evaluate_model(
        self,
        model: nn.Module,
        loader: Iterable[torch.Tensor],
    ) -> tuple[float, int]:
        device = next(model.parameters()).device
        total_loss = 0.0
        n_batches = 0

        with torch.inference_mode():
            model.eval()
            for i, batch in enumerate(loader):
                if batch is None or len(batch) == 0:
                    continue

                input_ids = (
                    batch.to(device, dtype=torch.long, non_blocking=True)
                    if isinstance(batch, torch.Tensor)
                    else torch.tensor(batch, dtype=torch.long, device=device)
                )

                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]
                labels[:, -1] = self.tokenizer.pad_token_id
                labels = torch.where(
                    labels == self.tokenizer.pad_token_id, -100, labels
                )

                has_valid = (labels != -100).any().item()
                all_ok = dist_helper.all_ok(has_valid, device)
                if not all_ok:
                    del input_ids, labels
                    continue

                with autocast(device_type=device.type, dtype=torch.bfloat16):
                    model_output = model(input_ids)

                if isinstance(model_output, tuple):
                    logits = model_output[0]
                else:
                    logits = model_output

                loss = compute_loss(logits, labels)
                total_loss += loss.item()
                n_batches += 1
                del input_ids, labels, logits
                torch.cuda.empty_cache()
                await asyncio.sleep(0)

        if self.world_size > 1 and dist_helper.is_distributed():
            total_loss = dist_helper.ddp_reduce(total_loss, device=device)
            n_batches = int(dist_helper.ddp_reduce(n_batches, device=device))

        return total_loss, n_batches

    # ------------------------------------------------------------------
    # Optimizer offload / prefetch
    # ------------------------------------------------------------------
    def _iter_inner_opt_state_tensors(self):
        opt = getattr(self, "inner_optimizer", None)
        if opt is None:
            return
        for _p, s in opt.state.items():
            if not isinstance(s, dict):
                continue
            for k, v in list(s.items()):
                if torch.is_tensor(v):
                    yield s, k, v

    def offload_inner_optimizer_states(self, *, log: bool = True) -> None:
        if not getattr(self.hparams, "offload_optimizer_states", True):
            return
        if getattr(self, "_inner_opt_offloaded", False):
            return
        moved = 0
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize(getattr(self, "device", None))
        for s, k, v in self._iter_inner_opt_state_tensors():
            if v.device.type == "cpu":
                if not v.is_pinned():
                    s[k] = v.pin_memory()
                continue
            dst = torch.empty_like(v, device="cpu", pin_memory=True)
            dst.copy_(v, non_blocking=True)
            s[k] = dst
            moved += v.element_size() * v.numel()
        if torch.cuda.is_available():
            torch.cuda.synchronize(getattr(self, "device", None))
            torch.cuda.empty_cache()
        self._inner_opt_offloaded = True
        if log:
            hone.logger.info(
                f"[OptOffload] → CPU pinned | {moved / 1e6:.1f} MB "
                f"in {time.time() - t0:.3f}s"
            )

    def prefetch_inner_optimizer_states(self, *, log: bool = True) -> None:
        if not getattr(self.hparams, "offload_optimizer_states", True):
            return
        if not getattr(self, "_inner_opt_offloaded", False):
            return
        device = getattr(self, "device", torch.device("cuda"))
        moved = 0
        t0 = time.time()
        for s, k, v in self._iter_inner_opt_state_tensors():
            if v.device == device:
                continue
            dst = torch.empty_like(v, device=device)
            dst.copy_(v, non_blocking=True)
            s[k] = dst
            moved += v.element_size() * v.numel()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        self._inner_opt_offloaded = False
        if log:
            hone.logger.info(
                f"[OptOffload] ← GPU | {moved / 1e6:.1f} MB "
                f"in {time.time() - t0:.3f}s"
            )

    @staticmethod
    def _coerce_reset_factor(value) -> float:
        """Normalise ``reset_inner_optimizer_per_window`` to a float in
        ``[0.0, 1.0]``.

        - ``False`` / ``None`` / ``0``      -> ``0.0`` (no-op)
        - ``True`` / ``1`` / ``1.0``        -> ``1.0`` (full clear)
        - any ``0.0 < x < 1.0``             -> ``x``   (soft decay)
        - out-of-range floats are clamped, non-numerics fall back to
          ``0.0`` with a warning so a typo in the hparam can't silently
          enable a hard reset that the operator didn't ask for.
        """
        if value is None or value is False:
            return 0.0
        if value is True:
            return 1.0
        try:
            f = float(value)
        except (TypeError, ValueError):
            hone.logger.warning(
                f"[InnerOpt] reset_inner_optimizer_per_window={value!r} "
                "not a bool/float; treating as 0.0 (no reset)"
            )
            return 0.0
        if f < 0.0:
            return 0.0
        if f > 1.0:
            return 1.0
        return f

    def reset_inner_optimizer_states(self, *, log: bool = True) -> None:
        """Reset (or soft-decay) inner-optimizer per-param state at the
        start of a window.

        ``reset_inner_optimizer_per_window`` semantics:

        - ``0.0`` / ``False``: no-op. Caller should ``prefetch`` instead.
        - ``1.0`` / ``True`` : full clear via ``optimizer.state.clear()``.
          Adam / Muon / SGD lazily re-allocate zero buffers on the next
          ``.step()`` call, on the param's own device.
        - ``0.0 < x < 1.0``  : soft decay. Iterate every state tensor
          and ``mul_(x)`` in place. Direction of momentum is preserved,
          its magnitude is attenuated. This is the recommended setting
          for live training -- it removes the "stale momentum applied
          to a discontinuous outer-stepped model" overshoot without
          throwing away the inner-loop's accumulated learning signal.

        The manual-warmup counter (``warmup_steps_taken``) is reset only
        when ``reset_warmup_per_window`` is true. Decoupling the two
        means we don't have to pay the 30-inner-step (k+1)/N LR ramp
        every window once the optimizer is calibrated. This is safe in
        combination with outer-grad clipping (which bounds the
        post-outer-step jump that the warmup was protecting against).
        """
        if not getattr(self, "inner_optimizer", None):
            return

        factor = self._coerce_reset_factor(
            getattr(self.hparams, "reset_inner_optimizer_per_window", False)
        )
        if factor <= 0.0:
            return

        t0 = time.time()
        n_states = sum(1 for _ in self._iter_inner_opt_state_tensors())
        if factor >= 1.0:
            # Hard clear: Muon / Adam will lazily re-allocate fresh
            # zero buffers on the next ``.step()`` call, on the
            # param's own device. Discard whatever's offloaded.
            self.inner_optimizer.state.clear()
            mode = "clear"
        else:
            # Soft decay: we must touch every state tensor in place.
            # If they're currently offloaded to pinned CPU (the steady
            # state at the end of every window) we have to bring them
            # back to GPU first, otherwise the next ``optimizer.step()``
            # will run ``buf.lerp_(grad, ...)`` with ``buf`` on CPU
            # and ``grad`` on cuda -- raising
            #   RuntimeError: Expected all tensors to be on the same
            #   device, but found at least two devices, cuda:N and cpu!
            # ``prefetch_inner_optimizer_states`` is a no-op when
            # ``_inner_opt_offloaded`` is False, so calling it
            # unconditionally is safe and cheap.
            self.prefetch_inner_optimizer_states(log=False)
            for _p, s in self.inner_optimizer.state.items():
                if not isinstance(s, dict):
                    continue
                for v in s.values():
                    if torch.is_tensor(v) and v.is_floating_point():
                        v.mul_(factor)
            mode = f"decay x{factor:.3g}"

        warmup_was_reset = False
        if getattr(self.hparams, "reset_warmup_per_window", True):
            self.warmup_steps_taken = 0
            warmup_was_reset = True

        # State now lives on GPU (either freshly prefetched + decayed,
        # or freshly cleared and about to be lazily re-allocated on
        # GPU by the next .step()). Mark not-offloaded so the next
        # prefetch is correctly a no-op.
        self._inner_opt_offloaded = False

        if log and getattr(self, "is_master", True):
            warmup_msg = " + warmup counter" if warmup_was_reset else ""
            hone.logger.info(
                f"[InnerOpt] {mode} {n_states} state tensors{warmup_msg} "
                f"in {time.time() - t0:.3f}s"
            )

    # ------------------------------------------------------------------
    # Inner training loop
    # ------------------------------------------------------------------
    async def inner_steps(
        self,
        loader: DataLoader,
        step_window: int,
        null_round: bool = False,
    ) -> dict:
        if not hasattr(self, "loop"):
            self.loop = asyncio.get_running_loop()
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=CPU_COUNT
            )
            self.loop.set_default_executor(self.executor)

        # The outer step creates a parameter discontinuity that stale
        # inner-optimizer momentum (calibrated against the *pre-outer-
        # step* params) doesn't expect. Applying it produces a one-step
        # overshoot that shows up as a sharp loss spike right after
        # each window flip (loss=7.69 -> 11.75 in a single inner step
        # in the original trace). Three options are gated by the
        # ``reset_inner_optimizer_per_window`` hparam (see
        # ``_coerce_reset_factor``): 0 = preserve state and prefetch
        # from CPU; 1 = full clear; 0<x<1 = soft decay (recommended,
        # keeps direction info, attenuates stale magnitude).
        reset_factor = self._coerce_reset_factor(
            getattr(self.hparams, "reset_inner_optimizer_per_window", False)
        )
        if reset_factor > 0.0:
            self.reset_inner_optimizer_states()
        else:
            self.prefetch_inner_optimizer_states()
        self.inner_optimizer.zero_grad()

        total_loss: float = 0.0
        batch_count: int = 0
        batch_tokens: int = 0
        accum_batch_size: int = 0
        window_entry_loss: float = 0.0
        global_tokens: int = 0
        global_loss_sum: float = 0.0
        local_tokens_sum: int = 0
        local_loss_sum: float = 0.0
        inner_step_count: int = 0

        # Reserve enough wall-clock time before the next chain-window
        # starts to compress + cross-rank-merge + S3 PUT the gradient. We
        # estimate "blocks left in this window" via the chain listener's
        # ``self.current_block`` (no extra RPC needed) and convert with
        # Bittensor's nominal block time. The check is broadcast by the
        # MAX reduction in step 5 so all ranks bail in lockstep even if
        # only one notices first.
        headroom_s = float(
            getattr(self.hparams, "window_flush_headroom_seconds", 0) or 0
        )
        blocks_per_window = int(self.hparams.blocks_per_window)
        next_window_block = (step_window + 1) * blocks_per_window
        # Bittensor's nominal block time is 12s; keep the constant local
        # to avoid reaching into chain hparams from a hot loop and to
        # make the back-of-envelope math obvious to anyone reading.
        approx_block_seconds = 12.0

        loader_iter = iter(loader)

        # Helper: fetch + tensorise + sanity-check one microbatch from the
        # dataloader. Returns ``(input_ids, labels, local_bs, tokens, ok)``
        # where ``ok=False`` flags either StopIteration on every rank or a
        # batch with no valid (non-pad) labels. Centralising this avoids
        # duplicating the prep logic between the non-PP per-microbatch
        # path and the PP "fetch grad_accum microbatches" path.
        async def _fetch_microbatch():
            try:
                batch = await self.loop.run_in_executor(None, next, loader_iter)
                local_has_batch = True
            except StopIteration:
                local_has_batch = False
                batch = None

            if self.world_size > 1:
                if not dist_helper.should_continue(local_has_batch, self.device):
                    return None
                if not local_has_batch:
                    return "skip"

            input_ids = (
                batch.to(self.device, dtype=torch.long, non_blocking=True)
                if isinstance(batch, torch.Tensor)
                else torch.tensor(batch, dtype=torch.long, device=self.device)
            )
            local_bs = len(batch)
            tokens_this = input_ids.numel()

            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = self.tokenizer.pad_token_id
            labels = torch.where(
                labels == self.tokenizer.pad_token_id, -100, labels
            )

            has_valid = (labels != -100).any().item()
            if not dist_helper.all_ok(has_valid, self.device):
                del input_ids, labels
                return "skip"

            return input_ids, labels, local_bs, tokens_this

        while not self.stop_event.is_set():
            # 1. Fetch one microbatch.
            corrected_accum = max(self.sampler.grad_accum_steps, 1)

            fetched = await _fetch_microbatch()
            if fetched is None:
                break
            if fetched == "skip":
                continue
            input_ids, labels, local_bs, tokens_this = fetched

            accum_batch_size += local_bs
            batch_tokens += tokens_this
            local_tokens_sum += tokens_this

            final_micro = (batch_count + 1) % corrected_accum == 0

            with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                model_output = self.model(input_ids)

            if isinstance(model_output, tuple):
                logits, aux_loss = model_output
            else:
                logits = model_output
                aux_loss = None

            calculated_loss = compute_loss(logits, labels)
            if aux_loss is not None:
                calculated_loss = calculated_loss + aux_loss

            loss = calculated_loss / self.sampler.grad_accum_steps
            loss_item = calculated_loss.detach().item()

            if (
                hasattr(self.model, "no_sync")
                and self.world_size > 1
                and not final_micro
            ):
                sync_ctx = self.model.no_sync()
            else:
                sync_ctx = nullcontext()

            with sync_ctx:
                self.scaler.scale(loss).backward()

            total_loss += loss_item
            local_loss_sum += loss_item
            batch_count += 1
            window_changed = self.current_window != step_window

            # 4. Optimizer step
            step_now = final_micro or window_changed
            if self.world_size > 1:
                from torch.distributed import ReduceOp
                step_now = bool(
                    dist_helper.ddp_reduce(
                        int(step_now), op=ReduceOp.MAX, device=self.device
                    )
                )

            if step_now:
                global_tokens_step = int(
                    dist_helper.ddp_reduce(local_tokens_sum, device=self.device)
                )
                global_loss_step = dist_helper.ddp_reduce(
                    local_loss_sum, device=self.device
                )
                global_tokens += global_tokens_step
                global_loss_sum += global_loss_step

                from torch.distributed import ReduceOp
                log_loss = dist_helper.ddp_reduce(
                    loss_item, op=ReduceOp.AVG, device=self.device
                )

                if not null_round:
                    # Manual warm-up: scale LR by (k+1)/N for the first
                    # warmup_inner_steps optimizer steps so Adam's running
                    # estimates can stabilise before full LR kicks in.
                    original_lrs = []
                    in_manual_warmup = (
                        self.warmup_steps_taken < self.warmup_inner_steps
                    )
                    if in_manual_warmup:
                        warmup_scale = (
                            self.warmup_steps_taken + 1
                        ) / self.warmup_inner_steps
                        for pg in self.inner_optimizer.param_groups:
                            original_lrs.append(pg["lr"])
                            pg["lr"] *= warmup_scale

                    self.scaler.unscale_(self.inner_optimizer)

                    # Skip step if gradients contain NaN/Inf to prevent
                    # permanent model corruption from bf16 overflow in the
                    # 96-layer effective backward path.
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        getattr(self.hparams, "max_grad_norm", 1.0),
                    )
                    # Under FSDP2 with ``dp_shard > 1``, ``clip_grad_norm_``
                    # returns a ``DTensor`` scalar whose ``__format__``
                    # rejects spec strings like ``{x:.4f}`` with
                    # ``TypeError: unsupported format string passed to
                    # DTensor.__format__``. Replicate the scalar onto
                    # the local rank so the value is a regular
                    # ``torch.Tensor`` for ``isfinite`` + every f-string
                    # below + the dashboard reporter. Cheap (single-
                    # element all-gather, already amortized by the
                    # surrounding all-reduce that produced the norm).
                    if hasattr(grad_norm, "full_tensor"):
                        grad_norm = grad_norm.full_tensor()
                    if torch.isfinite(grad_norm):
                        self.scaler.step(self.inner_optimizer)
                    else:
                        hone.logger.warning(
                            f"NaN/Inf grad_norm ({grad_norm:.4f}) at inner "
                            f"step {inner_step_count + 1} — skipping optimizer step"
                        )
                    self.scaler.update()

                    if in_manual_warmup:
                        for i, pg in enumerate(self.inner_optimizer.param_groups):
                            pg["lr"] = original_lrs[i]
                        self.warmup_steps_taken += 1

                    if not self.should_skip_scheduler_step():
                        self.inner_scheduler.step()
                    self.inner_scheduler_step_count += 1
                else:
                    self.scaler.update()

                self.inner_optimizer.zero_grad(set_to_none=True)
                inner_step_count += 1
                local_tokens_sum = 0
                local_loss_sum = 0

                accum_batch_size = int(
                    dist_helper.ddp_reduce(accum_batch_size, device=self.device)
                )
                if self.is_master:
                    step_msg = (
                        f"Inner Step {inner_step_count}, "
                        f"Batch {batch_count}, loss: {log_loss:.4f}, "
                        f"accum: {accum_batch_size}/{self.hparams.batch_size}"
                    )
                    if not null_round and self.warmup_steps_taken <= self.warmup_inner_steps:
                        sched_lr = self.inner_scheduler.get_last_lr()[0] if hasattr(self.inner_scheduler, "get_last_lr") else None
                        step_msg += (
                            f" | warmup {self.warmup_steps_taken}/{self.warmup_inner_steps}"
                            f", sched_lr={sched_lr:.2e}"
                            f", grad_norm={grad_norm:.4f}"
                            f", scaler_scale={self.scaler.get_scale():.0f}"
                        )
                    hone.logger.info(step_msg)

                    if hasattr(self, "dashboard_reporter"):
                        current_lr = (
                            self.inner_scheduler.get_last_lr()[0]
                            if hasattr(self.inner_scheduler, "get_last_lr")
                            else None
                        )
                        asyncio.create_task(
                            self.dashboard_reporter.report_inner_step(
                                window=step_window,
                                inner_step=inner_step_count,
                                global_step=getattr(self, "global_step", 0),
                                loss=float(log_loss),
                                batch_size=int(accum_batch_size),
                                batch_tokens=int(tokens_this),
                                inner_lr=float(current_lr) if current_lr else None,
                                grad_norm=float(grad_norm) if not null_round and torch.isfinite(grad_norm) else None,
                            )
                        )
                if window_entry_loss == 0.0:
                    total_first = int(
                        dist_helper.ddp_reduce(batch_count, device=self.device)
                    )
                    window_entry_loss = global_loss_sum / total_first
                accum_batch_size = 0

            # 5. Window control
            max_inner = getattr(self.hparams, "max_inner_steps", None) or self.hparams.inner_steps
            # Headroom: stop training if the wall-clock distance to the
            # next chain-window start drops below the buffer reserved for
            # post-train compression + FSDP gather + S3 PUT. Without
            # this the inner loop would happily run one more 12s step
            # past the window flip, then need ~30-40s of post-work, and
            # ship the upload 1-6s after the validator's deadline.
            blocks_left = next_window_block - int(self.current_block)
            seconds_left = blocks_left * approx_block_seconds
            headroom_exhausted = headroom_s > 0 and seconds_left <= headroom_s
            need_sync = (
                window_changed
                or inner_step_count >= max_inner
                or headroom_exhausted
            )
            if self.world_size > 1:
                from torch.distributed import ReduceOp
                global_done = bool(
                    dist_helper.ddp_reduce(
                        int(need_sync), op=ReduceOp.MAX, device=self.device
                    )
                )
            else:
                global_done = need_sync

            if global_done:
                if self.is_master:
                    if headroom_exhausted and not window_changed:
                        hone.logger.info(
                            f"<Headroom exhausted: ~{seconds_left:.0f}s left "
                            f"in window {step_window}, need {headroom_s:.0f}s "
                            "for upload — exiting>"
                        )
                    else:
                        hone.logger.info("<Exhausted window: exiting synchronously>")
                if not null_round:
                    for _ in range(inner_step_count, self.hparams.inner_steps):
                        if not self.should_skip_scheduler_step():
                            self.inner_scheduler.step()
                        self.inner_scheduler_step_count += 1
                break

            await asyncio.sleep(0)

        # 6. Gradient / weight norms
        local_grad_sq = []
        local_weight_sq = []
        for p in self.model.parameters():
            if p.requires_grad:
                local_weight_sq.append((p.norm() ** 2).item())
                if p.grad is not None:
                    local_grad_sq.append((p.grad.norm() ** 2).item())

        total_grad_sq = sum(local_grad_sq) if local_grad_sq else 0.0
        total_weight_sq = sum(local_weight_sq) if local_weight_sq else 0.0

        if self.world_size > 1 and dist_helper.is_distributed():
            total_grad_sq = dist_helper.ddp_reduce(total_grad_sq, device=self.device)
            total_weight_sq = dist_helper.ddp_reduce(total_weight_sq, device=self.device)

        batch_count = int(dist_helper.ddp_reduce(batch_count, device=self.device))

        return {
            "total_loss": global_loss_sum,
            "window_entry_loss": window_entry_loss,
            "batch_count": batch_count,
            "batch_tokens": global_tokens,
            # P2: ``inner_step_count`` is the number of inner optimizer
            # steps actually taken this window (≤ ``hparams.inner_steps``;
            # may be lower under headroom early-exit, may be capped by
            # ``max_inner_steps``). The miner uploads it as ``c_steps`` in
            # ``UploadMetadata`` so the validator can compute the
            # token-weighted aggregation weight ``w = c_tokens *
            # (c_tokens / c_steps)`` per Decoupled DiLoCo Algorithm 2.
            "inner_step_count": inner_step_count,
            "global_grad_norm": total_grad_sq ** 0.5 if total_grad_sq > 0 else 0.0,
            "global_weight_norm": total_weight_sq ** 0.5 if total_weight_sq > 0 else 0.0,
        }

    def outer_step(self, gather_result, log_wandb: bool = False):
        # Lazy-init the auto-clip EMA state so it survives across
        # outer steps within this miner's lifetime. ``outer_step`` only
        # mutates the dict on the source rank (master), but every rank
        # safely owns its own (empty) reference. Toggle and tuning
        # live in hparams; defaults match the legacy fixed-cap
        # behaviour when ``outer_grad_norm_auto`` is missing/false.
        if not hasattr(self, "_outer_auto_clip_state"):
            self._outer_auto_clip_state: dict = {}
        auto_on = bool(getattr(self.hparams, "outer_grad_norm_auto", False))
        # ``hone.neurons.outer_step`` returns
        # ``(fingerprint, outer_step_timings)`` post-P0a. The miner's
        # main loop only consumes the fingerprint today; if/when the
        # miner reporter wants the phase-split timings, plumb the
        # second element through here too.
        fingerprint, _outer_timings = hone.neurons.outer_step(
            self.model,
            self.outer_optimizer,
            gather_result=gather_result,
            transformer=self.transformer,
            compressor=self.compressor,
            xshapes=self.xshapes,
            totalks=self.totalks,
            device=str(self.device),
            is_master=self.is_master,
            world_size=self.world_size,
            use_dct=self.hparams.use_dct,
            wandb_run=self.wandb if self.is_master and log_wandb else None,
            global_step=self.global_step,
            max_grad_norm=getattr(
                self.hparams, "outer_max_grad_norm", None
            ),
            auto_clip_state=(
                self._outer_auto_clip_state if auto_on else None
            ),
            auto_clip_factor=float(
                getattr(self.hparams, "outer_grad_norm_auto_factor", 1.5)
            ),
            auto_clip_ema_decay=float(
                getattr(self.hparams, "outer_grad_norm_ema_decay", 0.95)
            ),
        )
        return fingerprint
