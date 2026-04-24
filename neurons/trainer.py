"""Trainer — handles local training with standard cross-entropy loss."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from contextlib import nullcontext
from typing import Iterable

import torch
import torch.nn as nn
from torch import autocast
from torch.distributed.tensor import DTensor as DT
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader

import torch.distributed as dist

import hone
from hone.distributed import dist_helper
from hone.loss import compute_loss
from hone.model import LoopLM, LoopLMConfig
from hone.muon import Muon, SingleDeviceMuonWithAuxAdam
from hone.pipeline import PipelineStage, PipelineStageBoundary, create_pipeline_stages
from neurons.base_node import CPU_COUNT


class _PipelineStageModel(nn.Module):
    """Wrapper module owning a single pipeline stage's submodules.

    Stage 0 owns ``embed_tokens``; the last stage owns ``norm`` + ``lm_head``;
    every stage owns its ``stage`` (a ``PipelineStage``) and a shared
    ``rotary_emb``. ``init_weights`` mirrors ``LoopLM.init_weights`` but only
    for the submodules that live on this stage, so the same call site works
    after carving.
    """

    def __init__(
        self,
        *,
        stage: PipelineStage,
        embed_tokens: nn.Module | None,
        norm: nn.Module | None,
        lm_head: nn.Module | None,
        rotary_emb: nn.Module,
        config: LoopLMConfig,
        is_first_stage: bool,
        is_last_stage: bool,
    ):
        super().__init__()
        self.stage = stage
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        if norm is not None:
            self.norm = norm
        if lm_head is not None:
            self.lm_head = lm_head
        self.rotary_emb = rotary_emb
        self.config = config
        self.is_first_stage = is_first_stage
        self.is_last_stage = is_last_stage

    def init_weights(self):
        """Initialize this stage's tensors in-place.

        Safe to call after ``to_empty(device)`` for the meta-init flow. This
        only touches the submodules that live on this stage, so embed/norm/
        lm_head are skipped on stages that don't own them.
        """
        if hasattr(self, "embed_tokens"):
            nn.init.normal_(self.embed_tokens.weight)
        self.stage.init_weights()
        if hasattr(self, "norm"):
            nn.init.ones_(self.norm.weight)
        if hasattr(self, "lm_head") and not getattr(
            self.config, "tie_embeddings", True
        ):
            final_std = self.config.dim ** -0.5
            cutoff = 3 * final_std
            nn.init.trunc_normal_(
                self.lm_head.weight,
                mean=0.0,
                std=final_std,
                a=-cutoff,
                b=cutoff,
            )


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
        pool_steps = max_steps if not validator else self.hparams.inner_steps

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
        self.loader = DataLoader(
            dataset=self.dataset,
            sampler=self.sampler,
            batch_size=self.hparams.micro_batch_size,
        )
        hone.logger.info("[Run] dataset + sampler ready")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def get_expected_params(self) -> set[str]:
        expected = set()
        for name, _ in self.model.named_parameters():
            expected.add(name + "idxs")
            expected.add(name + "vals")
            expected.add(name + "quant_params")
        return expected

    def init_model(self, validator=False, meta=False):
        config: LoopLMConfig = self.hparams.model_config
        pp_stages = getattr(self, "pp_degree", 1)
        ranks_in_stage = self.world_size // max(pp_stages, 1)

        if meta:
            # Meta-init path:
            #   build LoopLM on meta -> set up PP groups -> carve to local stage on
            #   meta -> wrap with FSDP2 (returns DTensor on meta).
            # Caller (Miner/Validator) is responsible for `to_empty(device)` and
            # then `init_weights()` on the carved model.
            with torch.device("meta"):
                self.model = LoopLM(config)
            self._pp_setup(pp_stages)
            if pp_stages > 1:
                self._init_pp_model(config, pp_stages, meta=True)

            if ranks_in_stage > 1 and dist_helper.is_distributed():
                self._apply_fsdp()
        else:
            # Eager-init path: real weights allocated immediately on `self.device`.
            self.model = LoopLM(config)
            self.model.init_weights()
            self.model.to(self.device)

            self._pp_setup(pp_stages)
            if pp_stages > 1:
                self._init_pp_model(config, pp_stages, meta=False)

            if ranks_in_stage > 1 and dist_helper.is_distributed():
                self._apply_fsdp()

            fsdp_cfg = getattr(self.hparams, "fsdp", None) or {}
            if isinstance(fsdp_cfg, dict):
                do_compile = fsdp_cfg.get("compile", False)
            else:
                do_compile = getattr(fsdp_cfg, "compile", False)

            if do_compile:
                self.model = torch.compile(self.model)
                hone.logger.info("[Model] torch.compile applied")

        self.expected_compressed_params = self.get_expected_params()
        self.tokenizer = self.hparams.tokenizer

    def _pp_setup(self, pp_stages: int):
        """Compute pipeline parallelism rank assignments.

        Per-stage torchrun model: each torchrun job runs ONE PP stage and
        owns its own NCCL world. ``pp_stage_id`` is set externally by the
        caller (Miner reads it from ``--pp-stage``). All ranks of this
        torchrun belong to the same stage. Cross-stage activation transfer
        runs over the separate :class:`hone.PPTransport` TCP channel,
        completely outside the NCCL process group, which avoids the stream
        contention between PP P2P and FSDP all_gather/reduce_scatter that
        deadlocks the single-torchrun design.
        """
        self.pp_stages = pp_stages
        # Legacy attributes kept None so any stale reference is obvious.
        self.pp_mesh = None
        self.dp_mesh = None
        self.world_mesh = None
        self.pp_stage_group = None
        self.pp_p2p_group_send = None
        self.pp_p2p_group_recv = None
        self.pp_send_rank = -1  # unused under TCP transport
        self.pp_recv_rank = -1  # unused under TCP transport

        if pp_stages <= 1:
            self.pp_stage_id = 0
            self.pp_is_first_stage = True
            self.pp_is_last_stage = True
            self.pp_stage_ranks = list(range(self.world_size))
            self.pp_stage: PipelineStage | None = None
            return

        # ``pp_stage_id`` was set by ``Miner.__init__`` from the ``--pp-stage``
        # CLI flag before ``init_model`` was called. Fall back to 0 only if
        # the caller forgot to set it (defensive).
        stage_id = int(getattr(self, "pp_stage_id", 0))
        if stage_id < 0 or stage_id >= pp_stages:
            raise ValueError(
                f"pp_stage_id={stage_id} outside [0, {pp_stages})"
            )
        self.pp_stage_id = stage_id
        self.pp_is_first_stage = stage_id == 0
        self.pp_is_last_stage = stage_id == pp_stages - 1
        # Within this torchrun, the stage's ranks are simply 0..world_size-1.
        self.pp_stage_ranks = list(range(self.world_size))

        hone.logger.info(
            f"[PP] stage={self.pp_stage_id}/{pp_stages}, "
            f"ranks={self.pp_stage_ranks}, "
            f"world_size={self.world_size}, "
            f"transport=TCP (PPTransport)"
        )

    def _init_pp_model(
        self,
        config: LoopLMConfig,
        pp_stages: int,
        *,
        meta: bool = False,
    ):
        """Partition the full model into pipeline stages and keep only this rank's stage.

        When ``meta=True`` the model parameters are still on the meta device;
        we skip both ``init_weights()`` and ``.to(device)`` so that the caller
        can ``to_empty(device)`` and then call ``self.model.init_weights()``
        once tensors have real storage.
        """
        pipeline_cfg = getattr(self.hparams, "pipeline", None)
        if pipeline_cfg is not None:
            if hasattr(pipeline_cfg, "bottleneck_dim"):
                bottleneck_dim = pipeline_cfg.bottleneck_dim
            else:
                bottleneck_dim = pipeline_cfg.get("bottleneck_dim", 16)
        else:
            bottleneck_dim = 16

        stages = create_pipeline_stages(
            model_layers=self.model.layers,
            num_stages=pp_stages,
            hidden_dim=config.dim,
            bottleneck_dim=bottleneck_dim,
        )

        my_stage = stages[self.pp_stage_id]
        if not meta:
            my_stage.init_weights()

        new_model = _PipelineStageModel(
            stage=my_stage,
            embed_tokens=self.model.embed_tokens if self.pp_is_first_stage else None,
            norm=self.model.norm if self.pp_is_last_stage else None,
            lm_head=self.model.lm_head if self.pp_is_last_stage else None,
            rotary_emb=self.model.rotary_emb,
            config=config,
            is_first_stage=self.pp_is_first_stage,
            is_last_stage=self.pp_is_last_stage,
        )

        del self.model
        if not meta and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not meta:
            self.model = new_model.to(self.device)
        else:
            self.model = new_model
        self.pp_stage = my_stage

        # Note: when meta=True, every parameter still reports `numel()` from
        # its meta shape (storage is unallocated), so this count is accurate
        # for the local stage's parameter footprint either way.
        n_params = sum(p.numel() for p in self.model.parameters())
        hone.logger.info(
            f"[PP] Stage {self.pp_stage_id}: {len(my_stage.layers)} layers, "
            f"{n_params/1e6:.1f}M params, bottleneck_dim={bottleneck_dim}"
            f"{' (meta)' if meta else ''}"
        )

    def _apply_fsdp(self):
        """Apply FSDP2 wrapping at the TransformerBlock level.

        With PP > 1, sharding is constrained to the within-stage ``dp``
        mesh axis so each PP stage's ranks form an independent FSDP group.
        Without PP we let ``fully_shard`` use its default world mesh, which
        is equivalent to a 1-D mesh of all ranks.

        FSDP2 installs its unshard/reshard hooks on each wrapped module's
        ``__call__``. ``_pp_forward_backward`` bypasses the outer model
        forward and calls leaf submodules directly, so those leaves
        (``embed_tokens``, ``lm_head``, the ResBM ``encoder``/``decoder``
        inside each ``PipelineStageBoundary``) need their *own* FSDP units
        or every call fails with "mixed torch.Tensor and DTensor".

        Tiny modules (RMSNorm, ResBM ``IdentityProjection``) carry no
        meaningful parameter footprint, so they stay replicated via
        ``ignored_params`` rather than incurring an extra all-gather.
        """
        try:
            from torch.distributed.fsdp import fully_shard
        except ImportError:
            from torch.distributed._composable.fsdp import fully_shard

        fsdp_kwargs: dict = {}
        if self.dp_mesh is not None:
            fsdp_kwargs["mesh"] = self.dp_mesh

        if self.pp_stages > 1 and self.pp_stage is not None:
            ignored_params: set = set()

            # Shard each transformer block in the local stage.
            for layer in self.pp_stage.layers:
                fully_shard(layer, **fsdp_kwargs)

            # ResBM boundaries: shard the inner encoder/decoder MLPs (the
            # bulk of the boundary params) so direct
            # ``boundary.encoder(x)`` / ``boundary.decoder(c)`` calls
            # trigger the unshard hook. The outer boundary itself stays
            # as a plain nn.Module so ``boundary.encode(...)`` works.
            for boundary in (
                self.pp_stage.input_boundary,
                self.pp_stage.output_boundary,
            ):
                if boundary is None:
                    continue
                fully_shard(boundary.encoder, **fsdp_kwargs)
                fully_shard(boundary.decoder, **fsdp_kwargs)
                # IdentityProjection has no parameters, nothing to ignore.

            # Stage-edge submodules called directly from
            # ``_pp_forward_backward``.
            if self.pp_is_first_stage and hasattr(self.model, "embed_tokens"):
                fully_shard(self.model.embed_tokens, **fsdp_kwargs)
            if self.pp_is_last_stage and hasattr(self.model, "lm_head"):
                fully_shard(self.model.lm_head, **fsdp_kwargs)

            # RMSNorm is one (hidden_dim,) vector; replicate it.
            if self.pp_is_last_stage and hasattr(self.model, "norm"):
                ignored_params.update(self.model.norm.parameters())

            # Outer wrapper: anything not yet wrapped (typically nothing
            # except the ignored norm) inherits a root FSDP hook.
            fully_shard(
                self.model, **fsdp_kwargs, ignored_params=ignored_params
            )
        else:
            for layer in self.model.layers:
                fully_shard(layer, **fsdp_kwargs)
            fully_shard(self.model, **fsdp_kwargs)

        hone.logger.info(
            f"[Model] FSDP2 applied, world_size={self.world_size}, "
            f"pp_stages={self.pp_stages}, "
            f"mesh={'dp' if self.dp_mesh is not None else 'default-world'}"
        )

    # ------------------------------------------------------------------
    # Pipeline-parallel forward + backward
    # ------------------------------------------------------------------
    def _pp_forward_backward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward/backward across pipeline stages with ResBM compression.

        All stages must call this synchronously. The first stage runs embedding
        + its layers + encodes + sends. The last stage receives + decodes + its
        layers + computes loss + backward + sends gradients back. Middle stages
        (if any) relay in both directions.

        Returns the scalar loss (only meaningful on the last stage; other
        stages return 0.0 as a placeholder).
        """
        assert self.pp_stage is not None
        config = self.model.config
        bottleneck_dim = self.pp_stage.output_boundary.bottleneck_dim if self.pp_stage.output_boundary else (
            self.pp_stage.input_boundary.bottleneck_dim if self.pp_stage.input_boundary else 16
        )

        B, S = input_ids.shape
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.model.rotary_emb(
            torch.empty(1, 1, config.dim, device=self.device), position_ids
        )

        # ---------------- DIAGNOSTIC: per-phase progress (--debug only) ----------------
        # First rank of each stage logs every microbatch; one line per phase
        # boundary so we see live progress when wall times are large.
        # Gated on the logger being at DEBUG level, which the miner enables
        # via ``--debug`` (calls ``hone.debug()``). When disabled we skip
        # the ``cuda.synchronize`` calls entirely so the steady-state
        # training loop pays zero diagnostic overhead.
        diag_log = (
            self.rank == self.pp_stage_ranks[0]
            and hone.logger.isEnabledFor(logging.DEBUG)
        )
        if not hasattr(self, "_pp_microbatch_idx"):
            self._pp_microbatch_idx = 0
        self._pp_microbatch_idx += 1
        mb = self._pp_microbatch_idx

        def _now() -> float:
            if diag_log and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            return time.time()

        def _phase(tag: str, t_prev: float) -> float:
            if not diag_log:
                return t_prev
            t = _now()
            hone.logger.debug(
                f"[Diag/PP {tag} mb={mb}] +{t - t_prev:.3f}s"
            )
            return t

        # Cross-stage transport runs over PPTransport (TCP), set up by
        # Miner.__init__ before init_model returns. PPTransport is a plain
        # blocking-socket abstraction that exchanges raw tensor bytes; it
        # does NOT touch the NCCL stream, which means PP send/recv and
        # FSDP all_gather can interleave freely without deadlocking.
        transport = self.pp_transport
        if transport is None:
            raise RuntimeError(
                "PP forward/backward called but pp_transport is None; "
                "Miner.__init__ should have constructed it when pp_num_stages > 1."
            )
        # ------------------------------------------------------------------------------

        if self.pp_is_first_stage:
            t = _now()
            if diag_log:
                hone.logger.debug(f"[Diag/PP s0 mb={mb}] enter")

            h = self.model.embed_tokens(input_ids)
            h.requires_grad_(True)
            t = _phase("s0 after_embed", t)

            for layer in self.pp_stage.layers:
                result = layer(h, position_embeddings)
                h = result[0] if isinstance(result, tuple) else result
            t = _phase("s0 after_layers", t)

            # Cast to amp_dtype before send so the wire contract is
            # deterministic: FSDP keeps params in their native (often
            # float32) dtype, and autocast only kicks in at op level for
            # whitelisted ops. ``F.embedding`` follows weight.dtype and a
            # subsequent ``bf16 + float32`` add promotes back to float32,
            # so ``compressed`` would otherwise leave the encoder as
            # float32 and double the wire payload.
            compressed = self.pp_stage.output_boundary.encode(h).to(
                self.amp_dtype
            )
            t = _phase("s0 after_encode", t)

            transport.send_next(compressed.contiguous())
            t = _phase("s0 after_send", t)

            grad_compressed = transport.recv_next(
                shape=(B, S, bottleneck_dim), dtype=self.amp_dtype
            )
            # Pull the scalar loss the last stage sends right after the
            # gradient so this rank can log a meaningful value too.
            loss_scalar = transport.recv_next(
                shape=(1,), dtype=self.amp_dtype
            )
            t = _phase("s0 after_recv_grad", t)

            grad_h = self.pp_stage.output_boundary.decoder(grad_compressed)
            h.backward(grad_h)
            _phase("s0 after_bwd", t)

            return loss_scalar.float().reshape(()).to(self.device)

        elif self.pp_is_last_stage:
            assert self.pp_stage.input_boundary is not None
            ib = self.pp_stage.input_boundary

            t = _now()
            if diag_log:
                hone.logger.debug(f"[Diag/PP sN mb={mb}] enter")

            compressed = transport.recv_prev(
                shape=(B, S, ib.bottleneck_dim), dtype=self.amp_dtype
            )
            t = _phase("sN after_recv_act", t)

            compressed.requires_grad_(True)

            h = ib.decode(compressed)

            for layer in self.pp_stage.layers:
                result = layer(h, position_embeddings)
                h = result[0] if isinstance(result, tuple) else result
            t = _phase("sN after_layers", t)

            h = self.model.norm(h)
            logits = self.model.lm_head(h)
            loss = compute_loss(logits, labels)
            t = _phase("sN after_loss", t)

            loss.backward()
            t = _phase("sN after_bwd", t)

            grad_compressed = compressed.grad
            if grad_compressed is None:
                grad_compressed = torch.zeros(
                    B, S, ib.bottleneck_dim,
                    device=self.device, dtype=self.amp_dtype,
                )
            # Same wire-contract reasoning as on the activation send:
            # match amp_dtype so stage 0's recv_next gets exactly what
            # it asked for.
            transport.send_prev(grad_compressed.to(self.amp_dtype).contiguous())
            # Forward the scalar loss back to the previous stage so every
            # stage's ``Inner Step ... loss=`` log line shows the real
            # value (otherwise stage 0 would always print 0.0000).
            loss_scalar = (
                loss.detach().to(self.amp_dtype).reshape(1).contiguous()
            )
            transport.send_prev(loss_scalar)
            _phase(f"sN after_send_grad loss={loss.item():.4f}", t)

            return loss.detach()

        else:
            assert self.pp_stage.input_boundary is not None
            assert self.pp_stage.output_boundary is not None
            ib = self.pp_stage.input_boundary
            ob = self.pp_stage.output_boundary

            compressed_in = transport.recv_prev(
                shape=(B, S, ib.bottleneck_dim), dtype=self.amp_dtype
            )
            compressed_in.requires_grad_(True)

            h = ib.decode(compressed_in)

            for layer in self.pp_stage.layers:
                result = layer(h, position_embeddings)
                h = result[0] if isinstance(result, tuple) else result

            compressed_out = ob.encode(h).to(self.amp_dtype)
            transport.send_next(compressed_out.contiguous())

            grad_compressed_out = transport.recv_next(
                shape=tuple(compressed_out.shape), dtype=self.amp_dtype
            )
            # Receive + relay the scalar loss alongside the gradient so it
            # propagates from the last stage all the way back to stage 0.
            loss_scalar = transport.recv_next(
                shape=(1,), dtype=self.amp_dtype
            )

            compressed_out.backward(grad_compressed_out)

            grad_compressed_in = compressed_in.grad
            if grad_compressed_in is None:
                grad_compressed_in = torch.zeros_like(compressed_in)
            transport.send_prev(
                grad_compressed_in.to(self.amp_dtype).contiguous()
            )
            transport.send_prev(loss_scalar.contiguous())

            return loss_scalar.float().reshape(()).to(self.device)

    # ------------------------------------------------------------------
    # Optimizers & Schedulers
    # ------------------------------------------------------------------
    def init_optimizers_schedulers(self, validator=False):
        self.lr = float(self.hparams.outer_learning_rate)
        self.outer_optimizer = SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            nesterov=True,
        )
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
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.get("learning_rate", 2e-4),
                weight_decay=cfg.get("weight_decay", 0.1),
                betas=tuple(cfg.get("betas", [0.9, 0.95])),
                eps=cfg.get("eps", 1e-8),
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

        loader_iter = iter(loader)

        while not self.stop_event.is_set():
            # 1. Fetch batch
            try:
                batch = await self.loop.run_in_executor(None, next, loader_iter)
                local_has_batch = True
            except StopIteration:
                local_has_batch = False
                batch = None

            if self.world_size > 1:
                if not dist_helper.should_continue(local_has_batch, self.device):
                    break
                if not local_has_batch:
                    continue

            # 2. Prepare inputs
            input_ids = (
                batch.to(self.device, dtype=torch.long, non_blocking=True)
                if isinstance(batch, torch.Tensor)
                else torch.tensor(batch, dtype=torch.long, device=self.device)
            )

            local_bs = len(batch)
            accum_batch_size += local_bs
            tokens_this = input_ids.numel()
            batch_tokens += tokens_this
            local_tokens_sum += tokens_this

            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = self.tokenizer.pad_token_id
            labels = torch.where(
                labels == self.tokenizer.pad_token_id, -100, labels
            )

            has_valid = (labels != -100).any().item()
            if not dist_helper.all_ok(has_valid, self.device):
                del input_ids, labels
                continue

            # 3. Forward + backward
            # ``corrected_accum`` / ``final_micro`` need to be defined
            # before either branch because the optimizer-step bookkeeping
            # below reads them in both PP and non-PP modes. In PP mode the
            # backward already happened inside ``_pp_forward_backward``
            # via ``loss.backward()`` on the last stage, so we don't run a
            # ``scaler.scale(loss).backward()`` here.
            corrected_accum = max(self.sampler.grad_accum_steps, 1)
            final_micro = (batch_count + 1) % corrected_accum == 0

            if self.pp_stages > 1:
                with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    calculated_loss = self._pp_forward_backward(input_ids, labels)
                loss_item = calculated_loss.item()
            else:
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
            need_sync = (
                window_changed or inner_step_count >= max_inner
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
            "global_grad_norm": total_grad_sq ** 0.5 if total_grad_sq > 0 else 0.0,
            "global_weight_norm": total_weight_sq ** 0.5 if total_weight_sq > 0 else 0.0,
        }

    def outer_step(self, gather_result, log_wandb: bool = False):
        return hone.neurons.outer_step(
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
        )
