"""Pseudo-gradient verification for Hone validators (statistical, loss, Freivalds-style)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class CheckResult:
    name: str
    passed: bool
    score: float
    detail: str = ""


@dataclass
class VerifyResult:
    passed: bool
    checks: list[CheckResult]
    loss_score: float


def _forward_loss(model: nn.Module, batch: dict[str, Tensor]) -> Tensor:
    """Run model forward and compute cross-entropy loss."""
    input_ids = batch["input_ids"]
    labels = batch.get("labels", input_ids)
    out = model(input_ids=input_ids, labels=labels)
    if isinstance(out, tuple):
        return out[0]
    loss = getattr(out, "loss", None)
    if loss is None:
        raise TypeError("Model output must expose .loss or be a (loss, ...) tuple")
    return loss


def _apply_grad_dict(model: nn.Module, grad_dict: dict[str, Tensor], scale: float) -> None:
    for name, p in model.named_parameters():
        if name in grad_dict:
            p.data.add_(grad_dict[name], alpha=-scale)


def _save_restore_params(model: nn.Module, names: frozenset[str]) -> dict[str, Tensor]:
    return {n: p.data.clone() for n, p in model.named_parameters() if n in names}


def _loss_improvement(
    model: nn.Module, batch: dict[str, Tensor], grad_dict: dict[str, Tensor], outer_lr: float
) -> Tensor:
    names = frozenset(grad_dict.keys())
    backup = _save_restore_params(model, names)
    with torch.no_grad():
        before = _forward_loss(model, batch)
        _apply_grad_dict(model, grad_dict, outer_lr)
        after = _forward_loss(model, batch)
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])
    return before - after


class GradientVerifier:
    def __init__(self, config: Any) -> None:
        self.config = config

    def verify(
        self,
        grad_dict: dict[str, Tensor],
        model: nn.Module,
        eval_batch: dict[str, Tensor],
        assigned_batch: dict[str, Tensor] | None = None,
        random_batch: dict[str, Tensor] | None = None,
    ) -> VerifyResult:
        checks: list[CheckResult] = []
        checks.append(self.statistical_check(grad_dict))
        ls = self.loss_score_check(grad_dict, model, eval_batch)
        checks.append(ls)
        checks.append(self.freivalds_check(grad_dict, model, eval_batch))
        if assigned_batch is not None and random_batch is not None:
            checks.append(self.data_provenance_check(grad_dict, model, assigned_batch, random_batch))
        passed = all(c.passed for c in checks)
        return VerifyResult(passed=passed, checks=checks, loss_score=ls.score)

    def statistical_check(self, grad_dict: dict[str, Tensor]) -> CheckResult:
        tensors = list(grad_dict.values())
        if not tensors:
            return CheckResult("statistical", False, 0.0, "empty_grad_dict")
        if any(torch.isnan(t).any() or torch.isinf(t).any() for t in tensors):
            return CheckResult("statistical", False, 0.0, "nan_or_inf")
        total_norm_sq = torch.stack([t.detach().float().norm() ** 2 for t in tensors]).sum()
        if total_norm_sq <= 0:
            return CheckResult("statistical", False, 0.0, "zero_gradient")
        norms = torch.stack([t.detach().float().norm() for t in tensors])
        if (norms == 0).any():
            return CheckResult("statistical", False, 0.0, "zero_tensor")
        med = torch.median(norms)
        if med <= 0:
            return CheckResult("statistical", False, 0.0, "zero_median_norm")
        ratio_high = (norms > med * 100).any()
        ratio_low = (norms < med * 0.01).any()
        if ratio_high or ratio_low:
            return CheckResult("statistical", False, 0.0, "norm_outlier")
        return CheckResult("statistical", True, 1.0, "")

    def loss_score_check(
        self, grad_dict: dict[str, Tensor], model: nn.Module, batch: dict[str, Tensor]
    ) -> CheckResult:
        outer_lr = float(getattr(self.config, "outer_lr", 1.0))
        names = frozenset(grad_dict.keys())
        backup = _save_restore_params(model, names)
        with torch.no_grad():
            loss_before = _forward_loss(model, batch)
            _apply_grad_dict(model, grad_dict, outer_lr)
            loss_after = _forward_loss(model, batch)
            for n, p in model.named_parameters():
                if n in backup:
                    p.data.copy_(backup[n])
        improvement = (loss_before - loss_after).detach().float()
        score = torch.sigmoid(improvement * 10).item()
        passed = bool(improvement > 0)
        return CheckResult("loss_score", passed, score, f"improvement={improvement.item():.6g}")

    def freivalds_check(
        self,
        grad_dict: dict[str, Tensor],
        model: nn.Module,
        batch: dict[str, Tensor],
        num_checks: int = 3,
    ) -> CheckResult:
        def _wkey(prefix: str) -> str:
            return f"{prefix}.weight" if prefix else "weight"

        linears: list[tuple[str, nn.Linear]] = [
            (n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and _wkey(n) in grad_dict
        ]
        if not linears:
            return CheckResult("freivalds", True, 1.0, "no_linear_with_grad")
        k = min(num_checks, len(linears))
        idx = torch.randperm(len(linears), device=next(model.parameters()).device)[:k]
        captured: dict[int, Tensor] = {}

        def make_hook(mid: int):
            def _hook(_m: nn.Module, inp: tuple[Tensor, ...], _o: Any) -> None:
                x = inp[0]
                captured[mid] = x.detach()

            return _hook

        hooks: list[Any] = []
        chosen: list[tuple[str, nn.Linear]] = []
        for j in range(k):
            name, layer = linears[int(idx[j])]
            chosen.append((name, layer))
            hooks.append(layer.register_forward_hook(make_hook(id(layer))))

        model.eval()
        with torch.no_grad():
            _ = _forward_loss(model, batch)

        for h in hooks:
            h.remove()

        passed_cnt = 0
        eps = 1e-8
        for name, layer in chosen:
            wid = f"{name}.weight" if name else "weight"
            delta_w = grad_dict[wid]
            x = captured.get(id(layer))
            if x is None:
                continue
            x2 = x.reshape(-1, x.shape[-1]).to(dtype=delta_w.dtype)
            r = torch.randn(delta_w.shape[0], device=delta_w.device, dtype=delta_w.dtype)
            proj = r @ (delta_w @ x2.T)
            val = proj.norm()
            cap = (r.norm() * delta_w.norm() * x2.norm() + eps)
            rel = (val / cap).item()
            if 1e-6 < rel < 2.0:
                passed_cnt += 1

        score = passed_cnt / max(k, 1)
        return CheckResult("freivalds", score >= 0.5, score, f"{passed_cnt}/{k}")

    def data_provenance_check(
        self,
        grad_dict: dict[str, Tensor],
        model: nn.Module,
        assigned_batch: dict[str, Tensor],
        random_batch: dict[str, Tensor],
    ) -> CheckResult:
        outer_lr = float(getattr(self.config, "outer_lr", 1.0))
        ia = _loss_improvement(model, assigned_batch, grad_dict, outer_lr).detach().float()
        ir = _loss_improvement(model, random_batch, grad_dict, outer_lr).detach().float()
        eps = 1e-8
        ia_p = ia.clamp_min(0)
        ir_p = ir.clamp_min(0)
        score = (ia_p / (ia_p + ir_p + eps)).item()
        passed = bool(ia >= ir)
        return CheckResult("data_provenance", passed, score, f"assigned={ia.item():.6g} random={ir.item():.6g}")

    @staticmethod
    def normalize_norms(grad_dicts: dict[int, dict[str, Tensor]]) -> dict[int, dict[str, Tensor]]:
        per_peer: list[tuple[int, Tensor]] = []
        for pid, g in grad_dicts.items():
            sq = torch.stack([t.float().norm() ** 2 for t in g.values()]).sum()
            per_peer.append((pid, torch.sqrt(sq)))
        if not per_peer:
            return {}
        norms = torch.stack([n for _, n in per_peer])
        med = torch.median(norms)
        eps = 1e-8
        out: dict[int, dict[str, Tensor]] = {}
        for pid, n in per_peer:
            scale = med / (n + eps)
            out[pid] = {k: v * scale for k, v in grad_dicts[pid].items()}
        return out
