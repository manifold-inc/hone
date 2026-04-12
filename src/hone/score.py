from __future__ import annotations

from typing import Any

import torch
from openskill.models import PlackettLuce

from hone.config import HoneConfig

_BMA_THRESHOLD = 0.1


class Scorer:
    """Tracks per-UID scores across windows using OpenSkill ratings."""

    def __init__(
        self,
        config: HoneConfig,
        max_uids: int = 256,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config
        self.max_uids = max_uids
        self.device = torch.device(device or "cpu")
        z = torch.zeros(max_uids, device=self.device, dtype=torch.float32)
        self.openskill_mu = torch.full_like(z, 25.0)
        self.openskill_sigma = torch.full_like(z, 25.0 / 3.0)
        self.binary_moving_avg = z.clone()
        self.final_scores = z.clone()
        self.weights = z.clone()
        self.windows_evaluated = torch.zeros(max_uids, device=self.device, dtype=torch.int32)
        self.openskill_model = PlackettLuce(beta=config.openskill_beta, tau=config.openskill_tau)

    def update_scores(self, window_scores: dict[int, float], evaluated_uids: list[int]) -> None:
        """Refresh OpenSkill ratings and BMA from window gradient scores."""
        for uid in evaluated_uids:
            if not (0 <= uid < self.max_uids):
                continue
            s = float(window_scores.get(uid, 0.0))
            hit = 1.0 if s > 0 else 0.0
            self.binary_moving_avg[uid] = 0.95 * self.binary_moving_avg[uid] + 0.05 * hit
            self.windows_evaluated[uid] += 1

        ranked = sorted(
            (u for u in evaluated_uids if 0 <= u < self.max_uids and u in window_scores),
            key=lambda u: window_scores[u],
            reverse=True,
        )
        if len(ranked) > 1:
            teams: list[list[Any]] = []
            scores_list: list[float] = []
            for uid in ranked:
                r = self.openskill_model.rating(
                    mu=float(self.openskill_mu[uid].item()),
                    sigma=float(self.openskill_sigma[uid].item()),
                    name=str(uid),
                )
                teams.append([r])
                scores_list.append(float(window_scores[uid]))
            rated = self.openskill_model.rate(teams, scores=scores_list)
            for i, uid in enumerate(ranked):
                nr = rated[i][0]
                self.openskill_mu[uid] = float(nr.mu)
                self.openskill_sigma[uid] = float(nr.sigma)

        sync = 1.0
        warm = self.config.bma_warmup_windows
        for uid in evaluated_uids:
            if not (0 <= uid < self.max_uids):
                continue
            mu, sig = self.openskill_mu[uid], self.openskill_sigma[uid]
            ordinal = float((mu - 3.0 * sig).item())
            bma = max(0.0, float(self.binary_moving_avg[uid].item()))
            if int(self.windows_evaluated[uid].item()) >= warm and bma < _BMA_THRESHOLD:
                bma = 0.0
            self.final_scores[uid] = float(ordinal) * bma * sync

    def compute_weights(self, burn_uid: int | None = None) -> None:
        """Allocate gather / reserve / burn from ``final_scores``; L1-normalize to 1."""
        self.weights.zero_()
        cfg = self.config
        burn = max(0.0, min(1.0, float(cfg.burn_rate))) if burn_uid is not None else 0.0
        g_share = cfg.gather_share
        g_count = cfg.gather_peer_count
        r_count = cfg.reserve_peer_count
        decay_r = float(cfg.reserve_decay_ratio)
        dev, dt = self.weights.device, self.weights.dtype

        pos = (self.final_scores > 0).nonzero(as_tuple=False).flatten().tolist()
        ranked = sorted(pos, key=lambda u: self.final_scores[int(u)].item(), reverse=True)
        gather_u = [int(u) for u in ranked[:g_count]]
        reserve_u = [int(u) for u in ranked[g_count : g_count + r_count]]

        if gather_u:
            n = len(gather_u)
            prof = torch.linspace(2.0, 1.0, n, device=dev, dtype=dt)
            total = (1.0 - burn) * g_share
            s = prof.sum()
            for uid, p in zip(gather_u, prof):
                self.weights[uid] = total * (p / s)

        if reserve_u:
            n = len(reserve_u)
            prof = torch.tensor([decay_r**i for i in range(n)], device=dev, dtype=dt)
            total = (1.0 - burn) * (1.0 - g_share)
            s = prof.sum()
            for uid, p in zip(reserve_u, prof):
                self.weights[uid] = total * (p / s)

        if gather_u and reserve_u:
            gi, ri = torch.tensor(gather_u, device=dev), torch.tensor(reserve_u, device=dev)
            min_g, max_r = self.weights[gi].min(), self.weights[ri].max()
            if max_r > min_g:
                self.weights[ri] *= min_g / max_r * decay_r

        if burn_uid is not None and 0 <= burn_uid < self.max_uids:
            self.weights[burn_uid] = burn

        non_burn = float(self.weights.sum().item()) - burn
        if non_burn > 0:
            self.weights.mul_((1.0 - burn) / non_burn)
            if burn_uid is not None and 0 <= burn_uid < self.max_uids:
                self.weights[burn_uid] = burn
        elif burn_uid is not None and 0 <= burn_uid < self.max_uids:
            self.weights.zero_()
            self.weights[burn_uid] = 1.0

    def get_weights(self) -> tuple[list[int], list[float]]:
        """Active UIDs and weights for ``set_weights``."""
        nz = (self.weights > 0).nonzero(as_tuple=False).flatten().tolist()
        uids = [int(u) for u in nz]
        ws = [float(self.weights[u].item()) for u in uids]
        return uids, ws

    def reset_uid(self, uid: int) -> None:
        """Clear tracked state for ``uid``."""
        if not (0 <= uid < self.max_uids):
            return
        self.openskill_mu[uid] = 25.0
        self.openskill_sigma[uid] = 25.0 / 3.0
        self.binary_moving_avg[uid] = 0.0
        self.final_scores[uid] = 0.0
        self.weights[uid] = 0.0
        self.windows_evaluated[uid] = 0
