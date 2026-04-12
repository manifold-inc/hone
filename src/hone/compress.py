from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class CompressedTensor:
    indices: Tensor
    values: Tensor
    quant_params: Tensor
    shape: tuple[int, ...]
    numel: int


def chunk_topk(tensor: Tensor, k: int, chunk_size: int) -> tuple[Tensor, Tensor, int]:
    flat = tensor.reshape(-1).float()
    pad = (-tensor.numel()) % chunk_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    n_chunks = flat.numel() // chunk_size
    chunks = flat.view(n_chunks, chunk_size)
    _, idx = torch.topk(chunks.abs(), k, dim=1, largest=True, sorted=False)
    vals = torch.gather(chunks, 1, idx)
    return idx.to(torch.int16), vals, n_chunks


def quantize_2bit(values: Tensor) -> tuple[Tensor, Tensor]:
    lo = values.amin(dim=1, keepdim=True)
    hi = values.amax(dim=1, keepdim=True)
    span = (hi - lo).clamp_min(1e-8)
    q = ((values - lo) / span * 3.0).round().clamp(0, 3).to(torch.uint8)
    params = torch.cat([lo, hi], dim=1).to(torch.float32)
    n, k = q.shape
    pad = (-k) % 4
    if pad:
        q = torch.nn.functional.pad(q, (0, pad))
    q4 = q.view(n, -1, 4)
    packed = (q4[..., 0] | (q4[..., 1] << 2) | (q4[..., 2] << 4) | (q4[..., 3] << 6)).to(torch.uint8)
    return packed, params


def dequantize_2bit(packed: Tensor, quant_params: Tensor, k: int) -> Tensor:
    n = packed.shape[0]
    b0, b1, b2, b3 = packed & 3, (packed >> 2) & 3, (packed >> 4) & 3, (packed >> 6) & 3
    codes = torch.stack([b0, b1, b2, b3], dim=-1).reshape(n, -1)[:, :k].to(torch.float32)
    lo = quant_params[:, 0:1]
    hi = quant_params[:, 1:2]
    span = (hi - lo).clamp_min(1e-8)
    return codes * (span / 3.0) + lo


def _scatter_flat(
    n_chunks: int, chunk_size: int, flat_len: int, idx: Tensor, vals: Tensor,
) -> Tensor:
    cid = torch.arange(n_chunks, device=vals.device, dtype=torch.long).unsqueeze(1).expand_as(idx)
    gidx = (cid * chunk_size + idx.long()).reshape(-1)
    out = torch.zeros(flat_len, device=vals.device, dtype=vals.dtype)
    out.scatter_(0, gidx, vals.reshape(-1))
    return out


def _ef_threshold(cfg: Any) -> int:
    if getattr(cfg, "ef_freeze_threshold", None) is not None:
        return int(cfg.ef_freeze_threshold)
    n = getattr(cfg, "num_outer_steps", None)
    return 0 if n is None else int(cfg.ef_freeze_pct * n)


def compress(
    pseudo_grad: dict[str, Tensor],
    error_bufs: dict[str, Tensor],
    config: Any,
) -> tuple[dict[str, CompressedTensor], dict[str, Tensor]]:
    if getattr(config, "quant_bits", 2) != 2:
        raise ValueError("only 2-bit quantization is supported")
    k, cs = int(config.topk_k), int(config.chunk_size)
    mom, thr = float(config.ef_momentum), _ef_threshold(config)
    outer = int(getattr(config, "outer_step", 0))
    out_ct: dict[str, CompressedTensor] = {}
    new_err = {}
    for name, pg in pseudo_grad.items():
        eb = error_bufs[name].float()
        e = pg.float() if outer < thr else mom * eb + pg.float()
        idx, vals, nc = chunk_topk(e, k, cs)
        packed, qp = quantize_2bit(vals)
        recon = dequantize_2bit(packed, qp, k)
        pad_len = nc * cs
        dense = _scatter_flat(nc, cs, pad_len, idx, recon)
        upd = (e.reshape(-1) - dense[: e.numel()]).reshape_as(e).to(eb.dtype)
        new_err[name] = upd
        ct = CompressedTensor(
            indices=idx, values=packed, quant_params=qp, shape=tuple(pg.shape), numel=int(pg.numel()),
        )
        ct.chunk_size = cs  # type: ignore[attr-defined]
        out_ct[name] = ct
    return out_ct, new_err


def decompress(compressed: dict[str, CompressedTensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, ct in compressed.items():
        k = int(ct.indices.shape[1])
        vals = dequantize_2bit(ct.values, ct.quant_params, k)
        cs_o = getattr(ct, "chunk_size", None)
        if cs_o is None:
            raise ValueError("CompressedTensor missing chunk_size; use compress() output")
        cs = int(cs_o)
        nc = ct.indices.shape[0]
        pad_len = nc * cs
        flat = _scatter_flat(nc, cs, pad_len, ct.indices, vals)
        out[name] = flat[: ct.numel].reshape(ct.shape).to(torch.float32)
    return out


def aggregate(all_grads: list[dict[str, CompressedTensor]]) -> dict[str, Tensor]:
    if not all_grads:
        return {}
    keys = all_grads[0].keys()
    acc: dict[str, list[Tensor]] = {k: [] for k in keys}
    for g in all_grads:
        d = decompress(g)
        for k, t in d.items():
            acc[k].append(t)
    return {k: torch.stack(ts, 0).mean(0) for k, ts in acc.items()}
