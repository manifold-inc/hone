"""Standard cross-entropy loss for language modelling."""

from __future__ import annotations

import torch
import torch.nn.functional as F


# Tokens-per-chunk default for the chunked cross-entropy below. Sized so
# the per-chunk fp32 working set
# (``chunk_tokens * vocab_size * 4 bytes``) stays at a few GiB even for
# very large vocabularies (e.g. 4096 tokens * 256k vocab * 4 = 4 GiB) so
# the OOM that bit us at micro_batch_size=8 -- a 32 GiB single-shot fp32
# logits allocation -- is structurally impossible. The chunk dim is the
# flattened (B*S) axis; chunks are independent so this trades almost
# nothing in throughput once chunk_tokens is large enough to keep the
# matmul tile shape tensor-core-friendly.
_DEFAULT_CHUNK_TOKENS = 4096


class _ChunkedCE(torch.autograd.Function):
    """Memory-efficient cross-entropy with float32 upcast.

    The eager ``F.cross_entropy(logits.float(), labels)`` materialises
    the full ``(N, V)`` tensor twice in fp32 (logits + log_probs) and
    keeps log_probs around for backward. At ``micro_batch_size=8`` and
    ``V=256000`` that is 32 GiB per tensor, blowing past the B200's
    180 GiB once activations + optimizer state + FSDP gather buffers
    are factored in (we OOM'd on exactly this allocation).

    Here we slice the flattened ``(B*S, V)`` matrix along the row axis
    into ``chunk_tokens``-sized chunks, do the fp32 softmax + NLL on
    each chunk in turn, and accumulate the loss sum and a counter of
    valid (non-ignored) labels. The fp32 working set per chunk is only
    ``chunk_tokens * V * 4`` bytes.

    Backward is also chunked: we save only the (bf16) ``logits`` and
    ``labels`` -- both of which would already be live for autograd in
    the eager path -- and rebuild ``softmax(logits) - one_hot(labels)``
    per chunk on demand, writing into a single ``grad_logits`` buffer
    of the same dtype as ``logits``.
    """

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int,
        chunk_tokens: int,
    ) -> torch.Tensor:
        # Inputs are guaranteed flattened to 2D / 1D by the caller.
        N, V = logits.shape
        device = logits.device

        loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        valid_count = torch.zeros((), device=device, dtype=torch.float32)

        # We never need the per-chunk forward graph -- backward is
        # custom and recomputes from saved ``logits``.
        with torch.no_grad():
            for i in range(0, N, chunk_tokens):
                end = min(i + chunk_tokens, N)
                lc = logits[i:end].float()
                tc = labels[i:end]

                # ``reduction='sum'`` ignores ``ignore_index`` rows and
                # gives us a partial sum we can divide once at the end
                # by the global valid count.
                loss_sum = loss_sum + F.cross_entropy(
                    lc, tc, ignore_index=ignore_index, reduction="sum",
                )
                valid_count = valid_count + (tc != ignore_index).sum().float()

        ctx.save_for_backward(logits, labels)
        ctx.ignore_index = ignore_index
        ctx.chunk_tokens = chunk_tokens
        # ``valid_count`` is needed to scale the per-chunk softmax-grad
        # back to a "mean over valid tokens" gradient. ``.detach()`` is
        # belt-and-suspenders since we built it inside ``no_grad``.
        ctx.valid_count = valid_count.detach()

        return loss_sum / valid_count.clamp(min=1)

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        logits, labels = ctx.saved_tensors
        ignore_index = ctx.ignore_index
        chunk_tokens = ctx.chunk_tokens
        valid_count = ctx.valid_count
        device = logits.device

        N, V = logits.shape

        # Single grad buffer in the same dtype as ``logits`` (typically
        # bf16 under FSDP mixed-precision); we fill it chunk-by-chunk
        # and never hold a fp32 copy of the full grad.
        grad_logits = torch.empty_like(logits)
        scale = grad_loss / valid_count.clamp(min=1)

        with torch.no_grad():
            for i in range(0, N, chunk_tokens):
                end = min(i + chunk_tokens, N)
                lc = logits[i:end].float()
                tc = labels[i:end]

                # NLL gradient through softmax cross-entropy:
                #   d/dz_j  -log(softmax(z)_label) = softmax(z)_j - 1[j==label]
                # for valid rows, zero for ignored rows. We mutate the
                # softmax tensor in place to avoid an extra full-chunk
                # allocation.
                sm = torch.softmax(lc, dim=-1)
                mask = tc != ignore_index
                if mask.any():
                    rows = torch.arange(end - i, device=device)
                    safe_labels = torch.where(
                        mask, tc, torch.zeros_like(tc)
                    )
                    sm[rows, safe_labels] -= 1.0
                    sm[~mask] = 0.0
                else:
                    sm.zero_()

                grad_logits[i:end] = (sm * scale).to(logits.dtype)

        return grad_logits, None, None, None


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_tokens: int | None = None,
) -> torch.Tensor:
    """Cross-entropy loss with float32 upcast for numerical stability.

    Routes through a chunked custom autograd function so the fp32 logits
    / log_probs working set stays bounded at ``chunk_tokens * V * 4``
    bytes per chunk regardless of ``B * S``. Behaviour matches
    ``F.cross_entropy(logits.float(), labels, ignore_index=-100)`` to
    within fp32 reduction-order rounding.

    Args:
        logits: ``(B, S, V)`` model output logits.
        labels: ``(B, S)`` target token ids, ``-100`` for ignored
            positions.
        chunk_tokens: rows of the flattened ``(B*S, V)`` matrix to
            process per fp32 chunk. ``None`` falls back to the
            module-level default (``_DEFAULT_CHUNK_TOKENS``).
    """
    V = logits.size(-1)
    flat_logits = logits.reshape(-1, V)
    flat_labels = labels.reshape(-1)
    n = chunk_tokens if chunk_tokens is not None else _DEFAULT_CHUNK_TOKENS
    return _ChunkedCE.apply(flat_logits, flat_labels, -100, int(n))
