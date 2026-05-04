# The MIT License (MIT)
# © 2025 hone.training

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Adapted from https://github.com/bloc97/DeMo and NousResearch


# Global imports

import math
from typing import Generic, Literal, Sequence, TypeAlias, TypeVar, cast, overload

import torch
import torch.fft
from einops import rearrange
from torch.distributed.tensor import DTensor as DT

import hone

# ─────────── type aliases ────────────────────────────────────────────────
# primitive shapes
ShapeT: TypeAlias = tuple[int, ...]  # original dense tensor shape
Shape4D = tuple[int, int, int, int]  # y, x, h, w
TotK: TypeAlias = int  # size of the last dim

# 12‑bit packed representation - just the uint8 buffer, no tuple
IdxT: TypeAlias = torch.Tensor  # 12-bit packed indices (stored as uint8 tensor)

# Quantisation params travel alongside every quantised values blob so the
# receiver can reconstruct floats. Two wire variants exist for backward-
# compatible rollout of 2-bit value packing (P0b):
#
#   * Legacy / version 0 (5-tuple): ``(shift, scale, offset, lookup, dtype)``.
#     ``vals`` is a raw uint8 tensor of shape ``(..., topk)`` with one byte
#     per quantised value.
#   * Packed / version 1 (7-tuple):
#     ``(shift, scale, offset, lookup, dtype, pack_version=1, original_last_dim)``.
#     ``vals`` is a 2-bit-packed uint8 tensor of shape ``(..., ceil(topk/4))``.
#     The receiver detects v1 via tuple length, restores the original last
#     dim using ``original_last_dim``, then dequantises through ``lookup``.
#
# Both variants are accepted by every decode path. New senders only emit v1
# when the operator opts in via the ``pack_values_2bit`` hparam (default
# ``False``); see ``TopKCompressor.__init__``.
LegacyQuantParamsT: TypeAlias = tuple[
    torch.Tensor, float, int, torch.Tensor, torch.dtype
]
PackedQuantParamsT: TypeAlias = tuple[
    torch.Tensor, float, int, torch.Tensor, torch.dtype, int, int
]
QuantParamsT: TypeAlias = LegacyQuantParamsT | PackedQuantParamsT

# For historical names kept elsewhere in the code
ValT: TypeAlias = torch.Tensor

# Boolean flag that propagates the chosen quantisation mode
Q = TypeVar("Q", Literal[True], Literal[False])


def pack_12bit_indices(indices: torch.Tensor) -> torch.Tensor:
    """
    Pack int64 indices into 12-bit representation.
    Every 2 indices (24 bits) are packed into 3 uint8 values.
    Assumes even number of indices (topk is always even).

    Args:
        indices: Tensor with values < 4096 (12-bit max), must have even number of elements

    Returns:
        packed_tensor as uint8
    """
    # Ensure indices fit in 12 bits
    max_idx = indices.max().item() if indices.numel() > 0 else 0
    if max_idx >= 4096:
        raise ValueError(f"Index {max_idx} exceeds 12-bit limit (4095)")

    # Flatten the tensor
    indices_flat = indices.flatten()
    n_indices = indices_flat.numel()

    # Ensure we have even number of indices
    if n_indices % 2 != 0:
        raise ValueError(f"Number of indices must be even, got {n_indices}")

    # Convert to int32 for bit manipulation
    indices_flat = indices_flat.to(torch.int32)

    # Process all as pairs
    indices_pairs = indices_flat
    n_pairs = n_indices // 2

    # Calculate packed size
    packed_size = n_pairs * 3
    packed = torch.zeros(packed_size, dtype=torch.uint8, device=indices.device)

    # Vectorized packing for pairs
    if n_pairs > 0:
        idx_pairs = indices_pairs.reshape(-1, 2)
        idx1 = idx_pairs[:, 0]
        idx2 = idx_pairs[:, 1]

        # Pack pairs: idx1 uses byte0 + lower 4 bits of byte1
        #            idx2 uses upper 4 bits of byte1 + byte2
        packed[0::3] = (idx1 & 0xFF).to(torch.uint8)  # Lower 8 bits of idx1
        packed[1::3] = (((idx1 >> 8) & 0x0F) | ((idx2 & 0x0F) << 4)).to(torch.uint8)
        packed[2::3] = ((idx2 >> 4) & 0xFF).to(torch.uint8)  # Upper 8 bits of idx2

    return packed


def unpack_12bit_indices(packed: torch.Tensor, values_shape: ShapeT) -> torch.Tensor:
    """
    Unpack 12-bit packed indices back to int64.
    Assumes even number of indices.

    Args:
        packed: Packed uint8 tensor
        values_shape: Shape of the values tensor (same as original indices shape)

    Returns:
        Unpacked indices as int64 tensor with original shape
    """
    n_indices = int(torch.prod(torch.tensor(values_shape)).item())

    if n_indices == 0:
        return torch.zeros(values_shape, dtype=torch.int64, device=packed.device)

    # Ensure even number of indices
    if n_indices % 2 != 0:
        raise ValueError(f"Number of indices must be even, got {n_indices}")

    # Prepare output
    indices = torch.zeros(n_indices, dtype=torch.int64, device=packed.device)

    # All indices are paired
    n_pairs = n_indices // 2

    if n_pairs > 0:
        # Vectorized unpacking
        byte0 = packed[0::3].to(torch.int64)
        byte1 = packed[1::3].to(torch.int64)
        byte2 = packed[2::3].to(torch.int64)

        # Reconstruct indices
        indices[0::2] = byte0 | ((byte1 & 0x0F) << 8)  # idx1
        indices[1::2] = ((byte1 >> 4) & 0x0F) | (byte2 << 4)  # idx2

    # Reshape to match values shape
    indices = indices.reshape(values_shape)

    return indices


def pack_2bit_values(values: torch.Tensor) -> torch.Tensor:
    """
    Pack uint8 values in [0, 3] into a 2-bit representation.

    Every 4 consecutive values along the LAST dim are packed into 1 byte:

        byte = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)

    i.e. v0 occupies bits 0-1, v1 bits 2-3, v2 bits 4-5, v3 bits 6-7
    (little-endian within each byte). This mirrors the bit ordering of
    ``pack_12bit_indices`` (lower-order bits of each value go in the
    lower-order bits of each byte) so the two packers feel consistent
    on the wire.

    The packing is row-wise so the leading dims of ``values`` are
    preserved in the output. When the last dim is not a multiple of 4
    it is right-padded with zeros; the receiver must know the original
    last-dim size to drop the padding (carried in the qparams as the
    7th element ``original_last_dim``; see ``QuantParamsT``).

    Args:
        values: uint8 tensor with all entries in [0, 3]. Arbitrary
            leading dims; the LAST dim is what gets packed.

    Returns:
        uint8 tensor with the same leading dims and last dim of
        ``ceil(N / 4)`` where ``N`` is the original last dim. For an
        empty input the tensor is returned unchanged.
    """
    if values.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 input, got {values.dtype}")
    if values.numel() == 0:
        return values

    # Bound check (cheap fail-fast: would silently corrupt other lanes
    # otherwise since 0x03 mask just truncates the upper bits).
    max_val = int(values.max().item())
    if max_val >= 4:
        raise ValueError(f"Value {max_val} exceeds 2-bit limit (3)")

    n = values.shape[-1]
    pad = (-n) % 4  # number of zero-padding entries to append along last dim
    if pad:
        pad_shape = (*values.shape[:-1], pad)
        values = torch.cat(
            [
                values,
                torch.zeros(pad_shape, dtype=torch.uint8, device=values.device),
            ],
            dim=-1,
        )

    # Reshape last dim into groups of 4: shape (..., n_packed, 4).
    n_packed = values.shape[-1] // 4
    quartets = values.reshape(*values.shape[:-1], n_packed, 4)

    # Vectorized OR-shift pack. uint8 left-shift of a [0,3] value by up
    # to 6 bits stays inside the byte (max is 3<<6 = 192) so the OR-
    # combine fits in uint8 without overflow.
    packed = (
        quartets[..., 0]
        | (quartets[..., 1] << 2)
        | (quartets[..., 2] << 4)
        | (quartets[..., 3] << 6)
    )
    return packed.contiguous()


def unpack_2bit_values(packed: torch.Tensor, original_last_dim: int) -> torch.Tensor:
    """
    Inverse of ``pack_2bit_values``.

    Unpacks 4 uint8 values per byte along the last dim using the same
    little-endian-within-byte layout as the packer:

        v0 = packed & 0x03
        v1 = (packed >> 2) & 0x03
        v2 = (packed >> 4) & 0x03
        v3 = (packed >> 6) & 0x03

    Then truncates the last dim to ``original_last_dim`` to drop any
    zero-padding the packer added.

    Args:
        packed: uint8 tensor produced by ``pack_2bit_values``. The last
            dim is expected to be ``ceil(original_last_dim / 4)``.
        original_last_dim: value of the last dim BEFORE packing/padding.
            Stored in the 7-tuple ``QuantParamsT`` and propagated end-to-end.

    Returns:
        uint8 tensor with the same leading dims as ``packed`` and last
        dim of ``original_last_dim``. All entries are in [0, 3].
    """
    if packed.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 packed input, got {packed.dtype}")

    expected_packed_len = (original_last_dim + 3) // 4
    if packed.shape[-1] != expected_packed_len:
        raise ValueError(
            f"Packed last dim {packed.shape[-1]} does not match expected "
            f"{expected_packed_len} for original_last_dim={original_last_dim}"
        )

    if original_last_dim == 0 or packed.numel() == 0:
        out_shape = (*packed.shape[:-1], original_last_dim)
        return torch.empty(out_shape, dtype=torch.uint8, device=packed.device)

    # Vectorized unpack. Layout matches the packer (v0 in low bits).
    v0 = packed & 0x03
    v1 = (packed >> 2) & 0x03
    v2 = (packed >> 4) & 0x03
    v3 = (packed >> 6) & 0x03

    # Stack along a new trailing dim and flatten so the recovered order
    # is (v0, v1, v2, v3, v0, v1, v2, v3, ...) — exactly the order the
    # packer consumed.
    quartets = torch.stack([v0, v1, v2, v3], dim=-1)
    unpacked = quartets.reshape(*packed.shape[:-1], -1)

    if unpacked.shape[-1] != original_last_dim:
        unpacked = unpacked[..., :original_last_dim]

    return unpacked.contiguous()


# TurboQuant codec (``hone.turboquant``) imports multi-bit-width pack /
# unpack helpers at module level. Only ``pack_2bit_values`` /
# ``unpack_2bit_values`` ship implemented today because
# ``turboquant_enabled: false`` by default and the P6a audit is the hard
# gate before a real b=1/3/4 codec lands. The stubs below keep the
# ``hone.turboquant`` module-level import healthy (otherwise every
# validator / miner boot raises ``ImportError`` from
# ``hone/src/hone/__init__.py:24 from . import turboquant``) without
# letting any default-on code path silently lose bits. If an operator
# flips ``turboquant_enabled: true`` AND picks a bit width that hasn't
# been implemented, they get a clear ``NotImplementedError`` at the
# first compress call rather than a silent NaN.


def _unsupported_pack_stub(bits: int, *, role: str):
    def _impl(values: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
        raise NotImplementedError(
            f"{role} for b={bits} is not implemented in this build. "
            f"TurboQuant b={bits} requires the P6a coordinate-distribution "
            f"audit to pass before the codec ships. Until then, keep "
            f"``turboquant_enabled: false`` or use ``turboquant_bits=2`` "
            f"(the only width with shipped pack/unpack helpers)."
        )

    _impl.__name__ = role
    return _impl


def _unsupported_unpack_stub(bits: int, *, role: str):
    def _impl(
        packed: torch.Tensor,  # noqa: ARG001
        original_last_dim: int,  # noqa: ARG001
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"{role} for b={bits} is not implemented in this build. "
            f"See ``pack_{bits}bit_values`` for the full explanation."
        )

    _impl.__name__ = role
    return _impl


pack_1bit_values = _unsupported_pack_stub(1, role="pack_1bit_values")
unpack_1bit_values = _unsupported_unpack_stub(1, role="unpack_1bit_values")
pack_3bit_values = _unsupported_pack_stub(3, role="pack_3bit_values")
unpack_3bit_values = _unsupported_unpack_stub(3, role="unpack_3bit_values")
pack_4bit_values = _unsupported_pack_stub(4, role="pack_4bit_values")
unpack_4bit_values = _unsupported_unpack_stub(4, role="unpack_4bit_values")


class ChunkingTransformer:
    """
    A transformer for chunking tensors to enable more efficient gradient processing.

    This class handles the chunking of tensors into smaller blocks, which can be
    processed more efficiently. It pre-calculates Discrete Cosine Transform (DCT)
    basis matrices for various tensor sizes to speed up the transformation process.
    """

    @torch.no_grad()
    def __init__(self, model, target_chunk, norm="ortho"):
        """
        Initialise the ChunkingTransformer.

        Args:
            model: The model whose parameters will be processed.
            target_chunk (int): The target size for tensor chunks.
            norm (str): The normalization to be used for DCT ('ortho' or None).
        """
        self.target_chunk = target_chunk

        self.shape_dict = dict()
        self.f_dict = dict()
        self.b_dict = dict()

        # Get all variants of model tensor sizes
        # Generate all possible valid DCT sizes for model tensors
        def _register_size(s: int, dtype, device) -> None:
            sc = _get_smaller_split(s, self.target_chunk)
            self.shape_dict[s] = sc
            if sc not in self.f_dict:
                I = torch.eye(sc)  # noqa: E741
                self.f_dict[sc] = _dct(I, norm=norm).to(dtype).to(device)
                self.b_dict[sc] = _idct(I, norm=norm).to(dtype).to(device)

        for _, p in model.named_parameters():
            if not p.requires_grad:
                continue
            for s in p.shape:
                _register_size(s, p.dtype, p.device)

            # For 3D params (stacked MoE expert weights ``(E, D, ffn)``)
            # the gradient path collapses the leading two dims into rows
            # so the codec can chunk a regular 2D matrix
            # ``(E*D, ffn)``. Pre-register the combined dim so
            # ``encode`` doesn't KeyError on a shape it never saw at
            # init time.
            if p.dim() == 3:
                combined = int(p.shape[0]) * int(p.shape[1])
                _register_size(combined, p.dtype, p.device)

    @torch.no_grad()
    def einsum_2d(self, x, b, d=None) -> torch.Tensor:
        """
        Apply a 2D einsum operation for encoding.

        Args:
            x (torch.Tensor): The input tensor.
            b (torch.Tensor): The first basis matrix.
            d (torch.Tensor, optional): The second basis matrix. Defaults to None.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        if d is None:
            return torch.einsum("...ij, jb -> ...ib", x, b)
        else:
            # Note: b-c axis output is transposed to chunk DCT in 2D
            return torch.einsum("...ijkl, kb, ld -> ...ijbd", x, b, d)

    @torch.no_grad()
    def einsum_2d_t(self, x, b, d=None) -> torch.Tensor:
        """
        Apply a 2D einsum operation for decoding (transpose).

        Args:
            x (torch.Tensor): The input tensor.
            b (torch.Tensor): The first basis matrix.
            d (torch.Tensor, optional): The second basis matrix. Defaults to None.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        if d is None:
            return torch.einsum("...ij, jb -> ...ib", x, b)
        else:
            # Note: b-c axis output is transposed to chunk DCT in 2D
            return torch.einsum("...ijbd, bk, dl -> ...ijkl", x, b, d)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, *, use_dct: bool = False) -> torch.Tensor:
        """
        Encode a tensor by chunking and optionally applying DCT.

        Args:
            x (torch.Tensor): The input tensor to encode.
            use_dct (bool): Whether to apply the Discrete Cosine Transform.

        Returns:
            torch.Tensor: The encoded tensor.
        """
        if len(x.shape) > 1:  # 2D weights
            n1 = self.shape_dict[x.shape[0]]
            n2 = self.shape_dict[x.shape[1]]
            n1w = self.f_dict[n1].to(device=x.device, dtype=x.dtype)
            n2w = self.f_dict[n2].to(device=x.device, dtype=x.dtype)
            self.f_dict[n1] = n1w
            self.f_dict[n2] = n2w

            x = rearrange(x, "(y h) (x w) -> y x h w", h=n1, w=n2)
            if use_dct:
                x = self.einsum_2d(x, n1w, n2w)

        else:  # 1D weights
            n1 = self.shape_dict[x.shape[0]]
            n1w = self.f_dict[n1].to(device=x.device, dtype=x.dtype)
            self.f_dict[n1] = n1w

            x = rearrange(x, "(x w) -> x w", w=n1)
            if use_dct:
                x = self.einsum_2d(x, n1w)

        return x

    @torch.no_grad()
    def decode(self, x: torch.Tensor, *, use_dct: bool = False) -> torch.Tensor:
        """
        Decode a tensor by un-chunking and optionally applying inverse DCT.

        Args:
            x (torch.Tensor): The input tensor to decode.
            use_dct (bool): Whether to apply the inverse Discrete Cosine Transform.

        Returns:
            torch.Tensor: The decoded tensor.
        """
        if len(x.shape) > 2:  # 2D weights
            if use_dct:
                n1 = x.shape[2]
                n2 = x.shape[3]
                n1w = self.b_dict[n1].to(device=x.device, dtype=x.dtype)
                n2w = self.b_dict[n2].to(device=x.device, dtype=x.dtype)
                self.b_dict[n1] = n1w
                self.b_dict[n2] = n2w

                x = self.einsum_2d_t(x, n1w, n2w)
            x = rearrange(x, "y x h w -> (y h) (x w)")

        else:  # 1D weights
            if use_dct:
                n1 = x.shape[1]
                n1w = self.b_dict[n1].to(device=x.device, dtype=x.dtype)
                self.b_dict[n1] = n1w

                x = self.einsum_2d_t(x, n1w)
            x = rearrange(x, "x w -> (x w)")

        return x


# ``pack_version`` constants stored at index 5 of the 7-tuple QuantParamsT.
# v0 is implicit (legacy 5-tuple, raw uint8 values). v1 is the new 2-bit
# packed format introduced in P0b. Bumping this past 1 requires adding a
# matching branch in ``_dequantize_values``.
PACK_VERSION_LEGACY: int = 0
PACK_VERSION_2BIT: int = 1


class TopKCompressor(Generic[Q]):
    """
    A gradient sparsifier/compressor that uses Top-K selection and optional quantization.

    This class can be used to compress gradients by selecting the top-k largest values
    and optionally quantizing them to 8-bit integers for further size reduction.
    It supports both 1D and 2D tensors.
    """

    use_quantization: Q
    n_bins: int
    range_in_sigmas: int
    pack_values_2bit: bool

    # ------------------------------------------------------------------ #
    # Constructor – two overloads so each instance "remembers" its mode
    # ------------------------------------------------------------------ #
    @overload
    def __init__(
        self: "TopKCompressor[Literal[True]]",
        *,
        use_quantization: Literal[True] = True,
        quantization_bins: int = 256,
        quantization_range: int = 6,
        pack_values_2bit: bool = False,
    ) -> None: ...

    @overload
    def __init__(
        self: "TopKCompressor[Literal[False]]",
        *,
        use_quantization: Literal[False] = False,
        quantization_bins: int = 256,
        quantization_range: int = 6,
        pack_values_2bit: bool = False,
    ) -> None: ...

    @torch.no_grad()
    def __init__(
        self,
        *,
        use_quantization: bool = False,
        quantization_bins: int = 256,
        quantization_range: int = 6,
        pack_values_2bit: bool = False,
    ) -> None:
        """
        Initialise the TopKCompressor.

        Args:
            use_quantization (bool): Whether to use 8-bit quantization.
            quantization_bins (int): The number of bins for quantization.
            quantization_range (int): The quantization range in standard deviations.
            pack_values_2bit (bool): Opt-in switch that bit-packs 4-bin
                quantised values from 8-bit-per-value (uint8) down to true
                2-bit-per-value on the wire (4 values per byte). Only honoured
                when ``use_quantization=True`` and ``quantization_bins == 4``;
                ignored otherwise. Decoders unconditionally accept BOTH wire
                formats (signalled per-blob via the qparams tuple length), so
                flipping this at the sender side is safe as long as
                receivers run a version of the codec that recognises the
                7-tuple qparams. Default ``False`` to keep the wire format
                bit-for-bit identical for staged rollouts.
        """
        self.use_quantization = cast(Q, use_quantization)
        if self.use_quantization:
            self.n_bins = quantization_bins
            self.range_in_sigmas = (
                quantization_range  # Quantization range in standard deviations
            )
        # Stored regardless of ``use_quantization`` so toggling the flag
        # later is observable, but only consulted inside ``_quantize_values``
        # which is itself gated on ``use_quantization``.
        self.pack_values_2bit = bool(pack_values_2bit) and quantization_bins == 4

    def _clamp_topk(self, x, topk) -> int:
        """
        Clamp the top-k value to be within the valid range.

        For dims >= 2 the result is even (required by 12-bit index packing).
        For dim == 1 the result is 1; the caller pads to even before packing.

        Args:
            x (torch.Tensor): The input tensor.
            topk (int): The desired top-k value.

        Returns:
            int: The clamped top-k value, guaranteed <= dim.
        """
        dim = x.shape[-1]
        topk = min(topk, dim)
        topk = max(topk, 1)
        if dim >= 2:
            # Ensure topk is even for 12-bit packing efficiency
            topk = topk - (topk % 2)
            topk = max(topk, 2)
        return int(topk)

    # ------------------------------------------------------------------ #
    # compress – returns a 5-tuple *or* a 4-tuple, depending on the mode
    # ------------------------------------------------------------------ #
    @overload
    def compress(
        self: "TopKCompressor[Literal[True]]",
        x: torch.Tensor,
        topk: int,
    ) -> tuple[IdxT, ValT, ShapeT, TotK, QuantParamsT]: ...
    @overload
    def compress(
        self: "TopKCompressor[Literal[False]]",
        x: torch.Tensor,
        topk: int,
    ) -> tuple[IdxT, ValT, ShapeT, TotK]: ...

    @torch.no_grad()
    def compress(self, x: torch.Tensor, topk: int):  # type: ignore[override]
        """
        Compress a tensor using top-k selection and optional quantization.

        Args:
            x (torch.Tensor): The input tensor to compress.
            topk (int): The number of top values to select.

        Returns:
            A tuple containing the compressed data. The format depends on whether
            quantization is used.
        """
        if isinstance(x, DT):  # check for dtensors
            x = x.to_local()
        xshape = x.shape

        if len(x.shape) > 2:  # 2D weights
            x = rearrange(x, "y x h w -> y x (h w)")

        # Limit topk to max size
        totalk = x.shape[-1]
        topk = self._clamp_topk(x, topk)

        idx_int64 = torch.topk(
            x.abs(), k=topk, dim=-1, largest=True, sorted=False
        ).indices
        val = torch.gather(x, dim=-1, index=idx_int64)

        # 12-bit packing requires an even number of indices per row.
        # For very small dims (e.g. bias of shape (1,)), topk may be odd;
        # duplicate the last entry so scatter_reduce mean is unchanged.
        if topk % 2 != 0:
            idx_int64 = torch.cat([idx_int64, idx_int64[..., -1:]], dim=-1)
            val = torch.cat([val, val[..., -1:]], dim=-1)

        # Pack indices into 12-bit representation for efficient storage
        # This reduces storage by 25% compared to int16
        idx = pack_12bit_indices(idx_int64)

        # Apply 8-bit quantization if enabled
        if self.use_quantization:
            val, quant_params = self._quantize_values(val)
            return idx, val, xshape, totalk, quant_params

        return idx, val, xshape, totalk

    @torch.no_grad()
    def decompress(
        self,
        p: torch.Tensor,
        idx: torch.Tensor,
        val: torch.Tensor,
        xshape: ShapeT,
        totalk: int,
        quantize_params: QuantParamsT | None = None,
        *,
        reduce: Literal["mean", "sum"] = "mean",
    ) -> torch.Tensor:
        """
        Decompress a tensor from its sparse representation.

        Args:
            p (torch.Tensor): A tensor with the target shape and device.
            idx (torch.Tensor): The indices of the non-zero values.
            val (torch.Tensor): The non-zero values.
            xshape (ShapeT): The original shape of the tensor.
            totalk (int): The total number of elements in the original tensor's last dim.
            quantize_params (QuantParamsT, optional): Quantization parameters. Defaults to None.
            reduce (Literal["mean", "sum"]): scatter_reduce mode applied
                across overlapping concatenated indices. Defaults to
                ``"mean"`` (legacy behaviour: position-wise average over
                whichever peers landed on each cell). ``batch_decompress``
                switches to ``"sum"`` only when caller supplied
                ``peer_weights`` (after normalisation, sum recovers the
                global weighted average; see ``batch_decompress``).
                Single-peer call sites should leave at the default.

        Returns:
            torch.Tensor: The decompressed tensor.
        """
        if self.use_quantization and quantize_params is not None:
            val = self._dequantize_values(val, quantize_params)

        x = torch.zeros(xshape, device=p.device, dtype=p.dtype)

        if len(xshape) > 2:  # 2D weights
            x = rearrange(x, "y x h w -> y x (h w)")

        # Unpack 12-bit indices using val shape (if needed)
        if idx.dtype == torch.uint8:
            # 12-bit packed format - unpack it
            idx_int64 = unpack_12bit_indices(idx, val.shape)
        elif idx.dtype in (torch.int64, torch.long):
            # Already unpacked (from batch_decompress)
            idx_int64 = idx
        else:
            raise ValueError(
                f"Expected uint8 (packed) or int64 (unpacked) indices, got {idx.dtype}"
            )
        # Ensure val has the same dtype as x for scatter operation
        if val.dtype != x.dtype:
            val = val.to(dtype=x.dtype)

        x.scatter_reduce_(
            dim=-1, index=idx_int64, src=val, reduce=reduce, include_self=False
        ).reshape(xshape)

        if len(x.shape) > 2:  # 2D weights
            xshape4 = cast(Shape4D, xshape)
            h_dim = xshape4[2]
            x = rearrange(x, "y x (h w) -> y x h w", h=h_dim)

        return x

    @torch.no_grad()
    def batch_decompress(
        self,
        p: torch.Tensor,
        idx: torch.Tensor | Sequence[torch.Tensor],
        val: torch.Tensor | Sequence[torch.Tensor],
        xshape: ShapeT,
        totalk: int,
        quantize_params: Sequence[QuantParamsT] | None = None,
        *,
        block_norms: torch.Tensor | None = None,
        normalise: bool = False,
        clip_norm: bool = True,
        peer_weights: Sequence[float] | None = None,
    ) -> torch.Tensor:
        """
        Decompress a batch of sparse tensors and combine them.

        Args:
            p (torch.Tensor): A tensor with the target shape and device.
            idx (torch.Tensor | Sequence[torch.Tensor]): A sequence of indices for each tensor in the batch.
            val (torch.Tensor | Sequence[torch.Tensor]): A sequence of values for each tensor in the batch.
            xshape (ShapeT): The original shape of the tensors.
            totalk (int): The total number of elements in the original tensor's last dim.
            quantize_params (Sequence[QuantParamsT], optional): A sequence of quantization parameters. Defaults to None.
            block_norms (torch.Tensor, optional): Pre-computed norms for each block. Defaults to None.
            normalise (bool): Whether to normalise the values. Defaults to False.
            clip_norm (bool): Whether to clip the norms of the values. Defaults to True.
            peer_weights (Sequence[float], optional): P2 token-weighted
                aggregation weights, one per peer in the same order as
                ``val``. When provided, weights are normalised to sum to
                1 and each peer's dequantised values are pre-multiplied
                by the normalised weight before the cross-peer
                ``scatter_reduce_``; the reduce mode is switched from
                ``"mean"`` to ``"sum"`` so that, for the dense
                ``vec_p[k] = vals_p[k] if k in idxs_p else 0`` view of
                each peer, the output equals
                ``sum_p w_p * vec_p[k]`` — the standard global weighted
                average where missing-peer cells contribute 0. ``None``
                preserves the legacy uniform ``"mean"`` reduce behaviour
                bit-for-bit. Length MUST equal ``len(val)``.

        Returns:
            torch.Tensor: The combined, decompressed tensor.
        """
        if quantize_params is not None and not isinstance(quantize_params, list):
            quantize_params = [quantize_params] * len(val)  # type: ignore[list-item]

        processed_vals: list[torch.Tensor] = []
        dequant_vals = None
        norms = None
        clip_norm_val = None
        if self.use_quantization and quantize_params:
            dequant_vals = [
                self._dequantize_values(v, quantize_params[i])
                for i, v in enumerate(val)
            ]
        if clip_norm:
            # If caller already supplied per-block norms, trust them.
            if block_norms is not None:
                norms = block_norms.to(p.device)
            else:
                vals_for_norm = dequant_vals if dequant_vals is not None else val
                norms = torch.stack(
                    [torch.norm(sparse_vals, p=2) for sparse_vals in vals_for_norm]
                )
            clip_norm_val = torch.median(norms)

        vals = dequant_vals if dequant_vals is not None else val

        # P2 token-weighted aggregation pre-multiply. Normalised so
        # ``sum_p w_p == 1``; combined with the ``reduce="sum"`` swap
        # downstream, this turns ``scatter_reduce_`` into the canonical
        # global weighted-average over the dense per-peer view of each
        # gradient. Length must match the cross-peer ``vals`` list; we
        # validate up front so a miswired caller fails loudly rather
        # than silently mis-weighting.
        normalised_weights: list[float] | None = None
        if peer_weights is not None:
            peer_weights_list = list(peer_weights)
            if len(peer_weights_list) != len(vals):
                raise ValueError(
                    f"peer_weights length {len(peer_weights_list)} does not "
                    f"match number of peer val tensors {len(vals)}"
                )
            weight_sum = float(sum(peer_weights_list))
            if weight_sum <= 0.0:
                # Degenerate input: every peer reported zero. Fall back
                # to uniform mean so the merge still completes.
                normalised_weights = None
            else:
                normalised_weights = [
                    float(w) / weight_sum for w in peer_weights_list
                ]

        for i, v in enumerate(vals):
            v = v.to(p.device)

            if normalise:
                eps = 1e-8
                if len(v.shape) == 3:  # 2D weights
                    l2_norm = torch.norm(v, p=2, dim=2, keepdim=True)
                    v = v / (l2_norm + eps)
                elif len(v.shape) == 2:  # 1D weights (biases)
                    l2_norm = torch.norm(v, p=2, dim=1, keepdim=True)
                    v = v / (l2_norm + eps)
                elif len(v.shape) == 1:  # Single values
                    l2_norm = torch.norm(v, p=2)
                    if l2_norm > eps:
                        v = v / l2_norm
            elif clip_norm and norms is not None and clip_norm_val is not None:
                current_norm = norms[i]
                clip_factor = torch.clamp(clip_norm_val / (current_norm + 1e-8), max=1)
                v = v * clip_factor

            if normalised_weights is not None:
                # Pre-multiply this peer's vals by their normalised
                # weight. Cast through ``v.dtype`` so the downstream
                # ``cat -> scatter_reduce`` path sees a uniform dtype
                # (the legacy mean-reduce path does the same implicit
                # broadcast through ``clip_factor``).
                v = v * v.new_tensor(normalised_weights[i])

            processed_vals.append(v)

        # Unpack and concatenate indices
        unpacked_indices = []
        idx_list = idx if isinstance(idx, Sequence) else [idx]

        # ``unpack_12bit_indices`` needs the ORIGINAL (pre-quantisation)
        # values shape to know how many indices to recover. ``processed_vals``
        # always carries that shape: it is the dequantised tensor when
        # quantisation ran (so 2-bit packing has been undone), or the
        # caller-supplied tensor otherwise. Reading the wire ``val`` shape
        # directly would be wrong under PACK_VERSION_2BIT, where the last
        # dim has been compressed 4×.
        for i, i_data in enumerate(idx_list):
            if i_data.dtype != torch.uint8:
                raise ValueError(
                    f"Expected uint8 for 12-bit packed indices, got {i_data.dtype}"
                )
            v_data = processed_vals[i]
            idx_unpacked = unpack_12bit_indices(i_data.to(p.device), v_data.shape)
            unpacked_indices.append(idx_unpacked)

        idx_concat = torch.cat(unpacked_indices, dim=-1)
        val_concat = torch.cat(processed_vals, dim=-1).to(p.dtype)

        # Reduce mode: when peer_weights normalised cleanly, sum gives
        # the canonical weighted average over the dense per-peer view
        # (see the kwarg docstring). Else fall back to the legacy mean
        # behaviour so call sites that don't pass ``peer_weights`` get
        # bit-for-bit identical results.
        reduce_mode: Literal["mean", "sum"] = (
            "sum" if normalised_weights is not None else "mean"
        )

        # Use decompress without quantization (since we already dequantized)
        return self.decompress(
            p,
            idx_concat,
            val_concat,
            xshape,
            totalk,
            quantize_params=None,
            reduce=reduce_mode,
        )

    @torch.no_grad()
    def _quantize_values(self, val: torch.Tensor) -> tuple[torch.Tensor, QuantParamsT]:
        """
        Quantize tensor values to 8-bit integers.

        Args:
            val (torch.Tensor): The tensor values to quantize.

        Returns:
            A tuple containing the quantized values (uint8) and the
            quantization parameters. When ``self.pack_values_2bit`` is on
            (only valid for ``n_bins == 4``) the returned values are
            additionally bit-packed 4-per-byte along the last dim, and
            the qparams tuple is extended from 5 to 7 elements with
            ``(pack_version=1, original_last_dim)`` so the receiver can
            invert the packing. The quantisation math itself is
            unchanged in either path.
        """
        offset = self.n_bins // 2  # 128 for 8-bit
        shift = val.mean()
        centered = val - shift

        std = centered.norm() / math.sqrt(centered.numel() - 1)
        scale = self.range_in_sigmas * std / self.n_bins
        if scale == 0 or torch.isnan(scale) or torch.isinf(scale):
            scale = torch.tensor(1.0, dtype=centered.dtype, device=val.device)

        centered_fp32 = centered.to(torch.float32)
        qval = (
            (centered_fp32 / scale + offset)
            .round()
            .clamp(0, self.n_bins - 1)
            .to(torch.uint8)
        )

        device = qval.device
        sums = torch.zeros(self.n_bins, dtype=torch.float32, device=device)
        counts = torch.zeros(self.n_bins, dtype=torch.float32, device=device)

        sums.scatter_add_(0, qval.flatten().long(), centered_fp32.flatten())
        counts.scatter_add_(
            0, qval.flatten().long(), torch.ones_like(centered_fp32.flatten())
        )

        lookup = torch.where(counts > 0, sums / counts, torch.zeros_like(sums))

        # P0b: optionally bit-pack the qval tensor down to true 2-bit
        # storage. Only valid for n_bins=4 (every quantised entry is in
        # [0, 3]); the constructor already ANDs with ``n_bins == 4`` so
        # ``self.pack_values_2bit`` can never be True otherwise.
        if self.pack_values_2bit:
            original_last_dim = int(qval.shape[-1])
            qval = pack_2bit_values(qval)
            qparams_packed: PackedQuantParamsT = (
                shift,
                float(scale),
                offset,
                lookup,
                val.dtype,
                PACK_VERSION_2BIT,
                original_last_dim,
            )
            return qval, qparams_packed

        qparams: LegacyQuantParamsT = (shift, float(scale), offset, lookup, val.dtype)
        return qval, qparams

    @torch.no_grad()
    def _dequantize_values(
        self, val: torch.Tensor, qparams: QuantParamsT
    ) -> torch.Tensor:
        """
        Dequantize tensor values from 8-bit integers back to their original dtype.

        Accepts both wire variants of ``QuantParamsT``:

        * Legacy 5-tuple ``(shift, scale, offset, lookup, dtype)`` -- ``val``
          is interpreted as raw uint8 quantised entries with the same shape
          as the original quantised tensor.
        * Packed 7-tuple ``(shift, scale, offset, lookup, dtype, pack_version,
          original_last_dim)`` -- when ``pack_version == PACK_VERSION_2BIT``,
          ``val`` is first 2-bit-unpacked along the last dim using
          ``original_last_dim`` to recover the pre-pack uint8 shape, then
          dequantised through ``lookup`` like the legacy path.

        The same compressor instance can mix both formats peer-by-peer
        during the rollout window. The output shape always matches the
        shape the caller would have seen pre-quantisation.

        Args:
            val (torch.Tensor): The quantized values (uint8) -- raw or
                2-bit-packed depending on ``qparams`` length.
            qparams (QuantParamsT): The quantization parameters.

        Returns:
            torch.Tensor: The dequantized values.
        """
        if val.dtype == torch.uint8:
            # Tolerant destructure: works for both 5-tuple (legacy) and
            # 7-tuple (PACK_VERSION_2BIT). ``extras`` empty => v0 by
            # construction; otherwise ``extras = (pack_version, original_last_dim)``.
            shift, _scale, _offset, lookup, orig_dtype, *extras = qparams

            if extras:
                pack_version = int(extras[0])
                if pack_version == PACK_VERSION_2BIT:
                    original_last_dim = int(extras[1])
                    val = unpack_2bit_values(val, original_last_dim)
                elif pack_version != PACK_VERSION_LEGACY:
                    # Future-proofing: a sender from a newer codec sent an
                    # unknown pack_version. Fail loudly rather than silently
                    # corrupting weights via a wrong lookup.
                    raise ValueError(
                        f"Unknown pack_version {pack_version} in qparams; "
                        f"upgrade hone to decode this gradient."
                    )

            lookup = (
                lookup.to(val.device) if isinstance(lookup, torch.Tensor) else lookup
            )
            deq = lookup[val.long()] + shift
            val = deq.to(orig_dtype)
        return val

    def maybe_dequantize_values(
        self,
        vals: list[torch.Tensor],
        qparams: list[QuantParamsT],
        device: torch.device,
    ) -> list[torch.Tensor]:
        """
        Dequantize a list of values if they are quantized.

        Args:
            vals (list[torch.Tensor]): A list of tensors that may be quantized.
            qparams (list[QuantParamsT]): A list of quantization parameters.
            device (torch.device): The device to move the tensors to.

        Returns:
            list[torch.Tensor]: A list of dequantized tensors.
        """
        if not isinstance(vals, (list, tuple)):
            vals = [vals]

        needs_dequantized = all([v.dtype == torch.uint8 for v in vals])
        if qparams is None or not needs_dequantized:
            return vals

        # A single QuantParamsT is itself a tuple whose first element is a
        # tensor (``shift``); a list-of-qparams is a list whose elements
        # are themselves tuples. Detect-by-element-type so we work for
        # both 5-tuple (legacy) and 7-tuple (PACK_VERSION_2BIT) shapes
        # without hard-coding the length here.
        if isinstance(qparams, tuple) and qparams and isinstance(qparams[0], torch.Tensor):
            qparams = [qparams]

        if not isinstance(qparams, list):
            qparams = [qparams]

        vals_f32: list[torch.Tensor] = []
        for i, v in enumerate(vals):
            v = v.to(device)
            if v.dtype == torch.uint8:  # still quantised → decode
                if qparams[i] is None:
                    hone.logger.warning(f"Missing quant_params for vals[{i}]]; skip.")
                    break
                qp = qparams[i]
                v = self._dequantize_values(v, qp).to(device)
            vals_f32.append(v)

        if len(vals_f32) != len(vals):  # some decode failed
            raise IndexError(
                f"Mismatch in val lengths: dequant({len(vals_f32)}) vs original({len(vals)})"
            )

        return vals_f32


# Code modified and sourced from https://github.com/zh217/torch-dct
def _dct_fft_impl(v) -> torch.Tensor:
    """FFT-based implementation of the DCT."""
    return torch.view_as_real(torch.fft.fft(v, dim=1))


def _idct_irfft_impl(V) -> torch.Tensor:
    """IRFFT-based implementation of the IDCT."""
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)


def _dct(x, norm=None) -> torch.Tensor:
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = _dct_fft_impl(v)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= math.sqrt(N) * 2
        V[:, 1:] /= math.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def _idct(X, norm=None) -> torch.Tensor:
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= math.sqrt(N) * 2
        X_v[:, 1:] *= math.sqrt(N / 2) * 2

    k = (
        torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :]
        * math.pi
        / (2 * N)
    )
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = _idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


def _get_prime_divisors(n: int) -> list[int]:
    """
    Get the prime divisors of a number.

    Args:
        n (int): The number to factorize.

    Returns:
        list[int]: A list of prime divisors.
    """
    divisors = []
    while n % 2 == 0:
        divisors.append(2)
        n //= 2
    while n % 3 == 0:
        divisors.append(3)
        n //= 3
    i = 5
    while i * i <= n:
        for k in (i, i + 2):
            while n % k == 0:
                divisors.append(k)
                n //= k
        i += 6
    if n > 1:
        divisors.append(n)
    return divisors


def _get_divisors(n: int) -> list[int]:
    """
    Get all divisors of a number.

    Args:
        n (int): The number to get divisors for.

    Returns:
        list[int]: A sorted list of divisors.
    """
    divisors = []
    if n == 1:
        divisors.append(1)
    elif n > 1:
        prime_factors = _get_prime_divisors(n)
        divisors = [1]
        last_prime = 0
        factor = 0
        slice_len = 0
        # Find all the products that are divisors of n
        for prime in prime_factors:
            if last_prime != prime:
                slice_len = len(divisors)
                factor = prime
            else:
                factor *= prime
            for i in range(slice_len):
                divisors.append(divisors[i] * factor)
            last_prime = prime
        divisors.sort()
    return divisors


def _get_smaller_split(n: int, close_to: int) -> int:
    """
    Find the largest divisor of n that is less than or equal to close_to.

    Args:
        n (int): The number to find a divisor for.
        close_to (int): The target value to be close to.

    Returns:
        int: The largest divisor of n that is <= close_to.
    """
    all_divisors = _get_divisors(n)
    for ix, val in enumerate(all_divisors):
        if val == close_to:
            return val
        if val > close_to:
            if ix == 0:
                return val
            return all_divisors[ix - 1]
    return n


# ─────────── self-test entrypoint ────────────────────────────────────────
# `hone` has no stand-alone test suite under ``hone/tests/``; the existing
# test coverage for the compressor lives in the sibling ``templar`` repo.
# Until that infrastructure lands here, expose the round-trip checks as a
# `python -m hone.compress` runnable so the P0b 2-bit packing is verified
# locally and in CI without pulling in pytest.
def _run_self_tests() -> None:  # pragma: no cover - exercised via __main__
    """Round-trip checks for the P0b 2-bit value packer.

    Validates:
    1. ``pack_2bit_values`` / ``unpack_2bit_values`` are exact inverses
       across edge-case sizes (powers of 4, +/-1 from a power of 4, etc).
    2. The packer preserves leading dims so the comms.py
       ``vals.shape[:-1] == xshape[:-1]`` invariant survives.
    3. The end-to-end ``TopKCompressor`` decompress path accepts BOTH the
       legacy 5-tuple (raw uint8 vals) and the new 7-tuple
       (``PACK_VERSION_2BIT``) qparams from the same instance — i.e. the
       backward-compat contract holds for mixed-format gather batches.
    """
    torch.manual_seed(0)

    # --- (1) raw pack/unpack round-trip --------------------------------
    # Sizes intentionally bracket every multiple-of-4 boundary the
    # padder might encounter (1..5, 63..65, 1000..1003 covers all
    # pad-by-{0,1,2,3} cases).
    for n in [1, 2, 3, 4, 5, 63, 64, 65, 1000, 1001, 1002, 1003]:
        vals_1d = torch.randint(0, 4, (n,), dtype=torch.uint8)
        packed = pack_2bit_values(vals_1d)
        unpacked = unpack_2bit_values(packed, n)
        assert torch.equal(vals_1d, unpacked), (
            f"1D round-trip failed at n={n}: vals={vals_1d.tolist()} "
            f"unpacked={unpacked.tolist()}"
        )
        # 4-values-per-byte invariant after padding
        assert packed.numel() == (n + 3) // 4, (
            f"unexpected packed size {packed.numel()} for n={n}"
        )

    # --- (2) leading-dim preservation (the comms.py shape contract) ----
    for shape in [(7, 32), (3, 1, 30), (5, 17), (2, 2, 2, 6)]:
        vals_nd = torch.randint(0, 4, shape, dtype=torch.uint8)
        packed = pack_2bit_values(vals_nd)
        assert packed.shape[:-1] == vals_nd.shape[:-1], (
            f"leading dims mutated: {packed.shape} vs {vals_nd.shape}"
        )
        assert packed.shape[-1] == (vals_nd.shape[-1] + 3) // 4
        unpacked = unpack_2bit_values(packed, vals_nd.shape[-1])
        assert torch.equal(vals_nd, unpacked), f"ND round-trip failed for {shape}"

    # --- (3) end-to-end compressor cross-format compat -----------------
    # Drive a small synthetic gradient through both code paths and check
    # the decompressor emits the SAME dense output. This is what gather
    # actually depends on during the rollout window: a single decompressor
    # consuming both wire variants from different peers.
    x = torch.randn(8, 64)
    topk = 32
    legacy = TopKCompressor(
        use_quantization=True, quantization_bins=4, quantization_range=6,
    )
    packed_compressor = TopKCompressor(
        use_quantization=True, quantization_bins=4, quantization_range=6,
        pack_values_2bit=True,
    )

    idx_l, val_l, xshape_l, totalk_l, qp_l = legacy.compress(x, topk)
    idx_p, val_p, xshape_p, totalk_p, qp_p = packed_compressor.compress(x, topk)

    # Wire-format invariants
    assert len(qp_l) == 5, f"legacy qparams should be 5-tuple, got len={len(qp_l)}"
    assert len(qp_p) == 7, f"packed qparams should be 7-tuple, got len={len(qp_p)}"
    assert qp_p[5] == PACK_VERSION_2BIT
    assert qp_p[6] == val_l.shape[-1], "original_last_dim mismatch"
    assert val_l.dtype == torch.uint8 and val_p.dtype == torch.uint8
    # Last-dim compression: packed should be 4× smaller (with possible
    # +1 for padding that pushes into the next byte).
    assert val_p.shape[-1] == (val_l.shape[-1] + 3) // 4, (
        f"expected packed last-dim {(val_l.shape[-1] + 3) // 4}, "
        f"got {val_p.shape[-1]}"
    )
    # Leading dims preserved
    assert val_p.shape[:-1] == val_l.shape[:-1]

    # Either decompressor instance must accept either format.
    p_ref = torch.zeros_like(x)
    out_legacy_via_packed = packed_compressor.decompress(
        p_ref, idx_l, val_l, xshape_l, totalk_l, qp_l
    )
    out_packed_via_legacy = legacy.decompress(
        p_ref, idx_p, val_p, xshape_p, totalk_p, qp_p
    )
    out_legacy_native = legacy.decompress(
        p_ref, idx_l, val_l, xshape_l, totalk_l, qp_l
    )
    out_packed_native = packed_compressor.decompress(
        p_ref, idx_p, val_p, xshape_p, totalk_p, qp_p
    )
    # Same input → same quantised values regardless of wire packing.
    assert torch.allclose(out_legacy_via_packed, out_legacy_native), (
        "decompressor mishandled legacy 5-tuple qparams"
    )
    assert torch.allclose(out_packed_via_legacy, out_packed_native), (
        "decompressor mishandled PACK_VERSION_2BIT 7-tuple qparams"
    )
    assert torch.allclose(out_legacy_native, out_packed_native), (
        "2-bit packing changed the recovered quantised gradient"
    )

    # --- (4) batch_decompress mixed-format batch -----------------------
    # Simulate a gather batch with one legacy peer and one packed peer.
    out_batch = packed_compressor.batch_decompress(
        p_ref,
        [idx_l, idx_p],
        [val_l, val_p],
        xshape_l,
        totalk_l,
        quantize_params=[qp_l, qp_p],
        clip_norm=False,
    )
    assert out_batch.shape == x.shape

    # --- (5) wire-size sanity --------------------------------------- --
    legacy_bytes = val_l.numel()
    packed_bytes = val_p.numel()
    ratio = packed_bytes / max(legacy_bytes, 1)
    assert ratio <= 0.30, (  # ~0.25 ideal; allow slack for padding.
        f"expected ~4× values shrinkage, got ratio={ratio:.3f} "
        f"(legacy={legacy_bytes}B, packed={packed_bytes}B)"
    )

    # --- (6) check_compressed_indices shape derivation under 2-bit ----
    # Regression for Rollout 2 (2026-05-02): if a downstream caller
    # forgets to pass ``qparams`` to ``check_compressed_indices`` the
    # packed wire ``vals.shape[-1]`` is 4× too small and the 12-bit
    # unpacker raises "expanded size ... must match existing size ..."
    # with EXACTLY a 4× ratio. This catches any future regression of
    # either the qparams plumbing or the shape-derivation math.
    assert qp_p[5] == PACK_VERSION_2BIT and len(qp_p) == 7, (
        "compress() did not produce a 7-tuple under pack_values_2bit=True"
    )

    # WITHOUT qparams override: indices_shape would come from packed
    # vals last dim → 4× too small → unpack throws. Asserting the
    # throw guards both directions of the bug (a future change that
    # silently accepts the wrong shape would ALSO be wrong).
    wrong_shape = val_p.shape
    try:
        unpack_12bit_indices(idx_p, wrong_shape)
    except RuntimeError as e:
        _msg = str(e)
        assert "expanded size" in _msg or "must match" in _msg, _msg
    else:  # pragma: no cover
        raise AssertionError(
            "Expected unpack_12bit_indices to fail with packed-vals shape; "
            "got success. Either pack_2bit_values no longer shrinks the last "
            "dim, or the 12-bit unpacker silently accepts mismatched shapes "
            "-- both mask the Rollout 2 bug."
        )

    # WITH qparams[6] override: indices_shape is the ORIGINAL last dim,
    # unpack succeeds, shape matches the legacy-path unpacked shape.
    correct_shape = (*val_p.shape[:-1], int(qp_p[6]))
    unpacked_ok = unpack_12bit_indices(idx_p, correct_shape)
    assert unpacked_ok.shape[-1] == int(qp_p[6])
    assert unpacked_ok.shape[:-1] == val_p.shape[:-1]

    print(
        "[hone.compress] self-tests passed: pack_2bit_values round-trips, "
        f"end-to-end shrinks values {legacy_bytes}B -> {packed_bytes}B "
        f"({ratio:.1%} of legacy); check_compressed_indices qparams "
        "plumbing regression covered."
    )


if __name__ == "__main__":  # pragma: no cover
    _run_self_tests()
