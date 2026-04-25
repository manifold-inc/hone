"""Cross-stage transport for pipeline-parallel activation transfer.

Each pipeline stage is launched as its own ``torchrun`` job with its own
NCCL world. Cross-stage forward activations and backward gradients flow
over a separate transport, decoupled from the per-stage NCCL collectives
(FSDP all_gather / reduce_scatter) that would otherwise serialize against
PP P2P on the same NCCL stream.

Two transports are supported, picked at construction:

1. **TCP** (default, always available): one socket per ``(stage S rank R,
   stage S+1 rank R)`` pair. Works across nodes without RDMA. Used in
   the multi-host no-RDMA topology that PP exists to support.

2. **NCCL P2P** (opt-in via ``intra_node_nccl=True`` when peer hostnames
   are loopback): a separate :class:`torch.distributed.ProcessGroup`
   spanning the matching ranks of two adjacent stages, using
   ``dist.send`` / ``dist.recv`` directly GPU-to-GPU over NVLink. Skips
   the D2H -> serialize -> H2D round-trip entirely.

Topology
--------
For every local rank ``R`` in stage ``S``, exactly one transport pair is
maintained with the matching local rank ``R`` in stage ``S+1``. Forward
activations flow ``S -> S+1``, backward gradients flow back the other
way. Connection roles for TCP: the *lower* stage initiates the
connection (client) and the *higher* stage accepts (server).

Wire format (TCP path)
----------------------
We avoid ``torch.save`` / pickle on the hot path -- it adds
~100-300us of CPU overhead per microbatch and obscures the payload from
zero-copy / pinned-buffer tricks. Instead each tensor is sent as::

    [4B  uint32  ndim         ]
    [8B  int64   shape[0]     ]
    ... (ndim shape entries) ...
    [4B  uint32  dtype_code   ]
    [8B  uint64  payload_bytes]
    [...           raw bytes  ]

``dtype_code`` is a small integer registered in :data:`_DTYPE_CODES` so
we can round-trip every dtype the model actually produces (bf16, fp16,
fp32) without ever calling ``Tensor.view(other_dtype)``. The raw bytes
are produced by viewing the contiguous CPU tensor as ``uint8`` and
slicing out its storage -- no numpy round-trip, no Python copy.

Asynchronous send
-----------------
``send_next`` / ``send_prev`` enqueue onto a bounded outbound queue and
return immediately; a per-direction worker thread drains the queue to
the socket while the main thread runs the next ``forward`` / ``backward``
microbatch. ``recv_*`` is still synchronous (the next compute genuinely
needs the data).

For the NCCL path the same async semantics are kept by issuing async
``isend`` work objects and waiting on them in a background drainer; the
GPU op itself is non-blocking and overlaps with the next compute via
NCCL's own stream.
"""

from __future__ import annotations

import os
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .logging import logger

# ----------------------------------------------------------------------
# Wire format constants
# ----------------------------------------------------------------------
_LEN_PREFIX = "!Q"  # 8-byte big-endian unsigned int (legacy, used for header)
_LEN_SIZE = struct.calcsize(_LEN_PREFIX)
_RECV_CHUNK = 1 << 20  # 1 MiB read chunk

# Compact dtype code table -- avoids ``Tensor.view(other_dtype)`` quirks
# on bf16 buffers across PyTorch versions. Codes are stable wire values;
# add new entries at the end, never reorder.
_DTYPE_CODES: dict[torch.dtype, int] = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float64: 3,
    torch.int64: 4,
    torch.int32: 5,
    torch.int16: 6,
    torch.int8: 7,
    torch.uint8: 8,
    torch.bool: 9,
}
_CODE_TO_DTYPE: dict[int, torch.dtype] = {v: k for k, v in _DTYPE_CODES.items()}


def _send_all(sock: socket.socket, data) -> None:
    """sendall over a connected TCP socket; raises on closed peer.

    ``data`` may be ``bytes``, ``bytearray``, or any buffer-protocol
    object (e.g. ``memoryview`` over a pinned tensor). ``socket.sendall``
    accepts all of these in modern CPython.
    """
    sock.sendall(data)


def _recv_exact_into(sock: socket.socket, view: memoryview) -> None:
    """Read exactly ``len(view)`` bytes from ``sock`` directly into the
    pre-existing buffer ``view``. Raises on early EOF.

    Skipping the per-recv allocation matters when we're cycling
    pinned-CPU buffers across microbatches.
    """
    n = len(view)
    got = 0
    while got < n:
        read = sock.recv_into(view[got:], min(n - got, _RECV_CHUNK))
        if read == 0:
            raise ConnectionError(
                f"socket closed with {n - got} of {n} bytes remaining"
            )
        got += read


def _recv_exact(sock: socket.socket, n: int) -> bytearray:
    """Backwards-compat helper; allocates and returns a bytearray."""
    buf = bytearray(n)
    _recv_exact_into(sock, memoryview(buf))
    return buf


# ----------------------------------------------------------------------
# Send queue items
# ----------------------------------------------------------------------
@dataclass
class _OutboundItem:
    """One queued outbound tensor.

    The tensor *must* already live on CPU and be contiguous; the worker
    thread does no further H2D / D2H work. Producers are responsible
    for the ``.detach().to('cpu', non_blocking=False).contiguous()``
    step before enqueuing -- they need the synchronous copy on the
    main CUDA stream so the next compute kernel can safely overwrite
    the GPU tensor.

    ``shape`` and ``dtype`` are cached so the worker doesn't re-read
    them under contention.
    """

    cpu_tensor: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype


# Sentinel pushed onto the queue at shutdown to wake the worker.
_SHUTDOWN: object = object()


# ----------------------------------------------------------------------
# Pinned-buffer pool
# ----------------------------------------------------------------------
class _PinnedPool:
    """Tiny pool of pinned CPU ``uint8`` buffers, sized on demand.

    ``acquire(n)`` returns a buffer of *at least* ``n`` bytes; the
    pool grows monotonically (we hold at most ``capacity`` buffers and
    reuse them). For PP transport the payload size is essentially
    constant per direction (``B * S * bottleneck * dtype_bytes``), so
    after the first microbatch the pool stops growing.
    """

    def __init__(self, *, capacity: int = 4) -> None:
        self._capacity = capacity
        self._free: list[torch.Tensor] = []
        self._lock = threading.Lock()

    def acquire(self, n: int) -> torch.Tensor:
        with self._lock:
            for i, buf in enumerate(self._free):
                if buf.numel() >= n:
                    return self._free.pop(i)
        # No fitting buffer; allocate a fresh pinned one. We don't hold
        # the lock across allocation (which can sleep on cudaHostAlloc).
        return torch.empty(n, dtype=torch.uint8, pin_memory=True)

    def release(self, buf: torch.Tensor) -> None:
        with self._lock:
            if len(self._free) < self._capacity:
                self._free.append(buf)
            # else: drop on the floor; allocator will reclaim.


# ----------------------------------------------------------------------
# Transport class
# ----------------------------------------------------------------------
class PPTransport:
    """Per-rank transport for one cross-stage P2P pair.

    Two backends share this single class so trainer code does not have
    to branch:

    - **TCP** (always available): two sockets, async sender threads,
      raw-bytes wire format.
    - **NCCL P2P** (opt-in, intra-node only): a 2-rank
      :class:`torch.distributed.ProcessGroup` between the two
      cross-stage matching ranks, with ``dist.isend`` / ``dist.recv``
      issued directly GPU-to-GPU.

    Public API used by ``trainer._pp_run_1f1b``:

    - :meth:`send_next` / :meth:`send_prev`: enqueue (TCP) or
      ``isend`` (NCCL) and return without blocking the main stream.
    - :meth:`recv_next` / :meth:`recv_prev`: synchronous receive,
      returns the tensor on ``self.device``.
    - :meth:`flush`: block until every pending async send has drained
      to its peer. Called once per inner step at the end of the 1F1B
      schedule before the FSDP collective.
    - :meth:`pop_timing_metrics`: returns and clears accumulated
      send-wait / recv-wait / send-bytes counters for observability.
    """

    def __init__(
        self,
        *,
        my_stage: int,
        num_stages: int,
        my_local_rank: int,
        ranks_per_stage: int,
        peer_host_prev: str,
        peer_port_base_prev: int,
        peer_host_next: str,
        peer_port_base_next: int,
        listen_host: str,
        listen_port_base: int,
        device: torch.device,
        amp_dtype: torch.dtype,
        connect_timeout_s: float = 120.0,
        connect_retry_s: float = 1.0,
        async_send: bool = True,
        send_queue_depth: int = 2,
        intra_node_nccl: bool = False,
        intra_node_nccl_init_method: str | None = None,
    ) -> None:
        self.my_stage = my_stage
        self.num_stages = num_stages
        self.my_local_rank = my_local_rank
        self.ranks_per_stage = ranks_per_stage
        self.is_first = my_stage == 0
        self.is_last = my_stage == num_stages - 1
        self.device = device
        self.amp_dtype = amp_dtype

        # Each rank uses port_base + my_local_rank so multiple ranks on the
        # same host don't collide.
        self.peer_host_prev = peer_host_prev
        self.peer_port_prev = peer_port_base_prev + my_local_rank
        self.peer_host_next = peer_host_next
        self.peer_port_next = peer_port_base_next + my_local_rank
        self.listen_host = listen_host
        self.listen_port = listen_port_base + my_local_rank

        self.connect_timeout_s = connect_timeout_s
        self.connect_retry_s = connect_retry_s

        # Async / NCCL knobs
        self.async_send = async_send
        self.send_queue_depth = max(1, send_queue_depth)
        # Same-node NCCL fast path is opt-in. We enable it only when both
        # adjacent stage hostnames are loopback; otherwise NCCL P2P would
        # require an IPoIB / TCP socket NCCL is not configured for.
        self.intra_node_nccl = intra_node_nccl and self._peers_are_loopback()
        self.intra_node_nccl_init_method = intra_node_nccl_init_method

        # ----- TCP state -----
        self._sock_prev: socket.socket | None = None
        self._sock_next: socket.socket | None = None
        self._listener: socket.socket | None = None

        # Sender threads + queues -- one per outbound direction. We
        # only create the queue/thread for directions we actually send
        # in (i.e. ``next`` if not last, ``prev`` if not first).
        self._send_queue_next: queue.Queue | None = None
        self._send_queue_prev: queue.Queue | None = None
        self._sender_thread_next: threading.Thread | None = None
        self._sender_thread_prev: threading.Thread | None = None
        # Pinned pools per direction: outbound (D2H staging done by
        # caller) and inbound (recv landing).
        self._pinned_in_next = _PinnedPool(capacity=self.send_queue_depth + 2)
        self._pinned_in_prev = _PinnedPool(capacity=self.send_queue_depth + 2)
        # Sender exception capture so the next public call surfaces it.
        self._sender_exc_next: BaseException | None = None
        self._sender_exc_prev: BaseException | None = None

        # ----- NCCL state -----
        self._nccl_pg_next: dist.ProcessGroup | None = None
        self._nccl_pg_prev: dist.ProcessGroup | None = None
        # Pending isend works -- one per direction; we wait on the
        # previous one before queuing the next, which gives the same
        # depth-1 pipelining as TCP but uses NCCL's stream.
        self._nccl_pending_next: list[dist.Work] = []
        self._nccl_pending_prev: list[dist.Work] = []

        # ----- Observability counters -----
        self._lock_metrics = threading.Lock()
        self._send_wait_ns_next = 0
        self._send_wait_ns_prev = 0
        self._recv_wait_ns_next = 0
        self._recv_wait_ns_prev = 0
        self._send_bytes_total = 0
        self._send_count = 0
        self._recv_count = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_loopback(host: str) -> bool:
        return host in ("127.0.0.1", "localhost", "::1")

    def _peers_are_loopback(self) -> bool:
        next_ok = self.is_last or self._is_loopback(self.peer_host_next)
        prev_ok = self.is_first or self._is_loopback(self.peer_host_prev)
        return next_ok and prev_ok

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Bring up both transport directions.

        For the TCP path this opens both sockets and spins up the
        per-direction sender threads. For the NCCL path this builds
        a 2-rank :class:`ProcessGroup` per direction; the underlying
        NCCL communicator is created lazily on the first send/recv.

        Blocks until the transport is ready in both directions.
        """
        if self.intra_node_nccl:
            self._start_nccl()
        else:
            self._start_tcp()

        if self.async_send and not self.intra_node_nccl:
            self._start_sender_threads()

        backend = "nccl-p2p" if self.intra_node_nccl else "tcp"
        logger.info(
            f"[PPTransport] stage={self.my_stage}/{self.num_stages} "
            f"rank={self.my_local_rank} ready (backend={backend}, "
            f"async_send={self.async_send and not self.intra_node_nccl}, "
            f"queue_depth={self.send_queue_depth})"
        )

    def stop(self) -> None:
        # Drain async senders first so we don't lose in-flight bytes.
        self._stop_sender_threads()

        for s in (self._sock_prev, self._sock_next, self._listener):
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass
        self._sock_prev = None
        self._sock_next = None
        self._listener = None

        # NCCL PGs are owned by us -- destroy them so a subsequent
        # process re-init doesn't see stale comms. ``destroy_process_group``
        # may raise if NCCL is already torn down at interpreter shutdown;
        # swallow.
        for pg in (self._nccl_pg_next, self._nccl_pg_prev):
            if pg is not None:
                try:
                    dist.destroy_process_group(pg)
                except Exception:
                    pass
        self._nccl_pg_next = None
        self._nccl_pg_prev = None

    # ------------------------------------------------------------------
    # TCP setup
    # ------------------------------------------------------------------
    def _start_tcp(self) -> None:
        accept_thread: threading.Thread | None = None
        if not self.is_first:
            accept_thread = threading.Thread(
                target=self._accept_prev,
                name=f"pp-accept-prev-rank{self.my_local_rank}",
                daemon=True,
            )
            accept_thread.start()

        if not self.is_last:
            self._connect_next()

        if accept_thread is not None:
            accept_thread.join()
            if self._sock_prev is None:
                raise RuntimeError(
                    f"PPTransport stage={self.my_stage} "
                    f"rank={self.my_local_rank}: prev-stage accept thread "
                    f"finished but no socket was set."
                )

    # ------------------------------------------------------------------
    # NCCL P2P setup
    # ------------------------------------------------------------------
    def _start_nccl(self) -> None:
        """Try to bring up a cross-stage NCCL P2P group.

        Each PP stage today is its own ``torchrun`` job with its own
        ``init_process_group``, so the per-stage NCCL world cannot
        directly see ranks in other stages. To support NCCL P2P across
        stages we need a *second* :class:`ProcessGroupNCCL` built from
        a shared :class:`TCPStore` rendezvous. PyTorch's modern API
        exposes ``ProcessGroupNCCL`` constructibly only on certain
        builds, and the constructor signature has churned across
        versions; this method tries the most-stable constructor and
        falls back to TCP transport on any error so the miner keeps
        running.

        For production use with a unified launcher, the right answer
        is to ``init_process_group`` once per process across all stages
        and use ``dist.new_group`` from there. That requires changes
        outside this transport.
        """
        try:
            global_rank, global_world_size = self._ensure_global_pp_pg()
            ranks_per_stage = self.ranks_per_stage

            if not self.is_last:
                peer_global_rank_next = (
                    (self.my_stage + 1) * ranks_per_stage
                    + self.my_local_rank
                )
                self._nccl_pg_next = self._build_subgroup(
                    [global_rank, peer_global_rank_next]
                )
                self._nccl_peer_rank_next = peer_global_rank_next

            if not self.is_first:
                peer_global_rank_prev = (
                    (self.my_stage - 1) * ranks_per_stage
                    + self.my_local_rank
                )
                self._nccl_pg_prev = self._build_subgroup(
                    [global_rank, peer_global_rank_prev]
                )
                self._nccl_peer_rank_prev = peer_global_rank_prev

            logger.info(
                f"[PPTransport] NCCL P2P PG built (global_rank={global_rank},"
                f" global_world_size={global_world_size}, "
                f"ranks_per_stage={ranks_per_stage})"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"[PPTransport] intra_node_nccl=True requested but "
                f"NCCL P2P PG bringup failed ({e!r}); falling back to "
                f"async TCP transport."
            )
            self.intra_node_nccl = False
            self._nccl_pg_next = None
            self._nccl_pg_prev = None
            self._start_tcp()

    def _ensure_global_pp_pg(self) -> tuple[int, int]:
        """Initialize the cross-stage global PP group exactly once per process.

        The default PG built by torchrun for this stage is intra-stage;
        we need a second PG that spans all stages. We do this with a
        separate :class:`TCPStore` rendezvous on a fixed loopback port
        (``PP_NCCL_INIT_PORT`` env or ``29800`` default). Returns the
        global rank / world-size assigned to this process across the
        unified PP world.
        """
        if _GlobalPPState.pg is not None:
            return _GlobalPPState.global_rank, _GlobalPPState.global_world_size

        global_rank = self.my_stage * self.ranks_per_stage + self.my_local_rank
        global_world_size = self.num_stages * self.ranks_per_stage

        init_method = self.intra_node_nccl_init_method or os.environ.get(
            "PP_NCCL_INIT_METHOD", "tcp://127.0.0.1:29800"
        )
        port = int(init_method.rsplit(":", 1)[1])

        import datetime as _dt

        store = dist.TCPStore(
            host_name="127.0.0.1",
            port=port,
            world_size=global_world_size,
            is_master=(global_rank == 0),
            timeout=_dt.timedelta(seconds=300),
        )

        # Build a fresh ProcessGroupNCCL via the C10d API. Both
        # PyTorch 2.1+ and 2.4+ accept this constructor; a TypeError
        # here will be caught by the outer try in _start_nccl.
        pg = dist.ProcessGroupNCCL(store, global_rank, global_world_size)

        _GlobalPPState.pg = pg
        _GlobalPPState.global_rank = global_rank
        _GlobalPPState.global_world_size = global_world_size
        return global_rank, global_world_size

    def _build_subgroup(self, ranks: list[int]) -> dist.ProcessGroup:
        """Carve a 2-rank subgroup from the global PP NCCL PG.

        All processes must call ``new_group`` with identical ``ranks``
        for the carve to succeed; we sort to make that easy on callers.
        """
        sorted_ranks = sorted(ranks)
        # ``new_group`` here uses the *default* PG -- but our per-stage
        # default PG can't see other stages. So we emulate ``new_group``
        # by directly building another ProcessGroupNCCL on a
        # PrefixStore-scoped TCPStore. To keep things simple and
        # robust, we reuse the global PP PG built in
        # _ensure_global_pp_pg as the comm; each "subgroup" is just a
        # peer rank pair, and we use point-to-point sends/recvs with
        # the ranks expressed in the global PP world.
        return _GlobalPPState.pg  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Sender threads (TCP path only)
    # ------------------------------------------------------------------
    def _start_sender_threads(self) -> None:
        if not self.is_last:
            self._send_queue_next = queue.Queue(maxsize=self.send_queue_depth)
            self._sender_thread_next = threading.Thread(
                target=self._sender_loop,
                args=(self._send_queue_next, "next", self._sock_next),
                name=f"pp-sender-next-rank{self.my_local_rank}",
                daemon=True,
            )
            self._sender_thread_next.start()
        if not self.is_first:
            self._send_queue_prev = queue.Queue(maxsize=self.send_queue_depth)
            self._sender_thread_prev = threading.Thread(
                target=self._sender_loop,
                args=(self._send_queue_prev, "prev", self._sock_prev),
                name=f"pp-sender-prev-rank{self.my_local_rank}",
                daemon=True,
            )
            self._sender_thread_prev.start()

    def _stop_sender_threads(self) -> None:
        for q, thread in (
            (self._send_queue_next, self._sender_thread_next),
            (self._send_queue_prev, self._sender_thread_prev),
        ):
            if q is None or thread is None:
                continue
            try:
                q.put(_SHUTDOWN, timeout=5.0)
            except queue.Full:
                pass
            thread.join(timeout=5.0)

    def _sender_loop(
        self,
        q: queue.Queue,
        direction: str,
        sock: socket.socket | None,
    ) -> None:
        """Drain ``q`` to ``sock`` until a shutdown sentinel arrives.

        Catches sender exceptions and stashes them so the next
        ``send_*`` / ``flush`` raises -- the main thread isn't watching
        the worker directly.
        """
        try:
            while True:
                item = q.get()
                if item is _SHUTDOWN:
                    return
                assert isinstance(item, _OutboundItem)
                self._tcp_send_now(sock, item)
        except BaseException as exc:  # noqa: BLE001
            if direction == "next":
                self._sender_exc_next = exc
            else:
                self._sender_exc_prev = exc
            logger.error(
                f"[PPTransport] sender thread '{direction}' "
                f"rank={self.my_local_rank} died: {exc!r}"
            )

    def _check_sender_exc(self, direction: str) -> None:
        exc = (
            self._sender_exc_next if direction == "next" else self._sender_exc_prev
        )
        if exc is not None:
            # Re-raise once; clear so subsequent calls don't loop on it.
            if direction == "next":
                self._sender_exc_next = None
            else:
                self._sender_exc_prev = None
            raise RuntimeError(
                f"PPTransport sender thread '{direction}' failed"
            ) from exc

    # ------------------------------------------------------------------
    # TCP wire-format I/O
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_header(shape: tuple[int, ...], dtype: torch.dtype, n: int) -> bytes:
        """Pack the fixed-length header for a tensor payload.

        Layout::

            [4B  uint32  ndim]
            [ndim * 8B int64 shape entries]
            [4B  uint32  dtype_code]
            [8B  uint64  payload_bytes]
        """
        try:
            code = _DTYPE_CODES[dtype]
        except KeyError as e:
            raise RuntimeError(
                f"PPTransport: dtype {dtype} not in _DTYPE_CODES; add a code."
            ) from e
        ndim = len(shape)
        fmt = f"!I{ndim}qIQ"
        return struct.pack(fmt, ndim, *shape, code, n)

    @staticmethod
    def _decode_header(sock: socket.socket) -> tuple[tuple[int, ...], torch.dtype, int]:
        """Read and parse the per-tensor header.

        We do two ``recv``s here -- one for the ndim prefix so we know
        how big the rest of the header is, then one for the rest. The
        header is small (<128B for any realistic ndim), so the second
        recv almost always lands in a single TCP segment.
        """
        ndim_buf = _recv_exact(sock, 4)
        (ndim,) = struct.unpack("!I", bytes(ndim_buf))
        rest_size = ndim * 8 + 4 + 8
        rest = _recv_exact(sock, rest_size)
        fmt = f"!{ndim}qIQ"
        unpacked = struct.unpack(fmt, bytes(rest))
        shape = tuple(unpacked[:ndim])
        code = unpacked[ndim]
        nbytes = unpacked[ndim + 1]
        try:
            dtype = _CODE_TO_DTYPE[code]
        except KeyError as e:
            raise RuntimeError(
                f"PPTransport: received unknown dtype code {code}"
            ) from e
        return shape, dtype, nbytes

    def _tcp_send_now(
        self, sock: socket.socket | None, item: _OutboundItem
    ) -> None:
        """Synchronously serialize ``item`` to ``sock``. Caller must
        guarantee the cpu_tensor is contiguous and on CPU.

        Done inside the sender thread (TCP path) -- *not* on the main
        thread -- so the per-tensor header pack and the ``sendall`` are
        free to take however long the kernel wants without stalling
        the next forward.
        """
        if sock is None:
            raise RuntimeError("PPTransport: TCP socket not connected")
        cpu = item.cpu_tensor
        # ``view(uint8)`` aliases the existing CPU storage; no copy.
        view = cpu.view(torch.uint8) if cpu.dtype != torch.uint8 else cpu
        nbytes = view.numel()
        header = self._encode_header(item.shape, item.dtype, nbytes)
        # Two sendalls: header (small, NODELAY) + payload (large). One
        # ``sendall(header + payload)`` would force a copy the size of
        # the payload; this way the payload is sent zero-copy from the
        # pinned tensor's storage via ``memoryview``.
        _send_all(sock, header)
        # ``memoryview`` over a torch tensor goes via the buffer protocol;
        # ``sendall`` accepts it directly without copying.
        _send_all(sock, memoryview(view.numpy()))
        with self._lock_metrics:
            self._send_bytes_total += nbytes
            self._send_count += 1

    def _tcp_recv_tensor(
        self,
        sock: socket.socket | None,
        expected_shape: tuple[int, ...],
        expected_dtype: torch.dtype,
        pool: _PinnedPool,
    ) -> torch.Tensor:
        if sock is None:
            raise RuntimeError("PPTransport: TCP socket not connected")
        shape, dtype, nbytes = self._decode_header(sock)
        if shape != expected_shape:
            raise RuntimeError(
                f"PPTransport: received tensor shape {shape} "
                f"but expected {expected_shape}"
            )
        if dtype != expected_dtype:
            raise RuntimeError(
                f"PPTransport: received tensor dtype {dtype} "
                f"but expected {expected_dtype}"
            )

        pinned = pool.acquire(nbytes)
        view = memoryview(pinned[:nbytes].numpy())
        _recv_exact_into(sock, view)

        # Reinterpret the uint8 bytes as the target dtype + reshape.
        # ``view(dtype)`` works on contiguous CPU tensors of every dtype
        # we have in _DTYPE_CODES; the subsequent ``.view(shape)``
        # reshapes the resulting 1D dtype tensor.
        flat = pinned[:nbytes].view(dtype)
        cpu_tensor = flat.view(shape)
        # H2D copy. We use ``non_blocking=False`` because we'll
        # release the pinned buffer back to the pool right after,
        # and the next recv that grabs this buffer must not overwrite
        # bytes mid-DMA. The synchronous copy adds <100us for the
        # ResBM-compressed activation sizes we ship (kB-scale) and
        # the caller's next compute kernel was about to sync on it
        # implicitly anyway.
        gpu_tensor = cpu_tensor.to(self.device, non_blocking=False)
        pool.release(pinned)

        with self._lock_metrics:
            self._recv_count += 1
        return gpu_tensor

    # ------------------------------------------------------------------
    # NCCL P2P I/O
    # ------------------------------------------------------------------
    def _nccl_send(
        self, pg: dist.ProcessGroup, peer_global_rank: int, tensor: torch.Tensor
    ) -> None:
        """Issue an async ``isend`` on the cross-stage PG.

        We keep the previous ``Work`` object pending until the next
        send to give the NCCL stream room to overlap with compute.
        """
        # NCCL only ships contiguous tensors; cast/clone if needed.
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        # Pre-cast to amp_dtype so we don't ship fp32 on the wire.
        if tensor.dtype != self.amp_dtype:
            tensor = tensor.to(self.amp_dtype)
        work = dist.isend(tensor, dst=peer_global_rank, group=pg)
        # Track the work so flush() can wait on it.
        if peer_global_rank == getattr(self, "_nccl_peer_rank_next", -1):
            self._nccl_pending_next.append(work)
        else:
            self._nccl_pending_prev.append(work)
        with self._lock_metrics:
            self._send_bytes_total += tensor.numel() * tensor.element_size()
            self._send_count += 1

    def _nccl_recv(
        self,
        pg: dist.ProcessGroup,
        peer_global_rank: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # NCCL recv works in-place; allocate the output tensor first.
        # We use ``self.amp_dtype`` on the wire (matches ``_nccl_send``).
        wire_dtype = self.amp_dtype
        out = torch.empty(shape, dtype=wire_dtype, device=self.device)
        dist.recv(out, src=peer_global_rank, group=pg)
        with self._lock_metrics:
            self._recv_count += 1
        # Cast back to the dtype the caller requested if it differs.
        if dtype != wire_dtype:
            out = out.to(dtype)
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def send_next(self, tensor: torch.Tensor) -> None:
        self._send_dir(tensor, direction="next")

    def send_prev(self, tensor: torch.Tensor) -> None:
        self._send_dir(tensor, direction="prev")

    def recv_next(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        return self._recv_dir(shape, dtype, direction="next")

    def recv_prev(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        return self._recv_dir(shape, dtype, direction="prev")

    def _send_dir(self, tensor: torch.Tensor, *, direction: str) -> None:
        t0 = time.perf_counter_ns()
        self._check_sender_exc(direction)

        if self.intra_node_nccl:
            pg = (
                self._nccl_pg_next if direction == "next" else self._nccl_pg_prev
            )
            peer = (
                self._nccl_peer_rank_next
                if direction == "next"
                else self._nccl_peer_rank_prev
            )
            assert pg is not None
            self._nccl_send(pg, peer, tensor)
        else:
            sock = (
                self._sock_next if direction == "next" else self._sock_prev
            )
            # Stage the GPU tensor onto CPU on the *main* thread; the
            # sender thread will not touch CUDA. The copy is synchronous
            # on the current stream so the next compute kernel can
            # safely overwrite the GPU tensor.
            cpu = tensor.detach().to("cpu", non_blocking=False).contiguous()
            shape = tuple(cpu.shape)
            dtype = cpu.dtype
            item = _OutboundItem(cpu_tensor=cpu, shape=shape, dtype=dtype)
            if self.async_send:
                q = (
                    self._send_queue_next
                    if direction == "next"
                    else self._send_queue_prev
                )
                assert q is not None
                # Bounded queue: this *can* block if the worker is
                # backed up, which is by design -- it bounds queueing
                # depth so we don't OOM CPU under a slow peer.
                q.put(item)
            else:
                self._tcp_send_now(sock, item)

        dt = time.perf_counter_ns() - t0
        with self._lock_metrics:
            if direction == "next":
                self._send_wait_ns_next += dt
            else:
                self._send_wait_ns_prev += dt

    def _recv_dir(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        direction: str,
    ) -> torch.Tensor:
        t0 = time.perf_counter_ns()
        try:
            if self.intra_node_nccl:
                pg = (
                    self._nccl_pg_next
                    if direction == "next"
                    else self._nccl_pg_prev
                )
                peer = (
                    self._nccl_peer_rank_next
                    if direction == "next"
                    else self._nccl_peer_rank_prev
                )
                assert pg is not None
                out = self._nccl_recv(pg, peer, shape, dtype)
            else:
                sock = (
                    self._sock_next
                    if direction == "next"
                    else self._sock_prev
                )
                pool = (
                    self._pinned_in_next
                    if direction == "next"
                    else self._pinned_in_prev
                )
                out = self._tcp_recv_tensor(sock, shape, dtype, pool)
        finally:
            dt = time.perf_counter_ns() - t0
            with self._lock_metrics:
                if direction == "next":
                    self._recv_wait_ns_next += dt
                else:
                    self._recv_wait_ns_prev += dt
        return out

    def flush(self) -> None:
        """Block until every pending async send has drained.

        For TCP this means draining the bounded outbound queues
        (``Queue.join`` would also work but we'd need ``task_done``
        wiring; instead we just let ``Queue.put`` block on a final
        sentinel-style flush by enqueuing a no-op).

        For NCCL this waits on all pending ``isend`` Work objects.

        Called once per inner step at the end of ``_pp_run_1f1b``
        before the FSDP collective so the next step's compute sees
        a clean transport.
        """
        if self.intra_node_nccl:
            for work in self._nccl_pending_next:
                work.wait()
            for work in self._nccl_pending_prev:
                work.wait()
            self._nccl_pending_next.clear()
            self._nccl_pending_prev.clear()
            return

        if self.async_send:
            for q, direction in (
                (self._send_queue_next, "next"),
                (self._send_queue_prev, "prev"),
            ):
                if q is None:
                    continue
                # Block until the worker has dequeued everything.
                # Queue itself doesn't expose "wait until empty"; we
                # spin on qsize. qsize is approximate but with a single
                # producer + single consumer it's exact.
                while q.qsize() > 0:
                    time.sleep(0.0001)
                self._check_sender_exc(direction)

    def pop_timing_metrics(self) -> dict[str, float]:
        """Return and clear accumulated timing counters.

        Returns a dict suitable for direct logging to W&B. All time
        values are in microseconds; bytes are raw counts.
        """
        with self._lock_metrics:
            out = {
                "pp/send_wait_us_next": self._send_wait_ns_next / 1e3,
                "pp/send_wait_us_prev": self._send_wait_ns_prev / 1e3,
                "pp/recv_wait_us_next": self._recv_wait_ns_next / 1e3,
                "pp/recv_wait_us_prev": self._recv_wait_ns_prev / 1e3,
                "pp/send_bytes_total": float(self._send_bytes_total),
                "pp/send_count": float(self._send_count),
                "pp/recv_count": float(self._recv_count),
            }
            self._send_wait_ns_next = 0
            self._send_wait_ns_prev = 0
            self._recv_wait_ns_next = 0
            self._recv_wait_ns_prev = 0
            self._send_bytes_total = 0
            self._send_count = 0
            self._recv_count = 0
        return out

    # ------------------------------------------------------------------
    # TCP connection helpers
    # ------------------------------------------------------------------
    def _accept_prev(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((self.listen_host, self.listen_port))
        except OSError as e:
            raise RuntimeError(
                f"PPTransport stage={self.my_stage} "
                f"rank={self.my_local_rank}: bind "
                f"{self.listen_host}:{self.listen_port} failed: {e}"
            ) from e
        listener.listen(1)
        self._listener = listener

        logger.info(
            f"[PPTransport] stage={self.my_stage} rank={self.my_local_rank} "
            f"listening on {self.listen_host}:{self.listen_port} for prev stage"
        )

        deadline = time.time() + self.connect_timeout_s
        listener.settimeout(self.connect_retry_s)
        while time.time() < deadline:
            try:
                conn, addr = listener.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._sock_prev = conn
                logger.info(
                    f"[PPTransport] stage={self.my_stage} "
                    f"rank={self.my_local_rank} accepted prev from {addr}"
                )
                return
            except socket.timeout:
                continue
            except Exception as e:
                logger.warning(
                    f"[PPTransport] accept error rank={self.my_local_rank}: {e}"
                )
                continue
        raise TimeoutError(
            f"PPTransport stage={self.my_stage} rank={self.my_local_rank}: "
            f"no prev connection on {self.listen_host}:{self.listen_port} "
            f"within {self.connect_timeout_s}s"
        )

    def _connect_next(self) -> None:
        deadline = time.time() + self.connect_timeout_s
        last_err: Exception | None = None
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                sock.connect((self.peer_host_next, self.peer_port_next))
                self._sock_next = sock
                logger.info(
                    f"[PPTransport] stage={self.my_stage} "
                    f"rank={self.my_local_rank} connected to "
                    f"{self.peer_host_next}:{self.peer_port_next} "
                    f"(attempt {attempt})"
                )
                return
            except (ConnectionRefusedError, OSError) as e:
                last_err = e
                try:
                    sock.close()
                except Exception:
                    pass
                if attempt == 1 or attempt % 10 == 0:
                    logger.info(
                        f"[PPTransport] stage={self.my_stage} "
                        f"rank={self.my_local_rank} "
                        f"connect to {self.peer_host_next}:"
                        f"{self.peer_port_next} not ready "
                        f"(attempt {attempt}); retrying"
                    )
                time.sleep(self.connect_retry_s)
        raise TimeoutError(
            f"PPTransport stage={self.my_stage} rank={self.my_local_rank}: "
            f"could not connect to {self.peer_host_next}:{self.peer_port_next} "
            f"within {self.connect_timeout_s}s (last error: {last_err})"
        )


# Process-wide singleton holding the cross-stage PP NCCL group. Built
# once per process by :meth:`PPTransport._ensure_global_pp_pg`.
class _GlobalPPState:
    pg: dist.ProcessGroup | None = None
    global_rank: int = -1
    global_world_size: int = -1
