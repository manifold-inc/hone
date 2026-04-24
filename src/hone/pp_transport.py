"""TCP transport for cross-stage activation transfer in pipeline parallelism.

Each pipeline stage is launched as its own ``torchrun`` job with its own
NCCL world. Cross-stage forward activations and backward gradients flow
over a separate TCP transport, decoupled from the per-stage NCCL collectives
(FSDP all_gather / reduce_scatter) that would otherwise serialize against
PP P2P on the same NCCL stream.

Topology
--------
For every local rank ``R`` in stage ``S``, exactly one TCP socket is
maintained with the matching local rank ``R`` in stage ``S+1``. The socket
is bidirectional: forward activations flow stage S -> stage S+1, backward
gradients flow back the other way.

Connection roles: the *lower* stage initiates the connection (client) and
the *higher* stage accepts (server). Listening starts before connecting
in start() to avoid the obvious race.

Wire format
-----------
Each tensor is sent as ``[8B big-endian length][raw payload]`` where the
payload is produced by ``torch.save(...)``. ``torch.save`` / ``torch.load``
handle every dtype (including ``bfloat16``) without numpy fallbacks, and
sidestep version-specific quirks in ``Tensor.view(other_dtype)`` that
otherwise mis-reinterpret bf16 byte buffers. The overhead of ``torch.save``
on a contiguous CPU tensor of <1 MB is sub-millisecond and dominated by
the network transfer itself.
"""

from __future__ import annotations

import io
import socket
import struct
import threading
import time

import torch

from .logging import logger

_LEN_PREFIX = "!Q"  # 8-byte big-endian unsigned int
_LEN_SIZE = struct.calcsize(_LEN_PREFIX)
_RECV_CHUNK = 1 << 20  # 1 MiB read chunk


def _send_all(sock: socket.socket, data: bytes) -> None:
    """sendall over a connected TCP socket; raises on closed peer."""
    sock.sendall(data)


def _recv_exact(sock: socket.socket, n: int) -> bytearray:
    """Read exactly ``n`` bytes from ``sock``; raises on early EOF.

    Uses ``recv_into`` against a pre-allocated bytearray so we avoid the
    per-chunk concatenation overhead that ``b"".join(chunks)`` incurs for
    multi-MiB payloads.
    """
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        read = sock.recv_into(view[got:], min(n - got, _RECV_CHUNK))
        if read == 0:
            raise ConnectionError(
                f"socket closed with {n - got} of {n} bytes remaining"
            )
        got += read
    return buf


def _send_bytes(sock: socket.socket, payload: bytes) -> None:
    _send_all(sock, struct.pack(_LEN_PREFIX, len(payload)))
    _send_all(sock, payload)


def _recv_bytes(sock: socket.socket) -> bytearray:
    header = _recv_exact(sock, _LEN_SIZE)
    (length,) = struct.unpack(_LEN_PREFIX, bytes(header))
    return _recv_exact(sock, length)


class PPTransport:
    """Per-rank TCP transport for one cross-stage P2P pair.

    Each rank opens at most two sockets:
    - ``sock_prev`` (server, accepts): connection with the previous stage's
      matching local rank. Set only when ``my_stage > 0``.
    - ``sock_next`` (client, connects): connection with the next stage's
      matching local rank. Set only when ``my_stage < num_stages - 1``.

    ``start()`` blocks until both expected sockets are connected. Subsequent
    ``send_next/recv_next/send_prev/recv_prev`` are blocking calls on the
    Python side and do not interact with the CUDA stream (the transport
    operates on CPU bytes).
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

        self._sock_prev: socket.socket | None = None
        self._sock_next: socket.socket | None = None
        self._listener: socket.socket | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Bring up both sockets. Blocks until both peers are connected.

        Server-side accept runs on a background thread so we can also issue
        the outbound ``connect`` from the same start() invocation, then
        join. This way the start() call returns only when the rank is fully
        wired into the pipeline.
        """
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

        logger.info(
            f"[PPTransport] stage={self.my_stage}/{self.num_stages} "
            f"rank={self.my_local_rank} ready "
            f"(prev={'set' if self._sock_prev else 'none'}, "
            f"next={'set' if self._sock_next else 'none'})"
        )

    def stop(self) -> None:
        for s in (self._sock_prev, self._sock_next, self._listener):
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass
        self._sock_prev = None
        self._sock_next = None
        self._listener = None

    # ------------------------------------------------------------------
    # Connection helpers
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

    # ------------------------------------------------------------------
    # Tensor send / receive
    # ------------------------------------------------------------------
    @staticmethod
    def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
        # ``torch.save`` is dtype-agnostic (handles bf16 cleanly) and the
        # serialization overhead on a contiguous CPU tensor of <1 MB is
        # negligible compared to the network round-trip we're about to do.
        cpu = tensor.detach().to("cpu").contiguous()
        buf = io.BytesIO()
        torch.save(cpu, buf)
        return buf.getvalue()

    def _bytes_to_tensor(
        self,
        buf: bytearray,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # We pass ``shape`` / ``dtype`` only as a sanity check; ``torch.load``
        # rebuilds the original tensor metadata directly from the payload.
        t = torch.load(io.BytesIO(bytes(buf)), weights_only=True)
        if tuple(t.shape) != tuple(shape):
            raise RuntimeError(
                f"PPTransport: received tensor shape {tuple(t.shape)} "
                f"but expected {tuple(shape)}"
            )
        if t.dtype != dtype:
            raise RuntimeError(
                f"PPTransport: received tensor dtype {t.dtype} "
                f"but expected {dtype}"
            )
        return t.to(self.device)

    def _send_tensor(
        self, sock: socket.socket | None, tensor: torch.Tensor
    ) -> None:
        if sock is None:
            raise RuntimeError("PPTransport: socket not connected")
        _send_bytes(sock, self._tensor_to_bytes(tensor))

    def _recv_tensor(
        self,
        sock: socket.socket | None,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if sock is None:
            raise RuntimeError("PPTransport: socket not connected")
        return self._bytes_to_tensor(_recv_bytes(sock), shape, dtype)

    # Public API used by trainer._pp_run_1f1b.
    def send_next(self, tensor: torch.Tensor) -> None:
        self._send_tensor(self._sock_next, tensor)

    def recv_next(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        return self._recv_tensor(self._sock_next, shape, dtype)

    def send_prev(self, tensor: torch.Tensor) -> None:
        self._send_tensor(self._sock_prev, tensor)

    def recv_prev(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        return self._recv_tensor(self._sock_prev, shape, dtype)
