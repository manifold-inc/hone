const fs = require("fs");
const path = require("path");

const UV = "/home/cvm/.local/bin/uv";
const CWD = "/home/cvm/hone";

// Miner-box ecosystem (8x B200, multi-hotkey).
//
// Topology
// --------
//   4 miner processes, each using 2 GPUs via torchrun.
//   Total: 4 miners x 2 GPUs = 8 GPUs. Each miner registers a
//   distinct UID on-chain via its own hotkey, so the one physical
//   box contributes 4 independent peers to the Hone subnet.
//
//     miner-default : GPUs 0,1  hotkey=default  master_port=29501
//     miner-1       : GPUs 2,3  hotkey=1        master_port=29502
//     miner-2       : GPUs 4,5  hotkey=2        master_port=29503
//     miner-3       : GPUs 6,7  hotkey=3        master_port=29504
//
// All share ``--wallet.name miner`` (single wallet, multiple hotkeys).
//
// IMPORTANT: hparams.json must match this per-miner topology:
//   - fsdp.dp_shard:        2  (must equal --nproc_per_node here)
//   - batch_size:          64  (was 256; scaled 4x down for 4x fewer
//                               GPUs per miner — same #grad-accum
//                               steps as before)
//   - target_batch_size:  256  (was 1024; scaled 4x down so per-miner
//                               effective batch stays at the same
//                               ratio vs batch_size)
//   - micro_batch_size:     8  (unchanged — per-GPU VRAM bound)
//
// Per-miner throughput is ~1/4 of the pre-split single-miner config,
// but total box throughput is identical (4 miners x 1/4). The real
// win is 4 independent UIDs on one box instead of 1.

// One-shot wipe of Hone-owned __pycache__ trees. Runs at config load
// time (i.e. on ``pm2 start`` / ``pm2 reload`` of this ecosystem), NOT
// on every auto-restart. That's the right granularity: once
// PYTHONDONTWRITEBYTECODE=1 is in effect below, the ranks can never
// produce a new corrupt .pyc, so we only need to clear leftovers from
// a previous crash loop.
//
// Why this matters: torchrun spawns N simultaneous Python ranks that
// each compile ``hone/src/hone/*.py`` -> ``.pyc`` on first import. On
// shared / overlay filesystems that don't fully honour POSIX
// ``rename(2)`` atomicity, a sibling rank can read a partially-written
// .pyc and die with ``EOFError: marshal data too short`` -- aborting
// the whole distributed run. pm2 restarts, the corrupt .pyc is still
// on disk, the loop continues. See README "SIGTERM troubleshooting".
function cleanStalePyCache(cwd) {
  if (!fs.existsSync(cwd)) return 0;
  let removed = 0;
  for (const relRoot of ["src/hone", "neurons"]) {
    const absRoot = path.join(cwd, relRoot);
    if (!fs.existsSync(absRoot)) continue;
    const stack = [absRoot];
    while (stack.length > 0) {
      const dir = stack.pop();
      let entries;
      try {
        entries = fs.readdirSync(dir, { withFileTypes: true });
      } catch (_) {
        continue;
      }
      for (const ent of entries) {
        if (!ent.isDirectory()) continue;
        const full = path.join(dir, ent.name);
        if (ent.name === "__pycache__") {
          try {
            fs.rmSync(full, { recursive: true, force: true });
            removed++;
          } catch (err) {
            console.error(
              `[ecosystem.miner] failed to remove ${full}: ${err.message}`
            );
          }
        } else {
          stack.push(full);
        }
      }
    }
  }
  return removed;
}

const _purged = cleanStalePyCache(CWD);
console.error(
  `[ecosystem.miner] CWD=${CWD} purged ${_purged} __pycache__ dir(s); ` +
    `PYTHONDONTWRITEBYTECODE=1 will be set on all ranks`
);

// Per-miner spec. Each entry produces one pm2 app with its own
// torchrun rendezvous port, GPU affinity, and bittensor hotkey. All
// four share ``--wallet.name miner`` — single wallet, four hotkeys.
const MINERS = [
  { hotkey: "default", cudaDevices: "0,1,2,3", masterPort: 29501 },
  { hotkey: "1",       cudaDevices: "4,5,6,7", masterPort: 29502 },
  // { hotkey: "2",       cudaDevices: "4,5", masterPort: 29503 },
  // { hotkey: "3",       cudaDevices: "6,7", masterPort: 29504 },
];

module.exports = {
  apps: MINERS.map(({ hotkey, cudaDevices, masterPort }) => ({
    name: `miner-${hotkey}`,
    cwd: CWD,
    script: UV,
    args: [
      "run", "torchrun",
      "--nproc_per_node=2",
      `--master_port=${masterPort}`,
      "neurons/miner.py",
      "--wallet.name", "miner",
      "--wallet.hotkey", hotkey,
      "--netuid", "5",
    ],
    interpreter: "none",
    env: {
      CUDA_VISIBLE_DEVICES: cudaDevices,
      // Prevents .pyc writes on every Python process in the tree
      // (pm2 -> uv -> torchrun -> N ranks). See cleanStalePyCache()
      // above for why.
      PYTHONDONTWRITEBYTECODE: "1",
    },
    // Supervision: keep pm2 from masking crash loops.
    //   min_uptime             -- process is "stable" only after
    //                             60s; faster exits count against
    //                             max_restarts.
    //   max_restarts           -- stop after 5 consecutive unstable
    //                             exits; operator must intervene.
    //   restart_delay          -- 5s between restarts (paired with
    //                             exp_backoff_restart_delay below
    //                             so repeated failures back off).
    //   exp_backoff_restart_delay -- 100ms base; doubles each
    //                             consecutive failure.
    //   kill_timeout           -- 30s to drain in-flight R2 PUTs /
    //                             NCCL handles before SIGKILL.
    autorestart: true,
    min_uptime: "60s",
    max_restarts: 5,
    restart_delay: 5000,
    exp_backoff_restart_delay: 100,
    kill_timeout: 30000,
  })),
};
