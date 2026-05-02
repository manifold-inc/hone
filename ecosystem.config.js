const fs = require("fs");
const path = require("path");

const UV = "/home/cvm/.local/bin/uv";
const CWD = "/home/cvm/hone";

// Miner-box ecosystem (8x B200, single-node FSDP).
//
// Topology
// --------
//   GPUs 0-7: 8-way FSDP (dp_shard=8). One torchrun job, one logical
//   miner. No pipeline parallelism.
//
// IMPORTANT: hparams.json must match this topology:
//   - fsdp.dp_shard:       8  (must equal --nproc_per_node here)
//   - batch_size:        256

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

module.exports = {
  apps: [
    {
      name: "miner",
      cwd: CWD,
      script: UV,
      args: [
        "run", "torchrun",
        "--nproc_per_node=8",
        "--master_port=29501",
        "neurons/miner.py",
        "--wallet.name", "miner",
        "--netuid", "5",
      ],
      interpreter: "none",
      env: {
        CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7",
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
    },
  ],
};
