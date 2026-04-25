// PM2 ecosystem for the validator/evaluator box (8x B200).
//
// Layout:
//   GPUs 0-3 -> validator (FSDP dp_shard=4)
//   GPUs 4-7 -> evaluator (4 ranks, in-process loglikelihood eval)
//
// The miner box has its own ecosystem (`ecosystem.config.js`) and uses
// all 8 GPUs for the 3-stage PP miner. This file is intended to live
// on the *second* 8xB200 box dedicated to subnet duties.

const { execSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

// CWD: this config file *is* checked into the hone repo root, so
// __dirname points at the local clone regardless of which user / box
// PM2 is running on. Using ``~/hone`` here would silently break --
// Node/PM2 do NOT tilde-expand cwd strings, so the spawn fails with
// ENOENT and pidusage logs "One of the pids provided is invalid".
const CWD = __dirname;

// UV: try the most reliable lookups in order, fall back to common
// install paths. We need an *existing* file (PM2 spawns it directly,
// so a missing path = failed spawn = the same pidusage error).
function findUv() {
  // 1) Honour an explicit override env var.
  if (process.env.UV_BIN && fs.existsSync(process.env.UV_BIN)) {
    return process.env.UV_BIN;
  }
  // 2) ``which uv`` (POSIX, more reliable than ``whereis``; respects PATH
  //    including the user's ~/.local/bin if the shell exported it).
  try {
    const out = execSync("command -v uv 2>/dev/null", {
      shell: "/bin/bash",
    })
      .toString()
      .trim();
    if (out && fs.existsSync(out)) return out;
  } catch (_) {
    // command -v returns 1 when not found; fall through.
  }
  // 3) ``whereis -b uv`` -- some containers ship this without ``which``.
  try {
    const out = execSync("whereis -b uv").toString().trim();
    const m = out.match(/uv:\s*(\S+)/);
    if (m && fs.existsSync(m[1])) return m[1];
  } catch (_) {
    // whereis missing on minimal images.
  }
  // 4) Probe the common install locations directly.
  const home = os.homedir();
  const candidates = [
    path.join(home, ".local", "bin", "uv"),
    "/root/.local/bin/uv",
    "/usr/local/bin/uv",
    "/usr/bin/uv",
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  // No fallback: PM2 won't spawn a non-existent script. Fail loudly so
  // the operator sees the message instead of an opaque pidusage trace.
  throw new Error(
    "ecosystem.validator.config.js: could not locate the 'uv' binary. " +
      "Install uv (https://docs.astral.sh/uv/) or set UV_BIN=/abs/path/to/uv."
  );
}

const UV = findUv();

// Hone API endpoint the dashboard reads from. Override via env when
// pointing at a non-prod stack.
const HONE_API_BASE_URL =
  process.env.HONE_API_BASE_URL || "https://api.hone.training";

// The eval ingest API key MUST be set in the validator-box environment
// (e.g. /etc/profile.d/hone.sh) -- pm2 will inherit it. This is the
// SAME value as hone-api's API_KEY env var so the evaluator can write.
const HONE_EVAL_API_KEY = process.env.HONE_EVAL_API_KEY || "";

// HF cache: keep on the running user's home so we don't redownload
// datasets each container restart. Operator can override with
// HF_HOME / HF_DATASETS_CACHE in the shell env.
const HOME = os.homedir();
const HF_HOME = process.env.HF_HOME || path.join(HOME, ".cache", "huggingface");
const HF_DATASETS_CACHE =
  process.env.HF_DATASETS_CACHE || path.join(HF_HOME, "datasets");

// Surface the resolved paths so a quick ``pm2 logs`` shows what we picked
// instead of forcing the operator to re-derive it from the trace.
console.error(
  `[ecosystem.validator] resolved UV=${UV} CWD=${CWD} ` +
    `HF_HOME=${HF_HOME} HONE_API_BASE_URL=${HONE_API_BASE_URL}`
);

module.exports = {
  apps: [
    {
      name: "vali",
      cwd: CWD,
      script: UV,
      args: [
        "run", "torchrun",
        "--nproc_per_node=4",
        "--master_port=29501",
        "neurons/validator.py",
        "--wallet.name", "vali",
        "--netuid", "5",
      ],
      interpreter: "none",
      env: {
        CUDA_VISIBLE_DEVICES: "0,1,2,3",
      },
    },
    {
      name: "eval",
      cwd: CWD,
      script: UV,
      args: [
        "run", "torchrun",
        "--nproc_per_node=4",
        "--master_port=29502",
        "neurons/evaluator.py",
        "--wallet.name", "vali",
        "--netuid", "5",
        "--eval-interval", "300",
        "--tasks",
        "arc_challenge,arc_easy,hellaswag,winogrande,piqa,openbookqa",
        "--api-base-url", HONE_API_BASE_URL,
      ],
      interpreter: "none",
      env: {
        CUDA_VISIBLE_DEVICES: "4,5,6,7",
        HONE_EVAL_API_KEY: HONE_EVAL_API_KEY,
        // Force HF datasets cache off /tmp so we don't redownload
        // each restart; on-box NVMe is plenty. Resolved at config
        // load time from os.homedir() (or HF_HOME / HF_DATASETS_CACHE
        // overrides if set in the shell env).
        HF_HOME: HF_HOME,
        HF_DATASETS_CACHE: HF_DATASETS_CACHE,
      },
    },
  ],
};
