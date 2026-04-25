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

// PM2 does NOT auto-load .env files. The Python ``hone`` package does
// (via python-dotenv on ``import hone``), but anything we explicitly
// set in the ``env:`` block below *overrides* dotenv -- so if we pass
// HONE_EVAL_API_KEY="" because process.env didn't have it, the Python
// side can never recover the real value from .env. Solve that by
// reading hone/.env ourselves at config load time and merging into
// process.env (without overwriting anything already exported by the
// shell/PM2). The parser is intentionally tiny -- the .env we ship is
// plain ``KEY=VALUE`` lines, no quoting / expansion.
function loadDotenv(envPath) {
  if (!fs.existsSync(envPath)) return {};
  const out = {};
  for (const raw of fs.readFileSync(envPath, "utf8").split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    if (eq < 0) continue;
    const key = line.slice(0, eq).trim();
    let val = line.slice(eq + 1).trim();
    // Strip surrounding quotes if any were used.
    if (
      (val.startsWith('"') && val.endsWith('"')) ||
      (val.startsWith("'") && val.endsWith("'"))
    ) {
      val = val.slice(1, -1);
    }
    out[key] = val;
    // Don't clobber an explicit shell export, but DO populate when
    // unset so the rest of this config file can read process.env.X.
    if (process.env[key] === undefined || process.env[key] === "") {
      process.env[key] = val;
    }
  }
  return out;
}
const _dotenvLoaded = loadDotenv(path.join(CWD, ".env"));

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

// The eval ingest API key. Source order: shell env > hone/.env (loaded
// above) > unset. We deliberately do NOT default to "" here because an
// empty string in PM2's ``env:`` block would *override* whatever
// python-dotenv loads from .env on ``import hone`` -- locking the
// Python side out of the value entirely. Leave undefined when missing
// and pass it through conditionally below.
const HONE_EVAL_API_KEY = process.env.HONE_EVAL_API_KEY;

// HF cache: keep on the running user's home so we don't redownload
// datasets each container restart. Operator can override with
// HF_HOME / HF_DATASETS_CACHE in the shell env.
const HOME = os.homedir();
const HF_HOME = process.env.HF_HOME || path.join(HOME, ".cache", "huggingface");
const HF_DATASETS_CACHE =
  process.env.HF_DATASETS_CACHE || path.join(HF_HOME, "datasets");

// Surface the resolved paths so a quick ``pm2 logs`` shows what we picked
// instead of forcing the operator to re-derive it from the trace. Don't
// log the actual key -- just whether we resolved one and where from.
const _evalKeyState = HONE_EVAL_API_KEY
  ? `set (${HONE_EVAL_API_KEY.length} chars)`
  : "MISSING";
console.error(
  `[ecosystem.validator] resolved UV=${UV} CWD=${CWD} ` +
    `HF_HOME=${HF_HOME} HONE_API_BASE_URL=${HONE_API_BASE_URL} ` +
    `HONE_EVAL_API_KEY=${_evalKeyState} ` +
    `(dotenv loaded ${Object.keys(_dotenvLoaded).length} keys)`
);

// Build the env block for the ``eval`` PM2 app. We only include
// HONE_EVAL_API_KEY when it actually has a value, so a missing-here
// case still lets python-dotenv inside the spawned process resolve
// it from .env at runtime. Same dance for HF_TOKEN / WANDB_API_KEY:
// pass them through if set so the eval inherits them, otherwise let
// dotenv handle it.
const evalEnv = {
  CUDA_VISIBLE_DEVICES: "4,5,6,7",
  HF_HOME: HF_HOME,
  HF_DATASETS_CACHE: HF_DATASETS_CACHE,
};
for (const k of ["HONE_EVAL_API_KEY", "HF_TOKEN", "WANDB_API_KEY"]) {
  if (process.env[k]) evalEnv[k] = process.env[k];
}

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
      env: evalEnv,
    },
  ],
};
