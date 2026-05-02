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
              `[ecosystem.validator] failed to remove ${full}: ${err.message}`
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
const _pycachePurged = cleanStalePyCache(CWD);

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
    `(dotenv loaded ${Object.keys(_dotenvLoaded).length} keys, ` +
    `purged ${_pycachePurged} __pycache__ dir(s); ` +
    `PYTHONDONTWRITEBYTECODE=1 will be set on all ranks)`
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
  // Belt-and-braces with the config-load-time __pycache__ wipe above:
  // every Python process in the pm2 -> uv -> torchrun -> N ranks tree
  // inherits this and never writes a .pyc. See cleanStalePyCache().
  PYTHONDONTWRITEBYTECODE: "1",
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
        // Disable torch.compile on the validator. Its evaluate_model
        // forward path triggers Inductor codegen on first call; if the
        // box's gcc / Triton runtime build fails to link (toolchain
        // mismatch, missing headers, container minimal install, ...)
        // the eval crashes with InductorError and there's no fallback.
        // Validator only runs forward-pass loss eval N times per
        // window per peer -- compile speedup is small, the risk is
        // not worth it. Honoured by both Trainer._apply_torch_compile
        // and Muon's Newton-Schulz JIT (see hone/src/hone/muon/*).
        HONE_DISABLE_TORCH_COMPILE: "1",
        // Belt-and-braces with the config-load-time __pycache__ wipe
        // above: every Python process in the pm2 -> uv -> torchrun ->
        // 4 ranks tree inherits this and never writes a .pyc. See
        // cleanStalePyCache().
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
      //   kill_timeout           -- 30s for workers to drain the
      //                             in-flight gather (up to 600s
      //                             timeout) / R2 PUTs / NCCL handles
      //                             before SIGKILL. Matters because
      //                             SIGTERM mid-gather strands peer
      //                             uploads.
      autorestart: true,
      min_uptime: "60s",
      max_restarts: 5,
      restart_delay: 5000,
      exp_backoff_restart_delay: 100,
      kill_timeout: 30000,
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
        ...evalEnv,
        // Same reasoning as ``vali`` above -- the in-process eval
        // forward also goes through compiled paths and would crash
        // on the same gcc / Triton failure mode.
        HONE_DISABLE_TORCH_COMPILE: "1",
      },
      // Same supervision profile as ``vali`` -- see comment block on
      // that app for details. kill_timeout is 30s here too because
      // the evaluator holds open a POST to hone-api while publishing
      // benchmark scores; cutting it short drops the POST mid-flight.
      autorestart: true,
      min_uptime: "60s",
      max_restarts: 5,
      restart_delay: 5000,
      exp_backoff_restart_delay: 100,
      kill_timeout: 30000,
    },
  ],
};
