// PM2 ecosystem for the validator/evaluator box (8x B200).
//
// Layout:
//   GPUs 0-3 -> validator (FSDP dp_shard=4)
//   GPUs 4-7 -> evaluator (4 ranks, in-process loglikelihood eval)
//
// The miner box has its own ecosystem (`ecosystem.config.js`) and uses
// all 8 GPUs for the 3-stage PP miner. This file is intended to live
// on the *second* 8xB200 box dedicated to subnet duties.

const { execSync } = require('child_process');

function findUv() {
  try {
    // 'whereis -b uv' returns lines like: "uv: /root/.local/bin/uv"
    const output = execSync('whereis -b uv').toString().trim();
    const match = output.match(/uv:\s*(\S+)/);
    if (match) {
      return match[1];
    } else {
      throw new Error('uv binary not found in whereis output');
    }
  } catch (e) {
    // Fallback to common install location if not found
    return '/usr/local/bin/uv';
  }
}

const UV = findUv();
const CWD = "~/hone";

// Hone API endpoint the dashboard reads from. Override via env when
// pointing at a non-prod stack.
const HONE_API_BASE_URL =
  process.env.HONE_API_BASE_URL || "https://api.hone.training";

// The eval ingest API key MUST be set in the validator-box environment
// (e.g. /etc/profile.d/hone.sh) -- pm2 will inherit it. This is the
// SAME value as hone-api's API_KEY env var so the evaluator can write.
const HONE_EVAL_API_KEY = process.env.HONE_EVAL_API_KEY || "";

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
        // each restart; on-box NVMe is plenty.
        HF_DATASETS_CACHE: "/home/cvm/.cache/huggingface/datasets",
        HF_HOME: "/home/cvm/.cache/huggingface",
      },
    },
  ],
};
