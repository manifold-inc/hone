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
      env: { CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7" },
    },
  ],
};
