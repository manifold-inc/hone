const UV = "/home/cvm/.local/bin/uv";
const CWD = "/home/cvm/hone";

// Miner-box ecosystem (8x B200). The validator + evaluator now live on
// the second box (see ecosystem.validator.config.js) so the miner gets
// all 8 GPUs.
//
// Topology
// --------
//   stage 0: GPUs 0-3 (4-way FSDP, dp_shard=4)
//   stage 1: GPUs 4-7 (4-way FSDP, dp_shard=4)
//   PP=2 boundaries (1 cross-stage hop), each carries half the model.
//
// PP=2 + dp=4 vs the previous PP=3 + dp=2:
//   - One fewer PP boundary -> ~33% less cross-stage activation traffic.
//   - dp=4 lets FSDP shard each layer 4 ways instead of 2 (smaller
//     per-rank all-gather payload, better overlap with compute).
//   - Bubble fraction P/(P+M-1) at micro_bs=4, batch_size=256:
//       M = 256 / (4 * 4) = 16
//       bubble = 2 / (2+16-1) = ~12%
//     vs the prior PP=3 + dp=2 + batch=128 setup which sat at ~17%.
//
// IMPORTANT: hparams.json must match this topology:
//   - pipeline.num_stages: 2  (validator's per-peer gather walks this many
//                              stage files; mismatch silently drops grads)
//   - fsdp.dp_shard:       4  (must equal --nproc_per_node here)
//   - batch_size:        256  (drives M=16; keep micro_batch_size=4)
//   - inner_steps:         4  (was 8 at batch=128; halved to keep
//                              per-window wall-clock under
//                              window_flush_headroom_seconds)
//
// Listen ports: stage S listens on PP_BASE + 100*S per rank. With PP=2
// we have a single boundary, but we keep the +100 offset convention so
// adding/removing stages later is trivial.
const PP_S0_NEXT = 50100;   // s0 -> s1
const PP_S1_LISTEN = 50100; // s1 listens

module.exports = {
  apps: [
    {
      name: "s0",
      cwd: CWD,
      script: UV,
      args: [
        "run", "torchrun",
        "--nproc_per_node=4",
        "--master_port=29501",
        "neurons/miner.py",
        "--wallet.name", "miner",
        "--netuid", "5",
        "--pp-stage", "0",
        "--pp-num-stages", "2",
        "--pp-peer-host-next", "127.0.0.1",
        "--pp-peer-port-base-next", String(PP_S0_NEXT),
        // s0 has no prev so its own listen port doesn't matter, but
        // PPTransport reads the flag at init -- give it a unique
        // unused port so a future hostname change doesn't collide.
        "--pp-listen-port-base", String(50000),
      ],
      interpreter: "none",
      env: { CUDA_VISIBLE_DEVICES: "0,1,2,3" },
    },
    {
      name: "s1",
      cwd: CWD,
      script: UV,
      args: [
        "run", "torchrun",
        "--nproc_per_node=4",
        "--master_port=29502",
        "neurons/miner.py",
        "--wallet.name", "miner",
        "--netuid", "5",
        "--pp-stage", "1",
        "--pp-num-stages", "2",
        "--pp-peer-host-prev", "127.0.0.1",
        "--pp-peer-port-base-prev", String(PP_S0_NEXT),
        "--pp-listen-port-base", String(PP_S1_LISTEN),
      ],
      interpreter: "none",
      env: { CUDA_VISIBLE_DEVICES: "4,5,6,7" },
    },
  ],
};
