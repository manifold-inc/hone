const UV = "/home/cvm/.local/bin/uv";
const CWD = "/home/cvm/hone";

// All four PM2 apps run on the same host, so each stage needs its own
// listening port range -- otherwise stage1 + stage2 both try to bind
// 0.0.0.0:50000 (each rank R adds R to ``pp_listen_port_base``) and the
// second one fails with "Address already in use".
//
// Layout: stage S listens on PP_BASE + 100*S (per rank); the previous
// stage's "peer-next" port matches.
//   stage 0: nothing to listen for (no prev). Connects to s1 @ 50100.
//   stage 1: listens 50100/50101 (prev=s0). Connects to s2 @ 50200.
//   stage 2: listens 50200/50201 (prev=s1). No next.
const PP_S0_NEXT = 50100; // s0 -> s1
const PP_S1_LISTEN = 50100; // s1 listens
const PP_S1_NEXT = 50200; // s1 -> s2
const PP_S2_LISTEN = 50200; // s2 listens

module.exports = {
  apps: [
    {
      name: "vali",
      cwd: CWD,
      script: UV,
      args: [
        "run", "torchrun",
        "--nproc_per_node=2",
        "neurons/validator.py",
        "--wallet.name", "vali",
        "--netuid", "5",
      ],
      interpreter: "none",
      // Validator gets physical GPUs 0,1.
      env: { CUDA_VISIBLE_DEVICES: "0,1" },
    },
    {
      name: "s0",
      cwd: CWD,
      script: UV,
      args: [
        "run", "torchrun",
        "--nproc_per_node=2",
        "--master_port=29501",
        "neurons/miner.py",
        "--wallet.name", "miner",
        "--netuid", "5",
        "--pp-stage", "0",
        "--pp-num-stages", "3",
        "--pp-peer-host-next", "127.0.0.1",
        "--pp-peer-port-base-next", String(PP_S0_NEXT),
        // s0 has no prev so its own listen port doesn't matter, but
        // PPTransport reads the flag at init -- give it a unique
        // unused port so a future hostname change doesn't collide.
        "--pp-listen-port-base", String(50000),
      ],
      interpreter: "none",
      env: { CUDA_VISIBLE_DEVICES: "2,3" },
    },
    {
      name: "s1",
      cwd: CWD,
      script: UV,
      args: [
        "run", "torchrun",
        "--nproc_per_node=2",
        "--master_port=29502",
        "neurons/miner.py",
        "--wallet.name", "miner",
        "--netuid", "5",
        "--pp-stage", "1",
        "--pp-num-stages", "3",
        "--pp-peer-host-prev", "127.0.0.1",
        "--pp-peer-port-base-prev", String(PP_S0_NEXT),
        "--pp-peer-host-next", "127.0.0.1",
        "--pp-peer-port-base-next", String(PP_S1_NEXT),
        "--pp-listen-port-base", String(PP_S1_LISTEN),
      ],
      interpreter: "none",
      env: { CUDA_VISIBLE_DEVICES: "4,5" },
    },
    {
      name: "s2",
      cwd: CWD,
      script: UV,
      args: [
        "run", "torchrun",
        "--nproc_per_node=2",
        "--master_port=29503",
        "neurons/miner.py",
        "--wallet.name", "miner",
        "--netuid", "5",
        "--pp-stage", "2",
        "--pp-num-stages", "3",
        "--pp-peer-host-prev", "127.0.0.1",
        "--pp-peer-port-base-prev", String(PP_S1_NEXT),
        "--pp-listen-port-base", String(PP_S2_LISTEN),
      ],
      interpreter: "none",
      env: { CUDA_VISIBLE_DEVICES: "6,7" },
    },
  ],
};
