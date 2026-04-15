# Hone

## Training language models with everyone, not for everyone

The premise behind Hone is simple: training a large language model shouldn't require a data center. It requires compute, yes — lots of it — but that compute doesn't need to sit under one roof. It's scattered across the world in research labs, mining rigs, personal workstations, and cloud instances. Hone is the system that stitches all of it together into one coherent training run.

## How it works

Every participant runs a node. Miners contribute gradient updates; validators evaluate them. There's no central server orchestrating the process — nodes discover each other, exchange compressed gradients, and converge on a shared set of weights through direct peer-to-peer communication.

The trick is making this work honestly. In a system where anyone can join, you need a way to distinguish real contributions from noise (or worse, adversarial updates). Hone handles this through a multi-layered scoring mechanism:

- **Gradient scoring** — Validators measure whether a miner's gradient actually improves the model. They evaluate loss before and after applying each contribution, on both the miner's own data and a random holdout set.
- **Binary moving average** — A running tally of whether contributions are net-positive over time, smoothed to avoid penalizing occasional bad batches.
- **OpenSkill ratings** — An Elo-like system adapted from competitive gaming that ranks miners against each other based on relative contribution quality.
- **Sync scoring** — Measures how up-to-date each miner's model is. Stale gradients computed against an outdated checkpoint are worth less.

These signals combine into a final score that determines each participant's weight — and their share of the rewards.

## The model progression

We started small. The 120M parameter model was a proving ground: small enough to iterate quickly, large enough to validate that the distributed training dynamics actually converge. Once the scoring mechanism stabilized and the gather protocol was battle-tested, we scaled up.

The 1.4B model is where things got interesting. At this scale, the compression and communication overhead starts to matter. We use top-k sparsification with DCT compression to keep gradient payloads small enough for peer-to-peer exchange without losing the information that drives convergence. The gather protocol pulls gradients from ~20 peers per window, with a reserve pool for fallback when nodes drop out.

The 2.6B model pushes further — deeper networks (48 layers vs 24), same hidden dimension, same battle-tested Gemma tokenizer. The training dynamics at this scale surface new challenges around gradient staleness and peer synchronization that feed back into improving the protocol itself.

## What makes this different

Most distributed training systems assume trusted participants and reliable networks. Hone assumes neither. The incentive mechanism means participants are economically motivated to contribute honestly, because the scoring system can detect and penalize low-quality or adversarial updates. Freeloaders see their weights decay; consistent contributors see theirs grow.

The result is a training process that's transparent end-to-end. Every gradient, every score, every weight update is visible on-chain. The dashboard shows the training run in real time — loss curves, peer participation rates, per-UID scoring breakdowns. Anyone can audit the process and verify that the model is being trained fairly.

## Where this goes

The system is live and training. Each version refines the mechanism: tighter scoring, better compression, more robust peer discovery. The goal isn't just to train one model — it's to build infrastructure that makes distributed training a viable alternative to centralized compute monopolies.

The models themselves are open. The training process is open. The only thing we're optimizing for is making the next one better than the last.
