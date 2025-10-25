import bittensor as bt
import time
from bittensor_drand.bittensor_drand import get_latest_round

FINNEY_ENDPOINT = "wss://entrypoint-finney.opentensor.ai:443"
NETUID = 5
U16_MAX = 65535

def main():
    cfg = bt.subtensor.config()
    cfg.subtensor.chain_endpoint = FINNEY_ENDPOINT
    cfg.subtensor.network = 'finney'
    subtensor = bt.subtensor(config=cfg)
    wallet = bt.wallet(name="default", hotkey="default")

    pairs = [
        (251, 0.99),
        (69,  0.003),
        (157, 0.003),
        (218, 0.004)
    ]

    # 1) sort by UID ascending
    pairs.sort(key=lambda x: x[0])
    uids = [u for u, _ in pairs]
    weights = [w for _, w in pairs]

    # 2) normalize to sum=1
    s = sum(weights)
    weights = [w / s for w in weights]

    # 3) show expected uint16 ticks after quantization
    ticks = [round(w * U16_MAX) for w in weights]
    # fix rounding drift to sum exactly to U16_MAX
    drift = U16_MAX - sum(ticks)
    if drift != 0:
        # add drift to the largest element
        i_max = max(range(len(ticks)), key=lambda i: ticks[i])
        ticks[i_max] += drift

    print(f"Submitting weights to netuid={NETUID} on {FINNEY_ENDPOINT}")
    print("UIDs (sorted):", uids)
    print("Weights (sum=1):", weights, "sum=", sum(weights))
    print("Expected ticks :", ticks, "sum=", sum(ticks))

    # Commit weights
    ok, info = subtensor.set_weights(
        wallet=wallet,
        netuid=NETUID,
        uids=uids,
        weights=weights,
        version_key=803,
        wait_for_inclusion=False,
        wait_for_finalization=True,
    )

    print(f"Commit Success: {ok}")
    print(f"Commit Info: {info}")

    if not ok:
        print("Failed to commit weights!")
        return

    # Extract reveal round from info
    reveal_round = int(info.split("reveal_round:")[1])
    print(f"\nWeights committed! Reveal round: {reveal_round}")

    # Check current round and wait if needed
    current_round = get_latest_round()
    print(f"Current Drand round: {current_round}")

    if current_round < reveal_round:
        rounds_to_wait = reveal_round - current_round
        # Each round is ~3 seconds on Drand Quicknet
        wait_time = rounds_to_wait * 3
        print(f"Need to wait {rounds_to_wait} rounds (~{wait_time} seconds)")
        print("Waiting for reveal round...")

        while get_latest_round() < reveal_round:
            time.sleep(10)
            current = get_latest_round()
            remaining = reveal_round - current
            print(f"Current round: {current}, remaining: {remaining}")

    print("\nRevealing weights...")
    reveal_ok, reveal_info = subtensor.reveal_weights(
        wallet=wallet,
        netuid=NETUID,
        uids=uids,
        weights=weights,
        salt=[],
        version_key=803,
        wait_for_inclusion=False,
        wait_for_finalization=True,
    )

    print(f"Reveal Success: {reveal_ok}")
    print(f"Reveal Info: {reveal_info}")

if __name__ == "__main__":
    main()