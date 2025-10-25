import argparse
from datetime import datetime, timezone
from substrateinterface import SubstrateInterface

U16_MAX = 65535
DEFAULT_ENDPOINT = "wss://entrypoint-finney.opentensor.ai:443"
SS58_FORMAT = 42


def read_weights_by_uid(substrate: SubstrateInterface, netuid: int, uid: int):
    """
    Reads SubtensorModule::Weights(netuid, uid) -> Vec<(u16 dest_uid, u16 weight)>
    """
    q = substrate.query("SubtensorModule", "Weights", [netuid, uid])
    vals = q.value or []
    return [(int(dest), int(w)) for dest, w in vals]


def get_last_update_block(substrate: SubstrateInterface, netuid: int, uid: int) -> int | None:
    """
    Reads SubtensorModule::LastUpdate(netuid) -> Vec<BlockNumber>
    and returns entry for 'uid' if present.
    """
    q = substrate.query("SubtensorModule", "LastUpdate", [netuid], block_hash=None)
    arr = q.value
    if not arr or uid >= len(arr):
        return None
    try:
        return int(arr[uid])
    except Exception:
        try:
            return int(arr[uid].value)
        except Exception:
            return None


def get_block_timestamp(substrate: SubstrateInterface, block_number: int) -> int | None:
    """
    Returns the timestamp (ms since UNIX epoch) at the given block, by:
      - block_hash = get_block_hash(block_number)
      - query Timestamp::Now at that block hash
    """
    if block_number is None:
        return None
    block_hash = substrate.get_block_hash(block_number)
    if block_hash is None:
        return None
    ts_q = substrate.query("Timestamp", "Now", block_hash=block_hash)
    ts = ts_q.value  # milliseconds
    return int(ts) if ts is not None else None


def get_current_block_and_timestamp(substrate: SubstrateInterface) -> tuple[int | None, int | None]:
    """
    Returns (current_block_number, current_timestamp_ms)
    """
    # current block number
    header = substrate.get_block_header()
    try:
        current_block = int(header["header"]["number"])
    except Exception:
        current_block = None

    # current timestamp (no block_hash => latest)
    ts_q = substrate.query("Timestamp", "Now")
    current_ts = int(ts_q.value) if ts_q and ts_q.value is not None else None

    return current_block, current_ts


def fmt_ts(ts_ms: int | None) -> str:
    if ts_ms is None:
        return "unknown"
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    # Show both UTC and local (system) time for convenience
    local_dt = dt.astimezone()  # convert to local tz of the machine running the script
    return f"{dt.strftime('%Y-%m-%d %H:%M:%S %Z')} / {local_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}"


def human_delta(ms_then: int | None, ms_now: int | None) -> str:
    if ms_then is None or ms_now is None:
        return "unknown"
    delta_s = max(0, (ms_now - ms_then) / 1000.0)
    mins, secs = divmod(int(delta_s), 60)
    hours, mins = divmod(mins, 60)
    days, hours = divmod(hours, 24)
    parts = []
    if days: parts.append(f"{days}d")
    if hours: parts.append(f"{hours}h")
    if mins: parts.append(f"{mins}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Read latest Bittensor weights and last set time")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Subtensor websocket endpoint")
    parser.add_argument("--netuid", type=int, default=5, help="Subnet netuid")
    parser.add_argument("--uid", type=int, required=True, help="Validator UID (e.g., 251)")
    parser.add_argument("--top", type=int, default=25, help="Show top N entries")
    args = parser.parse_args()

    print(f"Connecting to: {args.endpoint}")
    substrate = SubstrateInterface(url=args.endpoint, ss58_format=SS58_FORMAT, use_remote_preset=True)

    current_block, current_ts = get_current_block_and_timestamp(substrate)


    print(f"NetUID: {args.netuid}  |  Validator UID: {args.uid}")
    min_allowed = substrate.query("SubtensorModule", "MinAllowedWeights", [args.netuid])
    print(f"Minimum allowed weight %: {min_allowed.value / U16_MAX * 100:.2f}%")
    print(f"min_allowed.value: {min_allowed.value}")

    if current_block is not None:
        print(f"Current block: {current_block}")
    if current_ts is not None:
        print(f"Current time (UTC / local): {fmt_ts(current_ts)}")

    weights = read_weights_by_uid(substrate, args.netuid, args.uid)
    if not weights:
        print("\nNo weights set.")
    else:
        total = sum(w for _, w in weights)
        weights_sorted = sorted(weights, key=lambda x: x[1], reverse=True)

        print(f"\nEntries: {len(weights_sorted)}")
        print(f"Raw sum (uint16 ticks): {total} (target ≈ {U16_MAX})\n")

        top_n = weights_sorted[:args.top]
        print(f"Top {len(top_n)} destinations:")
        print(f"{'UID':>6}  {'Ticks':>8}  {'Share (sum=1)':>14}  {'Percent':>9}")
        print("-" * 44)
        for dest_uid, w in top_n:
            share = (w / total) if total > 0 else 0.0
            print(f"{dest_uid:>6}  {w:>8}  {share:>14.8f}  {share*100:>8.4f}%")

        if len(weights_sorted) > len(top_n):
            tail_sum = sum(w for _, w in weights_sorted[args.top:])
            tail_share = tail_sum / total if total > 0 else 0.0
            print("-" * 44)
            print(f"{'...':>6}  {tail_sum:>8}  {tail_share:>14.8f}  {tail_share*100:>8.4f}%")

        if len(weights_sorted) == 1 and weights_sorted[0][1] == U16_MAX:
            only_uid = weights_sorted[0][0]
            print("\nℹ️  Only one weight at 65535 ticks (all to UID {}).".format(only_uid))

    last_block = get_last_update_block(substrate, args.netuid, args.uid)
    if last_block is None:
        print("\nLast set_weights block: unknown (no entry in LastUpdate).")
        return

    last_ts = get_block_timestamp(substrate, last_block)
    print(f"\nLast set_weights block: {last_block}")
    print(f"Last set_weights time (UTC / local): {fmt_ts(last_ts)}")

    if current_ts is not None and last_ts is not None and current_block is not None:
        print(f"Blocks since last update: {current_block - last_block}")
        print(f"Time since last update:  {human_delta(last_ts, current_ts)}")


if __name__ == "__main__":
    main()
