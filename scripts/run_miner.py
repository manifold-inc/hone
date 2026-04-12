#!/usr/bin/env python3
"""Run a Hone miner on Bittensor subnet 5."""

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(description="Hone Miner")
    parser.add_argument("--netuid", type=int, default=5)
    parser.add_argument("--wallet.name", dest="wallet_name", default="default")
    parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", default="default")
    parser.add_argument("--wallet.path", dest="wallet_path", default="~/.bittensor/wallets")
    parser.add_argument("--subtensor.network", dest="subtensor_network", default="finney")
    parser.add_argument("--subtensor.address", dest="subtensor_address", default=None)
    parser.add_argument("--model_type", default=None)
    parser.add_argument("--inner_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--dataset_path", dest="dataset_bins_path", default=None)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    overrides = {
        k: v for k, v in vars(args).items()
        if v is not None and k != "log_level"
    }

    from hone.config import load_config
    from hone.miner import run_miner

    config = load_config(**overrides)
    run_miner(config)


if __name__ == "__main__":
    main()
