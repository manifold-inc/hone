from __future__ import annotations
import os
import argparse
from dataclasses import dataclass
from common.constants import MINER_PORT
from dotenv import load_dotenv

load_dotenv()

@dataclass
class MinerConfig:
    host: str = os.getenv("MINER_HOST", "0.0.0.0")
    port: int = int(os.getenv("MINER_PORT", MINER_PORT))
    
    wallet_name: str = os.getenv("WALLET_NAME", "default")
    wallet_hotkey: str = os.getenv("WALLET_HOTKEY", "default")
    wallet_path: str = os.getenv("WALLET_PATH", "~/.bittensor/wallets")
    
    hotkey: str = ""
    
    default_response_text: str = os.getenv("DEFAULT_RESPONSE_TEXT", "pong")

    @classmethod
    def from_args(cls, args=None) -> "MinerConfig":
        """Create MinerConfig from command line arguments, with env vars and defaults as fallback."""
        # Ensure .env is loaded before parsing arguments
        load_dotenv()
        
        parser = argparse.ArgumentParser(
            description="Hone Miner - ARC-AGI task solver",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Server configuration
        parser.add_argument(
            "--host",
            type=str,
            default=os.getenv("MINER_HOST", "0.0.0.0"),
            help="Host address to bind the miner server to (env: MINER_HOST)"
        )
        parser.add_argument(
            "--port",
            type=int,
            default=int(os.getenv("MINER_PORT", str(MINER_PORT))),
            help="Port to bind the miner server to (env: MINER_PORT)"
        )
        
        # Wallet configuration
        parser.add_argument(
            "--wallet-name",
            type=str,
            default=os.getenv("WALLET_NAME", "default"),
            help="Name of the wallet to use (env: WALLET_NAME)"
        )
        parser.add_argument(
            "--wallet-hotkey",
            type=str,
            default=os.getenv("WALLET_HOTKEY", "default"),
            help="Hotkey name for the wallet (env: WALLET_HOTKEY)"
        )
        parser.add_argument(
            "--wallet-path",
            type=str,
            default=os.getenv("WALLET_PATH", "~/.bittensor/wallets"),
            help="Path to the wallet directory (env: WALLET_PATH)"
        )
        
        # Response configuration
        parser.add_argument(
            "--default-response-text",
            type=str,
            default=os.getenv("DEFAULT_RESPONSE_TEXT", "pong"),
            help="Default response text for health checks (env: DEFAULT_RESPONSE_TEXT)"
        )
        
        parsed_args = parser.parse_args(args)
        
        # Create config instance with parsed arguments
        return cls(
            host=parsed_args.host,
            port=parsed_args.port,
            wallet_name=parsed_args.wallet_name,
            wallet_hotkey=parsed_args.wallet_hotkey,
            wallet_path=parsed_args.wallet_path,
            default_response_text=parsed_args.default_response_text
        )

    @classmethod  
    def from_env(cls) -> "MinerConfig":
        """Create MinerConfig from environment variables only (backward compatibility)."""
        load_dotenv()
        return cls()



