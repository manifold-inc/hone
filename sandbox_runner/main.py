"""
1. Loads configuration from YAML file and CLI arguments
2. Initializes the FastAPI application
3. Sets up logging and monitoring
4. Starts the HTTPS server with uvicorn
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import uvicorn

from api.gateway import create_app
from config import load_config, Config
from monitoring.logging import setup_logging


def parse_args():
    """Parse command-line arguments with support for config overrides."""
    parser = argparse.ArgumentParser(
        description="Hone Subnet Sandbox Runner - Secure GPU execution service"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration YAML file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        help="Override API port from config"
    )
    
    parser.add_argument(
        "--gpu-count",
        type=int,
        help="Override GPU count from config"
    )
    
    parser.add_argument(
        "--execution-mode",
        choices=["docker+gvisor", "docker", "direct"],
        help="Override execution mode from config"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--ssl-cert",
        type=Path,
        help="Path to SSL certificate file"
    )
    
    parser.add_argument(
        "--ssl-key",
        type=Path,
        help="Path to SSL private key file"
    )
    
    return parser.parse_args()


def apply_cli_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Apply command-line argument overrides to configuration."""
    if args.port:
        config.api.port = args.port
    
    if args.gpu_count:
        config.hardware.gpu_count = args.gpu_count
    
    if args.execution_mode:
        config.execution.mode = args.execution_mode
    
    if args.ssl_cert:
        config.api.ssl_cert_path = args.ssl_cert
    
    if args.ssl_key:
        config.api.ssl_key_path = args.ssl_key
    
    return config


async def startup_checks(config: Config, logger: logging.Logger):
    """Perform startup validation and health checks."""
    logger.info(f"Starting Hone Subnet Sandbox Runner (ID: {config.runner.id})")
    logger.info(f"Execution mode: {config.execution.mode}")
    logger.info(f"GPU count: {config.hardware.gpu_count}")
    logger.info(f"API port: {config.api.port}")
    
    # SSL certificates exist
    if config.api.ssl_cert_path and not config.api.ssl_cert_path.exists():
        logger.error(f"SSL certificate not found: {config.api.ssl_cert_path}")
        sys.exit(1)
    
    if config.api.ssl_key_path and not config.api.ssl_key_path.exists():
        logger.error(f"SSL key not found: {config.api.ssl_key_path}")
        sys.exit(1)
    
    logger.info("Startup validation completed successfully")


def main():
    """Main application entry point."""
    args = parse_args()    
    logger = setup_logging(args.log_level)
    
    try:
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        config = apply_cli_overrides(config, args)
        
        asyncio.run(startup_checks(config, logger))
        
        app = create_app(config)
        
        ssl_keyfile = str(config.api.ssl_key_path) if config.api.ssl_key_path else None
        ssl_certfile = str(config.api.ssl_cert_path) if config.api.ssl_cert_path else None
        
        logger.info(f"Starting HTTPS server on port {config.api.port}")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=config.api.port,
            #ssl_keyfile=ssl_keyfile,
            #ssl_certfile=ssl_certfile,
            log_level=args.log_level.lower(),
            access_log=True,
        )
        
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Fatal error during startup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()