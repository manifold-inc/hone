import asyncio
from loguru import logger
import signal
import time
import os

from common.chain import ChainInterface
from validator.config import ValidatorConfig
from validator.db import Database
from validator import cycle
from validator.telemetry import TelemetryClient

class Validator:
    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.should_stop = False
        self._stop_event = asyncio.Event()

        if config.use_mock_chain:
            from common.mock_chain import MockChainInterface
            self.chain = MockChainInterface(
                endpoint="mock://localhost",
                netuid=config.netuid,
                wallet_name="validator",
                num_mock_miners=3
            )
            logger.info("Using MockChainInterface for testing")
        else:
            self.chain = ChainInterface(
                endpoint=config.chain_endpoint, 
                netuid=config.netuid,
                wallet_name=config.wallet_name,
                wallet_hotkey=config.wallet_hotkey,
                wallet_path=config.wallet_path
            )

        self.db = Database(dsn=config.db_url)
        self.config.current_block_provider = self.get_current_block
        
        self.state = {
            'cycle_count': 0,
            'last_query_block': None,
            'last_weights_block': None
        }

        # REMOVED: self.synthetic_generator = ARC2Generator(max_chain_length=6)
        # synthetic generation now handled by sandbox runner
        
        self.telemetry_client = TelemetryClient(
            endpoint_base_url=os.getenv("TELEMETRY_ENDPOINT", ""),
            max_queue_size=1000,
            request_timeout_s=5.0,
        )

        self.last_cleanup_time = None
        
        logger.info(f"Validator initialized")
        logger.info(f"  Hotkey: {config.hotkey}")
        logger.info(f"  Sandbox endpoint: {config.sandbox_runner_endpoint}")
        logger.info(f"  Sandbox timeout: {config.sandbox_runner_timeout_hours}h")
    
    async def start(self):
        await self.db.connect()
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig, 
                lambda s=sig: asyncio.create_task(self.shutdown(s))
            )
        
        logger.info("Validator started successfully")
    
    async def shutdown(self, sig):
        logger.info(f"Received exit signal {sig.name}...")
        self.should_stop = True
        self._stop_event.set()
        
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        
        logger.info("Shutdown complete")

    def stop(self):
        self.should_stop = True
        self._stop_event.set()
        logger.info("Stopping validator...")

    def get_current_block(self) -> int:
        try:
            self.chain.connect()
            return self.chain.get_current_block()
        except Exception as e:
            logger.warning(f"Could not read current block from chain ({e}); falling back to 0")
            return 0
    
    async def maybe_cleanup_database(self):
        """Periodically cleanup old database records"""        
        current_time = time.time()
        cleanup_interval_seconds = self.config.cleanup_interval_hours * 3600
        
        if (self.last_cleanup_time is None or 
            current_time - self.last_cleanup_time > cleanup_interval_seconds):
            
            logger.info(f"Running database cleanup (retention: {self.config.retention_days} days)")
            await self.db.cleanup_old_data(self.config.retention_days)
            self.last_cleanup_time = current_time

    async def run(self):
        try:
            await cycle.run_continuous(self, self._stop_event)
        except asyncio.CancelledError:
            logger.info("Validator run cancelled")
        finally:
            await self.db.close()
            self.chain.substrate.close() if self.chain.substrate else None