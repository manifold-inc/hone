import asyncio
from loguru import logger
import signal

from common.chain import ChainInterface
from validator.config import ValidatorConfig
from validator.db import Database
from validator import cycle
from validator.synthetics.arcgen.arc_agi2_generator import ARC2Generator
import random

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

        self.synthetic_generator = ARC2Generator(max_chain_length=2)
    
    async def start(self):
        await self.db.connect()
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig, 
                lambda s=sig: asyncio.create_task(self.shutdown(s))
            )
        
        logger.info("Validator started")
    
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

    async def run(self):
        try:
            await cycle.run_continuous(self, self._stop_event)
        except asyncio.CancelledError:
            logger.info("Validator run cancelled")
        finally:
            await self.db.close()
            self.chain.substrate.close() if self.chain.substrate else None