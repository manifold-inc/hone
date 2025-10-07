import asyncio
from loguru import logger

from validator.config import ValidatorConfig
from validator.validator import Validator

async def main():
    config = ValidatorConfig()
    validator = Validator(config)
    
    try:
        await validator.start()
        await validator.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        validator.stop()
        await asyncio.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(main())