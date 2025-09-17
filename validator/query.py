import asyncio
import aiohttp
from typing import Dict, Any
from datetime import datetime, timezone
from loguru import logger
import json

from common.epistula import Epistula
from common.constants import DEFAULT_QUERY_BODY, QUERY_ENDPOINT, DEFAULT_TIMEOUT

async def _query_one(session: aiohttp.ClientSession, chain, config, uid: int, miner: Dict) -> Dict:
    ip = miner.get("ip")
    port = miner.get("port") or config.default_miner_port
    url = f"http://{ip}:{port}{QUERY_ENDPOINT}"
    
    logger.info(f"Querying UID {uid} at {ip}:{port}")
    
    body, headers = Epistula.create_request(
        keypair=chain.keypair,
        receiver_hotkey=miner.get("hotkey"),
        data=DEFAULT_QUERY_BODY,
        version=1
    )

    t0 = datetime.now(timezone.utc)
    try:
        async with session.post(url, json=body, headers=headers, timeout=DEFAULT_TIMEOUT) as resp:
            ok = resp.status == 200
            response_text = await resp.text()
            
            if ok:
                try:
                    response_json = json.loads(response_text)
                    payload = response_json.get('data', {})
                except json.JSONDecodeError:
                    payload = None
                    ok = False
            else:
                payload = None
            
            dt = (datetime.now(timezone.utc) - t0).total_seconds()
            
            if ok:
                logger.info(f"Success for UID {uid}")
            else:
                logger.error(f"Failed for UID {uid}: HTTP {resp.status}")
                
            return {
                "uid": uid, 
                "success": ok, 
                "response": payload, 
                "error": None if ok else f"HTTP {resp.status}", 
                "rt": dt
            }
            
    except asyncio.TimeoutError:
        dt = (datetime.now(timezone.utc) - t0).total_seconds()
        logger.error(f"Timeout for UID {uid}")
        return {"uid": uid, "success": False, "response": None, "error": "Timeout", "rt": dt}
    except Exception as e:
        dt = (datetime.now(timezone.utc) - t0).total_seconds()
        logger.error(f"Failed for UID {uid}: {e}")
        return {"uid": uid, "success": False, "response": None, "error": str(e), "rt": dt}

async def query_miners(chain, db, config, miners: Dict[int, Dict], current_block: int) -> Dict[int, Dict]:
    results: Dict[int, Dict] = {}
    timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [_query_one(session, chain, config, uid, miner) for uid, miner in miners.items()]
        
        for fut in asyncio.as_completed(tasks):
            res = await fut
            uid = res["uid"]
            results[uid] = res
            
            await db.record_query_result(
                block=current_block,
                uid=uid,
                success=res["success"],
                response=res["response"],
                error=res["error"],
                response_time=res["rt"],
                ts=datetime.utcnow(),
            )
            
    successful = sum(1 for r in results.values() if r["success"])
    logger.info(f"Queried {len(results)} miners (Epistula) - {successful} successful")
    return results