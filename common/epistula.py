import json
import time
from typing import Dict, Any, Tuple, Optional
from substrateinterface import Keypair
from loguru import logger


class Epistula:
    """
    Epistula protocol for secure miner-validator communication
    Based on https://github.com/manifold-inc/epistula-examples
    """
    
    ALLOWED_DELTA_NS = 5_000_000_000
    
    @staticmethod
    def create_request(
        keypair: Keypair,
        receiver_hotkey: str,
        data: Dict[str, Any],
        version: int = 1
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Create a signed Epistula request
        
        Args:
            keypair: Sender's keypair
            receiver_hotkey: Receiver's SS58 address
            data: Request data
            version: Protocol version
            
        Returns:
            Tuple of (body, headers)
        """
        body = {
            "data": data,
            "nonce": time.time_ns(),
            "signed_by": keypair.ss58_address,
            "signed_for": receiver_hotkey,
            "version": version
        }
        
        body_bytes = json.dumps(body, sort_keys=True)
        signature = keypair.sign(body_bytes)
        
        headers = {
            "Body-Signature": "0x" + signature.hex(),
            "Content-Type": "application/json"
        }
        
        logger.debug(f"Created Epistula request from {keypair.ss58_address[:8]}... to {receiver_hotkey[:8]}...")
        return body, headers
    
    @staticmethod
    def verify_request(
        body: bytes,
        signature: str,
        max_age_ns: Optional[int] = None
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Verify an incoming Epistula request
        
        Args:
            body: Request body as bytes
            signature: Signature from Body-Signature header
            max_age_ns: Maximum age in nanoseconds (default: 5 seconds)
            
        Returns:
            Tuple of (is_valid, error_message, parsed_body)
        """
        if max_age_ns is None:
            max_age_ns = Epistula.ALLOWED_DELTA_NS
        
        try:
            if isinstance(body, bytes):
                body_str = body.decode('utf-8')
            else:
                body_str = body
            
            body_json = json.loads(body_str)
            
            required_fields = ['data', 'nonce', 'signed_by', 'signed_for']
            for field in required_fields:
                if field not in body_json:
                    return False, f"Missing required field: {field}", None
            
            if not signature or not signature.startswith('0x'):
                return False, "Invalid signature format", None
            
            current_time_ns = time.time_ns()
            nonce = body_json['nonce']
            
            if not isinstance(nonce, int):
                return False, "Invalid nonce type", None
            
            if nonce + max_age_ns < current_time_ns:
                age_seconds = (current_time_ns - nonce) / 1e9
                return False, f"Request too stale ({age_seconds:.1f}s old)", None
            
            sender_address = body_json['signed_by']
            keypair = Keypair(ss58_address=sender_address)
            
            # remove '0x' prefix from signature
            sig_bytes = bytes.fromhex(signature[2:])
            
            # recreate the exact signed message
            body_to_verify = json.dumps(body_json, sort_keys=True)
            
            verified = keypair.verify(body_to_verify, sig_bytes)
            
            if not verified:
                return False, "Signature verification failed", None
            
            logger.debug(f"Verified Epistula request from {sender_address[:8]}...")
            return True, None, body_json
            
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}", None
        except Exception as e:
            logger.error(f"Error verifying Epistula request: {e}")
            return False, f"Verification error: {e}", None
    
    @staticmethod
    def extract_sender(body_json: Dict) -> Optional[str]:
        """Extract sender's SS58 address from verified body"""
        return body_json.get('signed_by')
    
    @staticmethod
    def extract_receiver(body_json: Dict) -> Optional[str]:
        """Extract intended receiver's SS58 address from verified body"""
        return body_json.get('signed_for')
    
    @staticmethod
    def extract_data(body_json: Dict) -> Any:
        """Extract data payload from verified body"""
        return body_json.get('data', {})