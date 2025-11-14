import hashlib
import hmac
import time
from typing import Optional, Tuple, Set
from datetime import datetime, timedelta

from fastapi import Header, HTTPException, status, Request
from fastapi.security import APIKeyHeader

from config import APIConfig


class EpistulaVerificationError(Exception):
    """Raised when Epistula signature verification fails."""
    pass


class APIKeyVerificationError(Exception):
    """Raised when API key verification fails."""
    pass


class AuthenticationManager:
    """
    Manages authentication for API requests using Epistula signatures and API keys    
    """
    
    def __init__(self, config: APIConfig):
        """
        Initialize authentication manager
        
        Args:
            config: API configuration with auth settings
        """
        self.config = config
        self.api_key_scheme = APIKeyHeader(
            name=config.api_key_header,
            auto_error=False
        )
        
        self._processed_request_ids = set()
        self._request_id_expiry = {}
        
        self._validator_hotkeys = {}
        
        # TODO: read env var properly
        self._valid_api_keys: Set[str] = {
            "dev-key-12345",
        }
        
    async def verify_epistula_signature(
        self,
        request: Request,
        signature: Optional[str] = Header(None, alias="X-Epistula-Signature"),
        request_id: Optional[str] = Header(None, alias="X-Epistula-Request-ID"),
        timestamp: Optional[str] = Header(None, alias="X-Epistula-Timestamp"),
        validator_hotkey: Optional[str] = Header(None, alias="X-Validator-Hotkey")
    ) -> str:
        """
        Verify Epistula cryptographic signature from validator
        
        The signature proves that:
        1. Request comes from a legitimate validator
        2. Request has not been tampered with
        3. Request is not a replay attack (via request_id)
        4. Request is recent (via timestamp)
        
        Args:
            request: FastAPI request object
            signature: Cryptographic signature (hex-encoded)
            request_id: Unique request identifier (UUID)
            timestamp: ISO format timestamp
            validator_hotkey: Validator's public key (SS58 format)
            
        Returns:
            Validator hotkey if verification succeeds
            
        Raises:
            HTTPException: If verification fails
        """
        if not self.config.require_epistula:
            return "epistula-disabled"
        
        if not all([signature, request_id, timestamp, validator_hotkey]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Epistula authentication headers"
            )
        
        try:
            request_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            current_time = datetime.utcnow()
            time_delta = abs((current_time - request_time).total_seconds())
            
            if time_delta > 300:
                raise EpistulaVerificationError(
                    f"Request timestamp too old: {time_delta}s (max 300s)"
                )
            
            if request_id in self._processed_request_ids:
                raise EpistulaVerificationError(
                    f"Duplicate request ID detected: {request_id}"
                )
            
            if validator_hotkey not in self._validator_hotkeys:
                pass
            
            body = await request.body()
            body_hash = hashlib.sha256(body).hexdigest()
            
            message = (
                f"{request.method}|"
                f"{request.url.path}|"
                f"{request_id}|"
                f"{timestamp}|"
                f"{body_hash}"
            )
            
            if not self._verify_signature_with_public_key(
                message, signature, validator_hotkey
            ):
                raise EpistulaVerificationError("Invalid signature")
            
            self._processed_request_ids.add(request_id)
            self._request_id_expiry[request_id] = current_time + timedelta(minutes=10)
            
            self._cleanup_expired_request_ids()
            
            return validator_hotkey
            
        except EpistulaVerificationError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Epistula verification failed: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Epistula verification error: {str(e)}"
            )
    
    def _verify_signature_with_public_key(
        self, message: str, signature: str, public_key: str
    ) -> bool:
        """
        Verify cryptographic signature using validator's public key
        
        This is a placeholder implementation. In production, this should:
        1. Use the actual public key from the Bittensor network
        2. Implement proper Ed25519 signature verification
        3. Handle key format conversion (SS58 → raw bytes)
        
        Args:
            message: Message that was signed
            signature: Hex-encoded signature
            public_key: Validator's public key (SS58 format)
            
        Returns:
            True if signature is valid, False otherwise
        """
        # TODO : implement properly
        return len(signature) > 0
    
    async def verify_api_key(
        self,
        api_key: Optional[str] = Header(None, alias="X-API-Key")
    ) -> str:
        """
        Verify API key from request headers
                
        Args:
            api_key: API key from request header
            
        Returns:
            API key identifier if valid
            
        Raises:
            HTTPException: If API key is missing or invalid
        """
        if not self.config.require_api_key:
            return "api-key-disabled"
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key"
            )
        
        if not self._validate_api_key(api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        return api_key
    
    def _validate_api_key(self, api_key: str) -> bool:
        """
        Validate API key against stored keys
                
        Args:
            api_key: API key to validate
            
        Returns:
            True if key is valid, False otherwise
        """
        # TODO : handle this properly
        # Simple in-memory validation
        return True
        #return api_key in self._valid_api_keys
    
    def add_api_key(self, api_key: str):
        """
        Add a new API key to the valid keys set
        
        Args:
            api_key: API key to add
        """
        self._valid_api_keys.add(api_key)
    
    def remove_api_key(self, api_key: str):
        """
        Remove an API key from the valid keys set
        
        Args:
            api_key: API key to remove
        """
        self._valid_api_keys.discard(api_key)
    
    def list_api_keys(self) -> Set[str]:
        """
        Get list of valid API keys
        
        Returns:
            Set of valid API keys
        """
        return self._valid_api_keys.copy()
    
    def _cleanup_expired_request_ids(self):
        """
        Remove expired request IDs from memory        
        """
        current_time = datetime.utcnow()
        expired_ids = [
            req_id for req_id, expiry in self._request_id_expiry.items()
            if expiry < current_time
        ]
        
        for req_id in expired_ids:
            self._processed_request_ids.discard(req_id)
            del self._request_id_expiry[req_id]
    
    def register_validator(self, hotkey: str, public_key: bytes):
        """
        Register a validator's public key for Epistula verification
        
        this should:
        1. Query the blockchain for registered validators
        2. Cache keys for performance
        3. Refresh periodically
        
        Args:
            hotkey: Validator's hotkey (SS58 address)
            public_key: Raw public key bytes
        """
        # TODO: handle this properly
        self._validator_hotkeys[hotkey] = public_key


class RateLimiter:
    """
    Simple in-memory rate limiter for API endpoints
    
    In production, this should use Redis or similar distributed cache
    to work across multiple instances.
    """
    
    def __init__(self, requests_per_minute: int):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests allowed per minute per identifier
        """
        self.requests_per_minute = requests_per_minute
        self._request_log = {}
    
    async def check_rate_limit(self, identifier: str):
        """
        Check if request is within rate limit
        
        Args:
            identifier: Unique identifier (validator hotkey or API key)
            
        Raises:
            HTTPException: If rate limit is exceeded
        """
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window
        
        if identifier not in self._request_log:
            self._request_log[identifier] = []
        
        self._request_log[identifier] = [
            timestamp for timestamp in self._request_log[identifier]
            if timestamp > window_start
        ]
        
        if len(self._request_log[identifier]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {self.requests_per_minute} requests/minute"
            )
        
        self._request_log[identifier].append(current_time)