import time
from typing import Optional, Set

from fastapi import Header, HTTPException, status, Request
from fastapi.security import APIKeyHeader
import os

from config import APIConfig


class APIKeyVerificationError(Exception):
    """Raised when API key verification fails."""
    pass


class AuthenticationManager:
    """
    Manages authentication for API requests using API keys    
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
        
        self._valid_api_keys: Set[str] = set(
            key.strip() for key in os.environ.get("API_KEYS", "").split(",") if key.strip()
        )
    
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
        return api_key in self._valid_api_keys
    
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


class RateLimiter:
    """
    Simple in-memory rate limiter for API endpoints    
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
            identifier: Unique identifier (API key)
            
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
        
        self._request_log[identifier].append(current_time)