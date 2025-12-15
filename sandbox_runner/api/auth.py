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
        api_key: Optional[str] = None
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
    Simple in-memory rate limiter for API endpoints.
    
    Uses a sliding window algorithm to track requests per identifier
    and enforces configurable rate limits.
    """
    
    # Cleanup stale identifiers after this many seconds of inactivity
    _CLEANUP_THRESHOLD_SECONDS = 300  # 5 minutes
    
    def __init__(self, requests_per_minute: int):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests allowed per minute per identifier
        """
        self.requests_per_minute = requests_per_minute
        self._request_log: dict[str, list[float]] = {}
        self._last_cleanup = time.time()
    
    async def check_rate_limit(self, identifier: str):
        """
        Check if request is within rate limit.
        
        Args:
            identifier: Unique identifier (API key or validator ID)
            
        Raises:
            HTTPException: 429 Too Many Requests if rate limit is exceeded
        """
        current_time = time.time()
        window_start = current_time - 60  # 1 minute sliding window
        
        # Periodically cleanup stale identifiers to prevent memory leak
        if current_time - self._last_cleanup > self._CLEANUP_THRESHOLD_SECONDS:
            self._cleanup_stale_identifiers(window_start)
            self._last_cleanup = current_time
        
        # Initialize request log for new identifiers
        if identifier not in self._request_log:
            self._request_log[identifier] = []
        
        # Remove timestamps outside the current window
        self._request_log[identifier] = [
            timestamp for timestamp in self._request_log[identifier]
            if timestamp > window_start
        ]
        
        # Check if rate limit would be exceeded
        if len(self._request_log[identifier]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per minute.",
                headers={"Retry-After": "60"}
            )
        
        # Record this request
        self._request_log[identifier].append(current_time)
    
    def _cleanup_stale_identifiers(self, window_start: float):
        """
        Remove identifiers with no recent requests to prevent memory leak.
        
        Args:
            window_start: Timestamp marking the start of the current window
        """
        stale_identifiers = [
            identifier for identifier, timestamps in self._request_log.items()
            if not timestamps or all(ts <= window_start for ts in timestamps)
        ]
        for identifier in stale_identifiers:
            del self._request_log[identifier]
    
    def get_remaining_requests(self, identifier: str) -> int:
        """
        Get the number of remaining requests for an identifier.
        
        Args:
            identifier: Unique identifier (API key or validator ID)
            
        Returns:
            Number of requests remaining in the current window
        """
        current_time = time.time()
        window_start = current_time - 60
        
        if identifier not in self._request_log:
            return self.requests_per_minute
        
        recent_requests = sum(
            1 for ts in self._request_log[identifier] if ts > window_start
        )
        return max(0, self.requests_per_minute - recent_requests)