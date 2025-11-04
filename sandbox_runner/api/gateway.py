"""
API Gateway - FastAPI Application (Updated for Phase 2)

Integrates Meta-Manager for job orchestration.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import logging

from config import Config
from api.auth import AuthenticationManager, RateLimiter
from monitoring.metrics import metrics_manager

logger = logging.getLogger("api.gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Startup:
    - Initialize metrics
    - Initialize meta-manager
    - Start background job processor
    
    Shutdown:
    - Stop meta-manager
    - Clean up resources
    """
    logger.info("Starting Hone Subnet Sandbox Runner API")
    
    # Initialize metrics
    metrics_manager.initialize()
    
    # Initialize and start meta-manager
    from core.meta_manager import MetaManager
    meta_manager = MetaManager(app.state.config)
    app.state.meta_manager = meta_manager
    await meta_manager.start()
    
    logger.info("Meta-Manager started successfully")
    
    yield
    
    logger.info("Shutting down Hone Subnet Sandbox Runner API")
    
    # Stop meta-manager
    await meta_manager.stop()
    
    logger.info("Shutdown complete")


def create_app(config: Config) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        config: Application configuration
        
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Hone Subnet Sandbox Runner API",
        description="Secure GPU execution service for Bittensor subnet miners",
        version="1.0.0",
        docs_url="/docs" if config.api.port != 8443 else None,
        redoc_url="/redoc" if config.api.port != 8443 else None,
        lifespan=lifespan
    )
    
    # Store config in app state
    app.state.config = config
    
    # Initialize auth and rate limiting
    app.state.auth_manager = AuthenticationManager(config.api)
    app.state.rate_limiter = RateLimiter(config.api.rate_limit_per_validator)
    
    # Add middleware
    _add_middleware(app, config)
    
    # Add exception handlers
    _add_exception_handlers(app)
    
    # Add routes
    from api.routes import create_router
    router = create_router(config)
    app.include_router(router, prefix="/v1")
    
    # Add metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        metrics_data, content_type = metrics_manager.export_metrics()
        return JSONResponse(
            content=metrics_data.decode('utf-8'),
            media_type=content_type
        )
    
    @app.get("/", tags=["health"])
    async def root():
        """Root endpoint."""
        return {
            "service": "Hone Subnet Sandbox Runner",
            "version": "1.0.0",
            "status": "operational"
        }
    
    return app


def _add_middleware(app: FastAPI, config: Config):
    """Add middleware to the application."""
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://localhost"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests with timing."""
        import time
        
        start_time = time.time()
        
        logger.info(
            f"Request: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else None
            }
        )
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        logger.info(
            f"Response: {response.status_code} ({process_time:.3f}s)",
            extra={
                "status_code": response.status_code,
                "process_time": process_time
            }
        )
        
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        """Add security headers."""
        response = await call_next(request)
        
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response


def _add_exception_handlers(app: FastAPI):
    """Add custom exception handlers."""
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle ValueError with 400."""
        logger.warning(f"ValueError: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Bad Request",
                "detail": str(exc)
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions with 500."""
        logger.exception(f"Unexpected error: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "detail": "An unexpected error occurred"
            }
        )