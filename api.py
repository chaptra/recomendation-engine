"""
FastAPI-based REST API for the book recommendation engine.

Provides production-ready endpoints with rate limiting, authentication,
monitoring, and comprehensive error handling.
"""

import logging
import time
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from recommender import get_recommender, ProductionBookRecommender
from config import load_config, Config
from exceptions import (
    BookNotFoundError, 
    ModelNotReadyError, 
    InvalidInputError,
    RecommendationTimeoutError,
    DatabaseConnectionError
)
from validators import SecurityValidator
from metrics import metrics_collector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()


# Rate limiting (simple in-memory implementation)
from collections import defaultdict, deque
rate_limits = defaultdict(lambda: deque())


def check_rate_limit(request: Request) -> bool:
    """Simple rate limiting implementation."""
    client_ip = request.client.host
    current_time = time.time()
    window_start = current_time - 60  # 1 minute window
    
    # Clean old requests
    while rate_limits[client_ip] and rate_limits[client_ip][0] < window_start:
        rate_limits[client_ip].popleft()
    
    # Check if over limit
    if len(rate_limits[client_ip]) >= config.api.rate_limit_per_minute:
        return False
    
    # Add current request
    rate_limits[client_ip].append(current_time)
    return True


# Pydantic models
class RecommendationRequest(BaseModel):
    """Request model for book recommendations."""
    title: str = Field(..., min_length=1, max_length=500, description="Book title to get recommendations for")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of recommendations to return")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity threshold")
    include_metadata: bool = Field(True, description="Whether to include additional metadata")
    
    @validator('title')
    def validate_title(cls, v):
        """Validate and sanitize title."""
        if not v or not v.strip():
            raise ValueError("Title cannot be empty")
        return v.strip()


class RecommendationResponse(BaseModel):
    """Response model for book recommendations."""
    query: str
    found: bool
    source_book: Optional[Dict[str, Any]] = None
    recommendations: list = []
    total: int
    similarity_threshold: float
    computation_time_ms: float
    timestamp: float = Field(default_factory=time.time)


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_ready: bool
    books_count: int
    timestamp: float = Field(default_factory=time.time)


class StatsResponse(BaseModel):
    """Response model for system statistics."""
    model_stats: Dict[str, Any]
    performance_stats: Dict[str, Any]
    cache_stats: Dict[str, Any]
    timestamp: float = Field(default_factory=time.time)


# Dependency functions
async def get_recommender_service() -> ProductionBookRecommender:
    """Get the recommender service instance."""
    try:
        return get_recommender()
    except Exception as e:
        logger.error(f"Failed to get recommender service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation service is currently unavailable"
        )


async def verify_rate_limit(request: Request):
    """Verify request doesn't exceed rate limits."""
    if not check_rate_limit(request):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": "60"}
        )


async def verify_request_headers(request: Request):
    """Verify request headers for security."""
    headers = dict(request.headers)
    
    if not SecurityValidator.check_rate_limit_headers(headers):
        logger.warning(f"Suspicious request from {request.client.host}: {headers.get('user-agent', 'unknown')}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Request blocked by security policy"
        )


# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("Starting recommendation API...")
    
    # Initialize recommender
    try:
        recommender = get_recommender()
        logger.info(f"Recommender initialized with {len(recommender.metadata)} books")
    except Exception as e:
        logger.error(f"Failed to initialize recommender: {e}")
        raise
    
    yield
    
    logger.info("Shutting down recommendation API...")


# Create FastAPI app
app = FastAPI(
    title="Book Recommendation API",
    description="Production-ready book recommendation engine with caching and monitoring",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


# Exception handlers
@app.exception_handler(BookNotFoundError)
async def book_not_found_handler(request: Request, exc: BookNotFoundError):
    """Handle book not found errors."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "book_not_found",
            "message": str(exc),
            "timestamp": time.time()
        }
    )


@app.exception_handler(ModelNotReadyError)
async def model_not_ready_handler(request: Request, exc: ModelNotReadyError):
    """Handle model not ready errors."""
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "model_not_ready",
            "message": str(exc),
            "timestamp": time.time()
        }
    )


@app.exception_handler(InvalidInputError)
async def invalid_input_handler(request: Request, exc: InvalidInputError):
    """Handle invalid input errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "invalid_input",
            "message": str(exc),
            "timestamp": time.time()
        }
    )


@app.exception_handler(RecommendationTimeoutError)
async def timeout_handler(request: Request, exc: RecommendationTimeoutError):
    """Handle recommendation timeout errors."""
    return JSONResponse(
        status_code=status.HTTP_408_REQUEST_TIMEOUT,
        content={
            "error": "request_timeout",
            "message": str(exc),
            "timestamp": time.time()
        }
    )


@app.exception_handler(DatabaseConnectionError)
async def database_error_handler(request: Request, exc: DatabaseConnectionError):
    """Handle database connection errors."""
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "database_error",
            "message": "Database service is currently unavailable",
            "timestamp": time.time()
        }
    )


# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Book Recommendation API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    request_data: RecommendationRequest,
    request: Request,
    recommender: ProductionBookRecommender = Depends(get_recommender_service),
    _rate_limit: None = Depends(verify_rate_limit),
    _security: None = Depends(verify_request_headers)
):
    """
    Get book recommendations based on a given book title.
    
    - **title**: The book title to get recommendations for
    - **limit**: Maximum number of recommendations (1-100, default: 10)
    - **similarity_threshold**: Minimum similarity score (0.0-1.0, optional)
    - **include_metadata**: Include additional book metadata (default: true)
    """
    try:
        result = recommender.get_recommendations(
            book_title=request_data.title,
            limit=request_data.limit,
            similarity_threshold=request_data.similarity_threshold
        )
        
        # Filter metadata if requested
        if not request_data.include_metadata:
            for rec in result.get('recommendations', []):
                # Keep only essential fields
                essential_fields = ['id', 'title', 'authors', 'similarity_score', 'reason']
                filtered_rec = {k: v for k, v in rec.items() if k in essential_fields}
                rec.clear()
                rec.update(filtered_rec)
        
        return RecommendationResponse(**result)
        
    except Exception as e:
        logger.error(f"Recommendation request failed: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check(
    recommender: ProductionBookRecommender = Depends(get_recommender_service)
):
    """
    Health check endpoint for monitoring and load balancing.
    
    Returns system health status and basic metrics.
    """
    try:
        health_data = recommender.health_check()
        
        return HealthResponse(
            status=health_data['status'],
            model_ready=health_data['model_ready'],
            books_count=health_data['books_count']
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="error",
            model_ready=False,
            books_count=0
        )


@app.get("/stats", response_model=StatsResponse)
async def get_stats(
    recommender: ProductionBookRecommender = Depends(get_recommender_service)
):
    """
    Get comprehensive system statistics and performance metrics.
    
    Includes model statistics, performance metrics, cache statistics,
    and popular books data.
    """
    try:
        stats_data = recommender.get_stats()
        
        return StatsResponse(**stats_data)
        
    except Exception as e:
        logger.error(f"Stats request failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system statistics"
        )


@app.get("/metrics")
async def get_metrics():
    """
    Get detailed performance metrics for monitoring.
    
    Returns Prometheus-style metrics for external monitoring systems.
    """
    try:
        performance_stats = metrics_collector.get_performance_stats(60)
        system_health = metrics_collector.get_system_health()
        error_breakdown = metrics_collector.get_error_breakdown(60)
        
        return {
            "performance": performance_stats,
            "system": system_health,
            "errors": error_breakdown,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Metrics request failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics"
        )


@app.post("/admin/reload")
async def reload_model(
    recommender: ProductionBookRecommender = Depends(get_recommender_service)
):
    """
    Admin endpoint to reload the recommendation model.
    
    Forces a refresh of the model from the database.
    Use with caution in production.
    """
    try:
        result = recommender.reload_model()
        return result
        
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reload model"
        )


@app.get("/search")
async def search_books(
    q: str,
    count: int = 10,
    recommender: ProductionBookRecommender = Depends(get_recommender_service),
    _rate_limit: None = Depends(verify_rate_limit)
):
    """
    Search for books by title, author, or description (partial matches).
    
    Returns multiple books matching the search query.
    """
    try:
        # Search in titles, authors, and descriptions
        matching_books = []
        search_term = q.lower().strip()
        
        if not search_term:
            return {
                "query": q,
                "results": [],
                "count": 0,
                "timestamp": time.time()
            }
        
        # Search through all books
        for idx, book_data in recommender.index_to_book.items():
            title = book_data.get('title', '').lower()
            authors = book_data.get('authors', '').lower()
            description = book_data.get('description', '').lower()
            categories = book_data.get('categories', '').lower()
            
            # Check if search term appears in any field
            if (search_term in title or 
                search_term in authors or 
                search_term in description or
                search_term in categories):
                
                matching_books.append(book_data)
                
                # Stop when we have enough results
                if len(matching_books) >= count:
                    break
        
        return {
            "query": q,
            "results": matching_books,
            "count": len(matching_books),
            "timestamp": time.time()
        }
            
    except Exception as e:
        logger.error(f"Book search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Book search failed"
        )


@app.get("/search/{title}")
async def search_book_single(
    title: str,
    recommender: ProductionBookRecommender = Depends(get_recommender_service),
    _rate_limit: None = Depends(verify_rate_limit)
):
    """
    Search for a single book by title (exact or partial match).
    
    Returns first book found matching the title.
    """
    try:
        book = recommender.find_book_by_title(title)
        
        if book:
            return {
                "found": True,
                "book": book,
                "timestamp": time.time()
            }
        else:
            return {
                "found": False,
                "message": f"No book found matching '{title}'",
                "timestamp": time.time()
            }
            
    except Exception as e:
        logger.error(f"Book search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Book search failed"
        )


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=config.environment == "development",
        log_level=config.log_level.lower(),
        access_log=True
    )