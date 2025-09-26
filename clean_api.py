from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import asyncio
from contextlib import asynccontextmanager
import os
from simple_recommender import get_recommender

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models
class BookDetails(BaseModel):
    id: int
    title: str
    slug: str
    cover_local_path: str
    authors: str
    categories: str
    description: str
    rating: float
    rating_count: int
    similarity_score: Optional[float] = None
    reason: Optional[str] = None

class RecommendationResponse(BaseModel):
    query: str
    found: bool
    message: Optional[str] = None
    source_book: Optional[BookDetails] = None
    suggestions: List[BookDetails]
    total: int

# Startup/Shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Initializing Book Recommendation Engine...")
    
    # Initialize recommender (loads data and computes similarity)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_recommender)
    
    logger.info("✅ Recommendation engine ready!")
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down...")

# Initialize FastAPI
app = FastAPI(
    title="Book Recommendation API",
    description="""
    🎯 **Simple & Fast Book Recommendations**
    
    **What it does:**
    - Takes a book title as input
    - Returns similar books you might like
    - Uses content-based filtering (not trained ML model)
    
    **How it works:**
    1. Analyzes book descriptions, titles, authors, categories
    2. Creates TF-IDF vectors for text similarity
    3. Uses cosine similarity to find similar books
    4. Returns ranked recommendations
    
    **Example:**
    `GET /suggest?title=Moby Dick&limit=10`
    
    **No Training Required:** Real-time recommendations using pre-computed similarity matrix.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Info"])
async def root():
    """API information"""
    return {
        "message": "Book Recommendation API",
        "version": "1.0.0",
        "usage": "GET /suggest?title=BOOK_TITLE&limit=10",
        "example": "/suggest?title=Moby Dick&limit=5",
        "docs": "/docs"
    }

@app.get("/suggest", response_model=RecommendationResponse, tags=["Recommendations"])
async def suggest_books(
    title: str = Query(..., description="Book title to get recommendations for"),
    limit: int = Query(10, ge=1, le=50, description="Number of recommendations (1-50)")
):
    """
    🎯 **Get Book Recommendations**
    
    **Input:** Book title (exact or partial match)
    **Output:** Similar books ranked by relevance
    
    **Algorithm:**
    - Content-based filtering using TF-IDF + cosine similarity
    - Analyzes: title, description, author, categories
    - No machine learning training required
    - Real-time similarity computation
    
    **Examples:**
    - `Moby Dick` → Sea adventures, Herman Melville books
    - `Harry Potter` → Fantasy, young adult fiction
    - `1984` → Dystopian fiction, George Orwell
    """
    
    try:
        # Get recommender instance
        recommender = get_recommender()
        
        # Get recommendations in thread pool (CPU intensive)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, recommender.get_recommendations, title, limit
        )
        
        # Convert to response model
        response_data = {
            "query": result["query"],
            "found": result["found"],
            "total": result.get("total", 0)
        }
        
        if result["found"]:
            # Source book
            source = result["source_book"]
            response_data["source_book"] = BookDetails(
                id=source["id"],
                title=source["title"],
                slug=source["slug"],
                cover_local_path=source["cover_local_path"],
                authors=source["authors"],
                categories=source["categories"],
                description=source["description"],
                rating=source["rating"],
                rating_count=source["rating_count"]
            )
            
            # Suggestions
            suggestions = []
            for book in result["suggestions"]:
                suggestions.append(BookDetails(
                    id=book["id"],
                    title=book["title"],
                    slug=book["slug"],
                    cover_local_path=book["cover_local_path"],
                    authors=book["authors"],
                    categories=book["categories"],
                    description=book["description"],
                    rating=book["rating"],
                    rating_count=book["rating_count"],
                    similarity_score=book.get("similarity_score"),
                    reason=book.get("reason")
                ))
            response_data["suggestions"] = suggestions
        else:
            response_data["message"] = result.get("message")
            response_data["suggestions"] = []
        
        return RecommendationResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error in suggest_books: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Recommendation engine error: {str(e)}"
        )

@app.get("/health", tags=["System"])
async def health_check():
    """Check if the recommendation engine is ready"""
    try:
        recommender = get_recommender()
        total_books = len(recommender.metadata) if recommender.metadata is not None else 0
        
        return {
            "status": "healthy",
            "message": "Book recommendation engine is operational",
            "total_books": total_books,
            "algorithm": "Content-based filtering with TF-IDF + Cosine Similarity",
            "training_required": False
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Recommendation engine not ready: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get('PORT', 8000))
    
    print("🚀 Starting Clean Book Recommendation API")
    print(f"📡 Server: http://localhost:{port}")
    print(f"📚 API Docs: http://localhost:{port}/docs")
    print(f"🎯 Main Endpoint: http://localhost:{port}/suggest?title=Moby Dick&limit=5")
    print("⚡ Algorithm: Content-based filtering (no training required)")
    
    uvicorn.run(
        "clean_api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )