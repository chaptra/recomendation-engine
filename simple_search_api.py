"""Simple search API for testing book search functionality."""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from simple_recommender import get_recommender
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Simple Book Search API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/search")
async def search_books(q: str = Query(..., description="Search query"), count: int = Query(10, description="Number of results")):
    """Search for books by title, author, or description."""
    try:
        recommender = get_recommender()
        
        # Search through the metadata
        search_term = q.lower().strip()
        if not search_term:
            return {"query": q, "results": [], "count": 0}
        
        # Search in the metadata DataFrame
        matching_books = []
        
        for idx, row in recommender.metadata.iterrows():
            title = str(row.get('title', '')).lower()
            authors = str(row.get('authors', '')).lower() 
            description = str(row.get('description', '')).lower()
            categories = str(row.get('categories', '')).lower()
            
            # Check if search term appears in any field
            if (search_term in title or 
                search_term in authors or 
                search_term in description or
                search_term in categories):
                
                book_data = {
                    "id": int(row['id']),
                    "title": str(row['title']),
                    "authors": str(row.get('authors', 'Unknown')),
                    "description": str(row.get('description', ''))[:200] + '...',
                    "categories": str(row.get('categories', 'General')),
                    "rating": float(row.get('average_rating', 0)),
                    "rating_count": int(row.get('rating_count', 0)),
                    "slug": str(row.get('slug', '')),
                    "cover_local_path": str(row.get('cover_local_path', ''))
                }
                
                matching_books.append(book_data)
                
                # Stop when we have enough results
                if len(matching_books) >= count:
                    break
        
        logger.info(f"Search for '{q}' found {len(matching_books)} results")
        
        return {
            "query": q,
            "results": matching_books,
            "count": len(matching_books)
        }
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {"query": q, "results": [], "count": 0, "error": str(e)}

@app.get("/health")
async def health():
    """Health check."""
    try:
        recommender = get_recommender()
        return {
            "status": "healthy",
            "total_books": len(recommender.metadata)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Simple Search API on port 8002")
    print("📚 Search endpoint: http://localhost:8002/search?q=sexy&count=10")
    uvicorn.run(
        "simple_search_api:app",
        host="0.0.0.0", 
        port=8002,
        reload=False,
        log_level="info"
    )