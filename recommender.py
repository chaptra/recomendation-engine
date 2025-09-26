"""
Production-ready book recommendation engine.

This module provides a scalable, cached, and monitored recommendation system
following industry best practices.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from config import Config, load_config
from database import DatabaseManager
from cache_manager import CacheManager
from validators import InputValidator
from metrics import metrics_collector
from exceptions import (
    BookNotFoundError, 
    ModelNotReadyError, 
    RecommendationTimeoutError,
    CacheError
)

logger = logging.getLogger(__name__)


class ProductionBookRecommender:
    """
    Production-ready book recommendation engine with caching, monitoring, and error handling.
    
    Features:
    - Database connection pooling
    - Intelligent caching with TTL
    - Input validation and sanitization
    - Comprehensive monitoring and metrics
    - Proper error handling and logging
    - Type hints and documentation
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the recommendation engine.
        
        Args:
            config: Configuration object. If None, loads from environment.
        """
        self.config = config or load_config()
        self.db_manager = DatabaseManager(self.config.database)
        self.cache_manager = CacheManager(self.config.cache)
        
        # Model components
        self.metadata: Optional[pd.DataFrame] = None
        self.tfidf_matrix: Optional[csr_matrix] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.book_title_to_index: Dict[str, int] = {}
        self.index_to_book: Dict[int, Dict[str, Any]] = {}
        
        # Status tracking
        self._is_ready = False
        self._last_loaded = None
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize or load the recommendation model."""
        try:
            logger.info("Initializing recommendation model...")
            start_time = time.time()
            
            # Try to load from cache first
            if self._load_from_cache():
                logger.info(f"Model loaded from cache in {time.time() - start_time:.2f}s")
            else:
                logger.info("Cache miss - building model from database...")
                self._build_model_from_database()
                logger.info(f"Model built from database in {time.time() - start_time:.2f}s")
            
            self._is_ready = True
            self._last_loaded = time.time()
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self._is_ready = False
            raise ModelNotReadyError(f"Failed to initialize model: {e}")
    
    def _get_cache_key(self) -> str:
        """Generate cache key based on database state and model configuration."""
        try:
            data_hash = self.db_manager.get_cache_key_data()
            
            model_config = {
                'max_features': self.config.model.max_features,
                'min_df': self.config.model.min_df,
                'max_df': self.config.model.max_df,
                'ngram_range': self.config.model.ngram_range,
                'description_max_length': self.config.model.description_max_length
            }
            
            config_hash = self.cache_manager.get_config_hash(model_config)
            return self.cache_manager.generate_cache_key(data_hash, config_hash)
            
        except Exception as e:
            logger.error(f"Cache key generation failed: {e}")
            # Fallback to timestamp-based key
            return f"fallback_{int(time.time())}"
    
    def _load_from_cache(self) -> bool:
        """
        Attempt to load model from cache.
        
        Returns:
            True if successfully loaded from cache
        """
        try:
            cache_key = self._get_cache_key()
            
            if not self.cache_manager.cache_exists(cache_key):
                logger.info("No valid cache found")
                return False
            
            # Load from cache
            metadata, tfidf_matrix, vectorizer, mappings = self.cache_manager.load_cache(cache_key)
            
            self.metadata = metadata
            self.tfidf_matrix = tfidf_matrix
            self.tfidf_vectorizer = vectorizer
            self.book_title_to_index = mappings['title_to_index']
            self.index_to_book = mappings['index_to_book']
            
            logger.info(f"Successfully loaded {len(self.metadata)} books from cache")
            return True
            
        except CacheError:
            logger.warning("Cache loading failed, will rebuild from database")
            return False
        except Exception as e:
            logger.error(f"Unexpected error loading cache: {e}")
            return False
    
    def _build_model_from_database(self) -> None:
        """Build model from database and save to cache."""
        try:
            # Load data from database
            self.metadata = self.db_manager.load_books_data()
            logger.info(f"Loaded {len(self.metadata)} books from database")
            
            # Prepare text features
            self._prepare_text_features()
            
            # Build TF-IDF matrix
            self._build_tfidf_matrix()
            
            # Create book mappings
            self._create_book_mappings()
            
            # Save to cache
            self._save_to_cache()
            
        except Exception as e:
            logger.error(f"Model building failed: {e}")
            raise
    
    def _prepare_text_features(self) -> None:
        """Prepare and clean text features for TF-IDF."""
        logger.info("Preparing text features...")
        
        # Combine and clean text
        self.metadata['clean_text'] = (
            self.metadata['title'].fillna('') + ' ' + 
            self.metadata['description'].fillna('').str[:self.config.model.description_max_length] + ' ' +
            self.metadata['authors'].fillna('') + ' ' +
            self.metadata['categories'].fillna('')
        ).str.lower().str.replace(r'[^a-zA-Z\s]', ' ', regex=True)
        
        # Normalize whitespace
        self.metadata['clean_text'] = self.metadata['clean_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    def _build_tfidf_matrix(self) -> None:
        """Build TF-IDF matrix from clean text."""
        logger.info("Building TF-IDF matrix...")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config.model.max_features,
            stop_words='english',
            ngram_range=self.config.model.ngram_range,
            min_df=self.config.model.min_df,
            max_df=self.config.model.max_df,
            dtype=np.float32  # Memory optimization
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.metadata['clean_text'])
        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
    
    def _create_book_mappings(self) -> None:
        """Create mappings between book titles/indices and book data."""
        logger.info("Creating book mappings...")
        
        self.book_title_to_index = {}
        self.index_to_book = {}
        
        for idx, row in self.metadata.iterrows():
            title_key = row['title'].lower().strip()
            self.book_title_to_index[title_key] = idx
            
            self.index_to_book[idx] = {
                'id': int(row['id']),
                'title': str(row['title']),
                'slug': str(row['slug']) if pd.notna(row['slug']) else '',
                'cover_local_path': str(row['cover_local_path']) if pd.notna(row['cover_local_path']) else '',
                'authors': str(row['authors']) if pd.notna(row['authors']) else 'Unknown',
                'categories': str(row['categories']) if pd.notna(row['categories']) else 'General',
                'description': str(row['description'])[:300] + ('...' if len(str(row['description'])) > 300 else ''),
                'rating': float(row['average_rating']) if pd.notna(row['average_rating']) else 0.0,
                'rating_count': int(row['rating_count']) if pd.notna(row['rating_count']) else 0
            }
    
    def _save_to_cache(self) -> None:
        """Save model to cache."""
        try:
            cache_key = self._get_cache_key()
            
            mappings = {
                'title_to_index': self.book_title_to_index,
                'index_to_book': self.index_to_book
            }
            
            self.cache_manager.save_cache(
                cache_key,
                self.metadata,
                self.tfidf_matrix,
                self.tfidf_vectorizer,
                mappings
            )
            
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
            # Don't raise - model is still functional
    
    def find_book_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Find book by exact or partial title match.
        
        Args:
            title: Book title to search for
            
        Returns:
            Book data if found, None otherwise
        """
        if not self._is_ready:
            raise ModelNotReadyError("Model is not ready")
        
        title_clean = InputValidator.sanitize_text_for_search(title).lower().strip()
        
        # Exact match first
        if title_clean in self.book_title_to_index:
            idx = self.book_title_to_index[title_clean]
            return self.index_to_book[idx]
        
        # Partial match
        for book_title, idx in self.book_title_to_index.items():
            if title_clean in book_title or book_title in title_clean:
                return self.index_to_book[idx]
        
        return None
    
    def get_recommendations(
        self, 
        book_title: str, 
        limit: int = 10,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get book recommendations based on title.
        
        Args:
            book_title: Title of the book to get recommendations for
            limit: Maximum number of recommendations to return
            similarity_threshold: Minimum similarity score to include
            
        Returns:
            Dictionary containing recommendations and metadata
            
        Raises:
            ModelNotReadyError: If model is not initialized
            BookNotFoundError: If book is not found
            InvalidInputError: If inputs are invalid
        """
        # Validate inputs
        validated_title = InputValidator.validate_book_title(book_title)
        validated_limit = InputValidator.validate_limit(limit)
        
        if similarity_threshold is None:
            similarity_threshold = self.config.model.similarity_threshold
        else:
            similarity_threshold = InputValidator.validate_similarity_threshold(similarity_threshold)
        
        # Check if model is ready
        if not self._is_ready:
            raise ModelNotReadyError("Recommendation model is not ready")
        
        # Track request metrics
        with metrics_collector.track_request(validated_title) as request_metric:
            start_time = time.time()
            
            try:
                # Find source book
                source_book = self.find_book_by_title(validated_title)
                if not source_book:
                    # Try broader search
                    matching_books = self._search_books_partial(validated_title)
                    if not matching_books:
                        raise BookNotFoundError(f'No books found matching "{validated_title}"')
                    source_book = matching_books[0]
                
                # Get source book index
                source_idx = self._get_book_index(source_book['id'])
                if source_idx is None:
                    raise BookNotFoundError("Book index not found")
                
                # Check cache first (if we had pre-computed similarities)
                request_metric.cache_hit = False  # We compute on-demand
                
                # Compute similarities on-demand
                similarities = self._compute_similarities(source_idx)
                
                # Get recommendations
                recommendations = self._filter_and_rank_recommendations(
                    similarities, source_idx, source_book, validated_limit, similarity_threshold
                )
                
                request_metric.recommendations_count = len(recommendations)
                
                # Check for timeout
                if time.time() - start_time > self.config.api.timeout_seconds:
                    raise RecommendationTimeoutError("Recommendation computation timed out")
                
                return {
                    'query': validated_title,
                    'found': True,
                    'source_book': source_book,
                    'recommendations': recommendations,
                    'total': len(recommendations),
                    'similarity_threshold': similarity_threshold,
                    'computation_time_ms': (time.time() - start_time) * 1000
                }
                
            except Exception as e:
                logger.error(f"Recommendation failed for '{validated_title}': {e}")
                raise
    
    def _search_books_partial(self, title: str) -> List[Dict[str, Any]]:
        """Search for books with partial title match."""
        matching_books = []
        title_lower = title.lower()
        
        for idx, row in self.metadata.iterrows():
            if title_lower in row['title'].lower():
                matching_books.append(self.index_to_book[idx])
                
        return matching_books
    
    def _get_book_index(self, book_id: int) -> Optional[int]:
        """Get book index by ID."""
        for idx, book in self.index_to_book.items():
            if book['id'] == book_id:
                return idx
        return None
    
    def _compute_similarities(self, source_idx: int) -> np.ndarray:
        """Compute similarity scores for source book against all others."""
        source_vector = self.tfidf_matrix[source_idx]
        similarities = cosine_similarity(source_vector, self.tfidf_matrix).flatten()
        return similarities
    
    def _filter_and_rank_recommendations(
        self,
        similarities: np.ndarray,
        source_idx: int,
        source_book: Dict[str, Any],
        limit: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Filter and rank recommendations based on similarity scores."""
        # Get similarity scores excluding source book
        sim_scores = [
            (i, similarities[i]) 
            for i in range(len(similarities)) 
            if i != source_idx and similarities[i] >= threshold
        ]
        
        # Sort by similarity score
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build recommendations with deduplication
        recommendations = []
        added_books = {source_book['id']}  # Track to avoid duplicates
        
        for idx, score in sim_scores:
            if len(recommendations) >= limit:
                break
                
            candidate_book = self.index_to_book[idx].copy()
            
            # Skip if already added
            if candidate_book['id'] in added_books:
                continue
            
            # Skip if duplicate of source book
            if self._is_duplicate_book(candidate_book, source_book):
                continue
            
            # Add recommendation
            candidate_book['similarity_score'] = round(float(score), 3)
            candidate_book['reason'] = self._get_recommendation_reason(score)
            
            recommendations.append(candidate_book)
            added_books.add(candidate_book['id'])
        
        return recommendations
    
    def _is_duplicate_book(self, book1: Dict[str, Any], book2: Dict[str, Any]) -> bool:
        """Check if two books are duplicates."""
        title1 = book1['title'].lower().strip()
        title2 = book2['title'].lower().strip()
        author1 = book1['authors'].lower().strip()
        author2 = book2['authors'].lower().strip()
        
        # Same author and very similar titles
        if author1 == author2:
            if title1 == title2:
                return True
            if title1 in title2 or title2 in title1:
                return True
        
        return False
    
    def _get_recommendation_reason(self, score: float) -> str:
        """Generate human-readable reason for recommendation."""
        if score > 0.7:
            return "Very similar content and themes"
        elif score > 0.5:
            return "Similar themes and writing style"
        elif score > 0.3:
            return "Shared categories and topics"
        else:
            return "Related content"
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the recommendation system.
        
        Returns:
            Health status and metrics
        """
        try:
            return {
                'status': 'healthy' if self._is_ready else 'unhealthy',
                'model_ready': self._is_ready,
                'last_loaded': self._last_loaded,
                'books_count': len(self.metadata) if self.metadata is not None else 0,
                'cache_stats': self.cache_manager.get_cache_stats(),
                'system_metrics': metrics_collector.get_system_health(),
                'performance_stats': metrics_collector.get_performance_stats(60)
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def reload_model(self) -> Dict[str, Any]:
        """
        Reload the model from database (force refresh).
        
        Returns:
            Reload status and timing
        """
        try:
            start_time = time.time()
            self._is_ready = False
            
            # Clear cache for this model
            cache_key = self._get_cache_key()
            self.cache_manager.clear_cache(cache_key)
            
            # Rebuild model
            self._initialize_model()
            
            reload_time = time.time() - start_time
            
            return {
                'status': 'success',
                'reload_time_seconds': reload_time,
                'books_count': len(self.metadata),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Model reload failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            return {
                'model_stats': {
                    'ready': self._is_ready,
                    'books_count': len(self.metadata) if self.metadata is not None else 0,
                    'last_loaded': self._last_loaded,
                    'tfidf_features': self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0
                },
                'performance_stats': metrics_collector.get_performance_stats(60),
                'cache_stats': self.cache_manager.get_cache_stats(),
                'popular_books': metrics_collector.get_popular_books(10, 1440),
                'error_breakdown': metrics_collector.get_error_breakdown(1440)
            }
        except Exception as e:
            logger.error(f"Stats collection failed: {e}")
            return {'error': str(e)}


# Global recommender instance
_recommender_instance: Optional[ProductionBookRecommender] = None


def get_recommender() -> ProductionBookRecommender:
    """Get global recommender instance (singleton pattern)."""
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = ProductionBookRecommender()
    return _recommender_instance