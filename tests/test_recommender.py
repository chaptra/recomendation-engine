"""Tests for the production book recommender."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from scipy.sparse import csr_matrix

from recommender import ProductionBookRecommender, get_recommender
from config import Config, ModelConfig, DatabaseConfig, CacheConfig, APIConfig
from exceptions import (
    BookNotFoundError, 
    ModelNotReadyError, 
    InvalidInputError,
    DatabaseConnectionError
)


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return Config(
        database=DatabaseConfig(url="postgresql://test:test@localhost:5432/test"),
        cache=CacheConfig(directory="./test_cache", ttl_hours=1),
        model=ModelConfig(max_features=100, min_df=1, max_df=0.9),
        api=APIConfig(max_recommendations=20, timeout_seconds=10),
        log_level="DEBUG"
    )


@pytest.fixture
def sample_books_data():
    """Create sample books data for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'title': [
            'The Great Gatsby',
            'To Kill a Mockingbird', 
            'Pride and Prejudice',
            'The Catcher in the Rye',
            '1984'
        ],
        'slug': ['gatsby', 'mockingbird', 'pride', 'catcher', '1984'],
        'cover_local_path': ['path1.jpg', 'path2.jpg', 'path3.jpg', 'path4.jpg', 'path5.jpg'],
        'description': [
            'A story of decadence and excess in the Jazz Age',
            'A gripping tale of racial injustice and childhood innocence',
            'A romantic novel about manners and marriage',
            'A coming-of-age story in post-war America',
            'A dystopian social science fiction novel'
        ],
        'average_rating': [4.2, 4.5, 4.3, 3.8, 4.6],
        'rating_count': [1000, 1500, 1200, 800, 2000],
        'authors': ['F. Scott Fitzgerald', 'Harper Lee', 'Jane Austen', 'J.D. Salinger', 'George Orwell'],
        'categories': ['Fiction|Classic', 'Fiction|Drama', 'Romance|Classic', 'Fiction|Coming-of-age', 'Science Fiction|Dystopian']
    })


class TestProductionBookRecommender:
    """Test cases for ProductionBookRecommender."""
    
    @patch('recommender.DatabaseManager')
    @patch('recommender.CacheManager')
    def test_initialization_with_cache_hit(self, mock_cache_manager, mock_db_manager, mock_config, sample_books_data):
        """Test successful initialization with cache hit."""
        # Mock cache hit
        mock_cache_instance = Mock()
        mock_cache_instance.cache_exists.return_value = True
        mock_cache_instance.load_cache.return_value = (
            sample_books_data,
            csr_matrix(np.random.rand(5, 100)),
            Mock(),
            {'title_to_index': {}, 'index_to_book': {}}
        )
        mock_cache_manager.return_value = mock_cache_instance
        
        # Mock database manager
        mock_db_instance = Mock()
        mock_db_instance.get_cache_key_data.return_value = "test_hash"
        mock_db_manager.return_value = mock_db_instance
        
        recommender = ProductionBookRecommender(mock_config)
        
        assert recommender._is_ready is True
        assert len(recommender.metadata) == 5
        mock_cache_instance.load_cache.assert_called_once()
    
    @patch('recommender.DatabaseManager')
    @patch('recommender.CacheManager')
    def test_initialization_with_cache_miss(self, mock_cache_manager, mock_db_manager, mock_config, sample_books_data):
        """Test initialization with cache miss (build from database)."""
        # Mock cache miss
        mock_cache_instance = Mock()
        mock_cache_instance.cache_exists.return_value = False
        mock_cache_manager.return_value = mock_cache_instance
        
        # Mock database manager
        mock_db_instance = Mock()
        mock_db_instance.get_cache_key_data.return_value = "test_hash"
        mock_db_instance.load_books_data.return_value = sample_books_data
        mock_db_manager.return_value = mock_db_instance
        
        with patch.object(ProductionBookRecommender, '_save_to_cache'):
            recommender = ProductionBookRecommender(mock_config)
        
        assert recommender._is_ready is True
        mock_db_instance.load_books_data.assert_called_once()
    
    def test_find_book_by_title_exact_match(self, mock_config):
        """Test finding book by exact title match."""
        with patch.object(ProductionBookRecommender, '_initialize_model'):
            recommender = ProductionBookRecommender(mock_config)
            recommender._is_ready = True
            recommender.book_title_to_index = {'the great gatsby': 0}
            recommender.index_to_book = {
                0: {
                    'id': 1,
                    'title': 'The Great Gatsby',
                    'authors': 'F. Scott Fitzgerald'
                }
            }
        
        book = recommender.find_book_by_title('The Great Gatsby')
        
        assert book is not None
        assert book['title'] == 'The Great Gatsby'
        assert book['authors'] == 'F. Scott Fitzgerald'
    
    def test_find_book_by_title_partial_match(self, mock_config):
        """Test finding book by partial title match."""
        with patch.object(ProductionBookRecommender, '_initialize_model'):
            recommender = ProductionBookRecommender(mock_config)
            recommender._is_ready = True
            recommender.book_title_to_index = {'the great gatsby': 0}
            recommender.index_to_book = {
                0: {
                    'id': 1,
                    'title': 'The Great Gatsby',
                    'authors': 'F. Scott Fitzgerald'
                }
            }
        
        book = recommender.find_book_by_title('Great Gatsby')
        
        assert book is not None
        assert book['title'] == 'The Great Gatsby'
    
    def test_find_book_not_found(self, mock_config):
        """Test book not found scenario."""
        with patch.object(ProductionBookRecommender, '_initialize_model'):
            recommender = ProductionBookRecommender(mock_config)
            recommender._is_ready = True
            recommender.book_title_to_index = {}
            recommender.index_to_book = {}
        
        book = recommender.find_book_by_title('Nonexistent Book')
        
        assert book is None
    
    def test_get_recommendations_model_not_ready(self, mock_config):
        """Test getting recommendations when model is not ready."""
        with patch.object(ProductionBookRecommender, '_initialize_model'):
            recommender = ProductionBookRecommender(mock_config)
            recommender._is_ready = False
        
        with pytest.raises(ModelNotReadyError):
            recommender.get_recommendations('The Great Gatsby')
    
    def test_get_recommendations_book_not_found(self, mock_config):
        """Test getting recommendations for non-existent book."""
        with patch.object(ProductionBookRecommender, '_initialize_model'):
            recommender = ProductionBookRecommender(mock_config)
            recommender._is_ready = True
            recommender.book_title_to_index = {}
            recommender.index_to_book = {}
            recommender.metadata = pd.DataFrame()
        
        with pytest.raises(BookNotFoundError):
            recommender.get_recommendations('Nonexistent Book')
    
    @patch('recommender.metrics_collector')
    def test_get_recommendations_success(self, mock_metrics, mock_config):
        """Test successful recommendation generation."""
        with patch.object(ProductionBookRecommender, '_initialize_model'):
            recommender = ProductionBookRecommender(mock_config)
            recommender._is_ready = True
            
            # Setup mock data
            recommender.book_title_to_index = {'the great gatsby': 0, 'pride and prejudice': 1}
            recommender.index_to_book = {
                0: {'id': 1, 'title': 'The Great Gatsby', 'authors': 'F. Scott Fitzgerald'},
                1: {'id': 2, 'title': 'Pride and Prejudice', 'authors': 'Jane Austen'}
            }
            recommender.metadata = pd.DataFrame({'title': ['The Great Gatsby', 'Pride and Prejudice']})
            
            # Mock TF-IDF matrix and similarity computation
            recommender.tfidf_matrix = csr_matrix(np.array([[1, 0], [0.8, 0.2]]))
            
            # Mock metrics context manager
            mock_request_metric = Mock()
            mock_metrics.track_request.return_value.__enter__ = Mock(return_value=mock_request_metric)
            mock_metrics.track_request.return_value.__exit__ = Mock(return_value=None)
        
        result = recommender.get_recommendations('The Great Gatsby', limit=1)
        
        assert result['found'] is True
        assert result['query'] == 'The Great Gatsby'
        assert result['source_book']['title'] == 'The Great Gatsby'
        assert len(result['recommendations']) <= 1
    
    def test_input_validation_invalid_title(self, mock_config):
        """Test input validation for invalid title."""
        with patch.object(ProductionBookRecommender, '_initialize_model'):
            recommender = ProductionBookRecommender(mock_config)
            recommender._is_ready = True
        
        with pytest.raises(InvalidInputError):
            recommender.get_recommendations('')
    
    def test_input_validation_invalid_limit(self, mock_config):
        """Test input validation for invalid limit."""
        with patch.object(ProductionBookRecommender, '_initialize_model'):
            recommender = ProductionBookRecommender(mock_config)
            recommender._is_ready = True
        
        with pytest.raises(InvalidInputError):
            recommender.get_recommendations('Valid Title', limit=-1)
    
    def test_health_check_healthy(self, mock_config):
        """Test health check when system is healthy."""
        with patch.object(ProductionBookRecommender, '_initialize_model'):
            recommender = ProductionBookRecommender(mock_config)
            recommender._is_ready = True
            recommender.metadata = pd.DataFrame({'id': [1, 2, 3]})
            recommender._last_loaded = 1234567890
        
        with patch.object(recommender.cache_manager, 'get_cache_stats', return_value={}):
            with patch('recommender.metrics_collector') as mock_metrics:
                mock_metrics.get_system_health.return_value = {'status': 'healthy'}
                mock_metrics.get_performance_stats.return_value = {'requests_count': 10}
                
                health = recommender.health_check()
        
        assert health['status'] == 'healthy'
        assert health['model_ready'] is True
        assert health['books_count'] == 3
    
    def test_reload_model(self, mock_config):
        """Test model reload functionality."""
        with patch.object(ProductionBookRecommender, '_initialize_model') as mock_init:
            recommender = ProductionBookRecommender(mock_config)
            
            # Setup initial state
            recommender._is_ready = True
            recommender.metadata = pd.DataFrame({'id': [1, 2]})
            
            # Mock cache clearing
            with patch.object(recommender.cache_manager, 'clear_cache'):
                result = recommender.reload_model()
        
        assert result['status'] == 'success'
        assert 'reload_time_seconds' in result
        mock_init.assert_called()  # Should be called twice (init + reload)
    
    def test_get_stats(self, mock_config):
        """Test getting comprehensive statistics."""
        with patch.object(ProductionBookRecommender, '_initialize_model'):
            recommender = ProductionBookRecommender(mock_config)
            recommender._is_ready = True
            recommender.metadata = pd.DataFrame({'id': [1, 2, 3]})
            recommender.tfidf_matrix = csr_matrix(np.array([[1, 0], [0, 1], [1, 1]]))
            recommender._last_loaded = 1234567890
        
        with patch.object(recommender.cache_manager, 'get_cache_stats', return_value={}):
            with patch('recommender.metrics_collector') as mock_metrics:
                mock_metrics.get_performance_stats.return_value = {'requests_count': 10}
                mock_metrics.get_popular_books.return_value = {'popular_books': []}
                mock_metrics.get_error_breakdown.return_value = {}
                
                stats = recommender.get_stats()
        
        assert stats['model_stats']['ready'] is True
        assert stats['model_stats']['books_count'] == 3
        assert stats['model_stats']['tfidf_features'] == 2
    
    def test_singleton_pattern(self):
        """Test that get_recommender returns the same instance."""
        with patch.object(ProductionBookRecommender, '_initialize_model'):
            recommender1 = get_recommender()
            recommender2 = get_recommender()
        
        assert recommender1 is recommender2


class TestInputValidation:
    """Test input validation functions."""
    
    def test_validate_book_title_valid(self):
        """Test validation of valid book titles."""
        from validators import InputValidator
        
        valid_titles = [
            "The Great Gatsby",
            "To Kill a Mockingbird",
            "Pride and Prejudice: A Novel",
            "The Lord of the Rings (Book 1)"
        ]
        
        for title in valid_titles:
            result = InputValidator.validate_book_title(title)
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_validate_book_title_invalid(self):
        """Test validation of invalid book titles."""
        from validators import InputValidator
        
        invalid_titles = [
            "",
            None,
            "   ",
            "a" * 501,  # Too long
        ]
        
        for title in invalid_titles:
            with pytest.raises(InvalidInputError):
                InputValidator.validate_book_title(title)
    
    def test_validate_limit_valid(self):
        """Test validation of valid limits."""
        from validators import InputValidator
        
        valid_limits = [1, 5, 10, 50, 100]
        
        for limit in valid_limits:
            result = InputValidator.validate_limit(limit)
            assert result == limit
    
    def test_validate_limit_invalid(self):
        """Test validation of invalid limits."""
        from validators import InputValidator
        
        invalid_limits = [0, -1, 101, "invalid", None, 1.5]
        
        for limit in invalid_limits:
            with pytest.raises(InvalidInputError):
                InputValidator.validate_limit(limit)


@pytest.fixture(autouse=True)
def reset_global_recommender():
    """Reset global recommender instance before each test."""
    import recommender
    recommender._recommender_instance = None
    yield
    recommender._recommender_instance = None