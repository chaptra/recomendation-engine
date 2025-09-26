"""Database connection management with pooling."""

import logging
from contextlib import contextmanager
from typing import Generator
from urllib.parse import urlparse, unquote

import pandas as pd
import psycopg2
from psycopg2 import pool

from config import DatabaseConfig
from exceptions import DatabaseConnectionError

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections with connection pooling."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._pool = None
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """Initialize the connection pool."""
        try:
            parsed = urlparse(self.config.url)
            if not all([parsed.hostname, parsed.username, parsed.password, parsed.path]):
                raise ValueError("Invalid database URL format")
            
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                1,  # minconn
                self.config.pool_size,  # maxconn
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path[1:],  # Remove leading '/'
                user=parsed.username,
                password=unquote(parsed.password),
                connect_timeout=self.config.pool_timeout
            )
            logger.info("Database connection pool initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise DatabaseConnectionError(f"Database pool initialization failed: {e}")
    
    @contextmanager
    def get_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """Get a database connection from the pool."""
        connection = None
        try:
            connection = self._pool.getconn()
            if connection is None:
                raise DatabaseConnectionError("Failed to get connection from pool")
            
            # Test the connection
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            
            yield connection
            
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Database connection error: {e}")
            raise DatabaseConnectionError(f"Database operation failed: {e}")
        
        finally:
            if connection:
                try:
                    self._pool.putconn(connection)
                except Exception as e:
                    logger.error(f"Failed to return connection to pool: {e}")
    
    def execute_query(self, query: str, params: tuple = None) -> pd.DataFrame:
        """Execute a query and return results as DataFrame."""
        try:
            with self.get_connection() as conn:
                return pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise DatabaseConnectionError(f"Query failed: {e}")
    
    def get_cache_key_data(self) -> str:
        """Get data for cache key generation."""
        query = """
        SELECT COUNT(*), COALESCE(MAX(updated_at), NOW()) as max_updated
        FROM books_with_details 
        WHERE deleted_at IS NULL
        """
        try:
            result = self.execute_query(query)
            return f"{result.iloc[0, 0]}_{result.iloc[0, 1]}"
        except Exception as e:
            logger.error(f"Failed to get cache key data: {e}")
            # Fallback to timestamp
            import time
            return f"fallback_{int(time.time())}"
    
    def load_books_data(self) -> pd.DataFrame:
        """Load all books data from database."""
        query = """
        SELECT 
            b.id, b.title, b.slug, b.cover_local_path, b.description, 
            b.average_rating, b.rating_count,
            string_agg(DISTINCT a.name, '|') as authors,
            string_agg(DISTINCT c.name, '|') as categories
        FROM books_with_details b
        LEFT JOIN book_authors ba ON b.id = ba.canonical_book_id
        LEFT JOIN authors a ON ba.author_id = a.id AND a.deleted_at IS NULL
        LEFT JOIN book_categories bc ON b.id = bc.canonical_book_id
        LEFT JOIN categories c ON bc.category_id = c.id AND c.deleted_at IS NULL
        WHERE b.deleted_at IS NULL 
        AND b.description IS NOT NULL 
        AND b.description != ''
        AND b.title IS NOT NULL
        AND b.language = 'en'
        GROUP BY b.id, b.title, b.slug, b.cover_local_path, b.description, 
                 b.average_rating, b.rating_count
        ORDER BY b.id
        """
        
        logger.info("Loading books data from database...")
        try:
            return self.execute_query(query)
        except Exception as e:
            logger.error(f"Failed to load books data: {e}")
            raise DatabaseConnectionError(f"Books data loading failed: {e}")
    
    def close(self) -> None:
        """Close all connections in the pool."""
        if self._pool:
            self._pool.closeall()
            logger.info("Database connection pool closed")