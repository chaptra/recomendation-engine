"""Intelligent cache management for the recommendation engine."""

import hashlib
import logging
import pickle
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz

from config import CacheConfig
from exceptions import CacheError

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching for TF-IDF matrices, metadata, and model artifacts."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = Path(config.directory)
        self.cache_dir.mkdir(exist_ok=True)
        self._ensure_cache_size()
    
    def _ensure_cache_size(self) -> None:
        """Ensure cache directory doesn't exceed size limit."""
        try:
            cache_size_gb = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file()) / (1024**3)
            if cache_size_gb > self.config.max_size_gb:
                logger.warning(f"Cache size {cache_size_gb:.2f}GB exceeds limit {self.config.max_size_gb}GB")
                self._cleanup_old_cache()
        except Exception as e:
            logger.error(f"Cache size check failed: {e}")
    
    def _cleanup_old_cache(self) -> None:
        """Remove old cache files to free up space."""
        try:
            cache_files = list(self.cache_dir.rglob('*'))
            cache_files.sort(key=lambda f: f.stat().st_mtime)  # Sort by modification time
            
            # Remove oldest 30% of files
            files_to_remove = cache_files[:len(cache_files) // 3]
            for file in files_to_remove:
                if file.is_file():
                    file.unlink()
                    logger.debug(f"Removed old cache file: {file}")
                    
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
    
    def generate_cache_key(self, data_hash: str, config_hash: str) -> str:
        """Generate a cache key based on data and configuration."""
        combined = f"{data_hash}_{config_hash}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]
    
    def get_config_hash(self, config_dict: Dict[str, Any]) -> str:
        """Generate hash for configuration parameters."""
        config_str = str(sorted(config_dict.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _get_cache_paths(self, cache_key: str) -> Dict[str, Path]:
        """Get all cache file paths for a given key."""
        return {
            'metadata': self.cache_dir / f"metadata_{cache_key}.pkl",
            'tfidf_matrix': self.cache_dir / f"tfidf_{cache_key}.npz",
            'vectorizer': self.cache_dir / f"vectorizer_{cache_key}.pkl",
            'mappings': self.cache_dir / f"mappings_{cache_key}.pkl",
            'timestamp': self.cache_dir / f"timestamp_{cache_key}.txt"
        }
    
    def _is_cache_valid(self, timestamp_file: Path) -> bool:
        """Check if cache is still valid based on TTL."""
        try:
            if not timestamp_file.exists():
                return False
            
            cache_time = float(timestamp_file.read_text())
            ttl_seconds = self.config.ttl_hours * 3600
            return (time.time() - cache_time) < ttl_seconds
            
        except Exception as e:
            logger.error(f"Cache validation failed: {e}")
            return False
    
    def cache_exists(self, cache_key: str) -> bool:
        """Check if valid cache exists for the given key."""
        try:
            paths = self._get_cache_paths(cache_key)
            
            # Check if all required files exist
            required_files = ['metadata', 'tfidf_matrix', 'vectorizer', 'mappings', 'timestamp']
            if not all(paths[f].exists() for f in required_files):
                return False
            
            # Check if cache is still valid
            return self._is_cache_valid(paths['timestamp'])
            
        except Exception as e:
            logger.error(f"Cache existence check failed: {e}")
            return False
    
    def save_cache(
        self,
        cache_key: str,
        metadata: pd.DataFrame,
        tfidf_matrix: csr_matrix,
        vectorizer: Any,
        book_mappings: Dict[str, Any]
    ) -> None:
        """Save all model artifacts to cache."""
        try:
            paths = self._get_cache_paths(cache_key)
            
            # Save metadata
            metadata.to_pickle(paths['metadata'])
            
            # Save TF-IDF matrix
            save_npz(paths['tfidf_matrix'], tfidf_matrix)
            
            # Save vectorizer
            with open(paths['vectorizer'], 'wb') as f:
                pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save mappings
            with open(paths['mappings'], 'wb') as f:
                pickle.dump(book_mappings, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save timestamp
            paths['timestamp'].write_text(str(time.time()))
            
            logger.info(f"Cache saved successfully with key: {cache_key}")
            
        except Exception as e:
            logger.error(f"Cache save failed: {e}")
            # Clean up partial cache files
            self._cleanup_partial_cache(cache_key)
            raise CacheError(f"Failed to save cache: {e}")
    
    def load_cache(self, cache_key: str) -> Tuple[pd.DataFrame, csr_matrix, Any, Dict[str, Any]]:
        """Load all model artifacts from cache."""
        try:
            if not self.cache_exists(cache_key):
                raise CacheError("Cache does not exist or is invalid")
            
            paths = self._get_cache_paths(cache_key)
            
            # Load metadata
            metadata = pd.read_pickle(paths['metadata'])
            
            # Load TF-IDF matrix
            tfidf_matrix = load_npz(paths['tfidf_matrix'])
            
            # Load vectorizer
            with open(paths['vectorizer'], 'rb') as f:
                vectorizer = pickle.load(f)
            
            # Load mappings
            with open(paths['mappings'], 'rb') as f:
                mappings = pickle.load(f)
            
            logger.info(f"Cache loaded successfully with key: {cache_key}")
            return metadata, tfidf_matrix, vectorizer, mappings
            
        except Exception as e:
            logger.error(f"Cache load failed: {e}")
            raise CacheError(f"Failed to load cache: {e}")
    
    def _cleanup_partial_cache(self, cache_key: str) -> None:
        """Remove partially created cache files."""
        try:
            paths = self._get_cache_paths(cache_key)
            for path in paths.values():
                if path.exists():
                    path.unlink()
                    logger.debug(f"Cleaned up partial cache file: {path}")
        except Exception as e:
            logger.error(f"Partial cache cleanup failed: {e}")
    
    def clear_cache(self, pattern: Optional[str] = None) -> None:
        """Clear cache files matching pattern (or all if None)."""
        try:
            if pattern:
                files_to_remove = list(self.cache_dir.glob(f"*{pattern}*"))
            else:
                files_to_remove = list(self.cache_dir.rglob('*'))
            
            for file in files_to_remove:
                if file.is_file():
                    file.unlink()
                    
            logger.info(f"Cache cleared: {len(files_to_remove)} files removed")
            
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            raise CacheError(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            cache_files = list(self.cache_dir.rglob('*'))
            total_size = sum(f.stat().st_size for f in cache_files if f.is_file())
            
            return {
                'cache_dir': str(self.cache_dir),
                'total_files': len([f for f in cache_files if f.is_file()]),
                'total_size_mb': total_size / (1024**2),
                'max_size_gb': self.config.max_size_gb,
                'ttl_hours': self.config.ttl_hours
            }
        except Exception as e:
            logger.error(f"Cache stats failed: {e}")
            return {'error': str(e)}