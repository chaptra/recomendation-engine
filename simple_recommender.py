import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import pickle
import logging
import os
import psycopg2
from urllib.parse import urlparse, unquote
from dotenv import load_dotenv
import hashlib
import time
from scipy.sparse import save_npz, load_npz
from pathlib import Path

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleBookRecommender:
    def __init__(self, cache_dir="./cache"):
        self.metadata = None
        self.similarity_matrix = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.book_title_to_index = {}
        self.index_to_book = {}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load and prepare data
        self._load_and_prepare_data()
        
    def _connect_to_db(self):
        db_url = os.getenv('RDS_DB_URL')
        parsed = urlparse(db_url)
        return psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port,
            database=parsed.path[1:],
            user=parsed.username,
            password=unquote(parsed.password)
        )
    
    def _get_cache_key(self):
        """Generate cache key based on database content hash"""
        conn = self._connect_to_db()
        # Get a hash of the book count and latest update time
        hash_query = "SELECT COUNT(*), MAX(updated_at) FROM books_with_details WHERE deleted_at IS NULL"
        result = pd.read_sql_query(hash_query, conn)
        conn.close()
        
        cache_string = f"{result.iloc[0, 0]}_{result.iloc[0, 1]}"
        return hashlib.md5(cache_string.encode()).hexdigest()[:8]
    
    def _load_and_prepare_data(self):
        logger.info("Loading and preparing book data...")
        
        # Check cache first
        cache_key = self._get_cache_key()
        metadata_cache = self.cache_dir / f"metadata_{cache_key}.pkl"
        tfidf_cache = self.cache_dir / f"tfidf_{cache_key}.npz"
        vectorizer_cache = self.cache_dir / f"vectorizer_{cache_key}.pkl"
        mappings_cache = self.cache_dir / f"mappings_{cache_key}.pkl"
        
        if all(f.exists() for f in [metadata_cache, tfidf_cache, vectorizer_cache, mappings_cache]):
            logger.info("Loading from cache...")
            self.metadata = pd.read_pickle(metadata_cache)
            self.tfidf_matrix = load_npz(tfidf_cache)
            with open(vectorizer_cache, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            with open(mappings_cache, 'rb') as f:
                mappings = pickle.load(f)
                self.book_title_to_index = mappings['title_to_index']
                self.index_to_book = mappings['index_to_book']
            logger.info(f"Loaded {len(self.metadata)} books from cache!")
            return
        
        # Load books from database
        start_time = time.time()
        conn = self._connect_to_db()
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
        """
        
        self.metadata = pd.read_sql_query(query, conn)
        conn.close()
        logger.info(f"Loaded {len(self.metadata)} books in {time.time() - start_time:.2f}s")
        
        # Clean and prepare text more efficiently
        logger.info("Preparing text features...")
        self.metadata['clean_text'] = (
            self.metadata['title'].fillna('') + ' ' + 
            self.metadata['description'].fillna('').str[:500] + ' ' +  # Limit description length
            self.metadata['authors'].fillna('') + ' ' +
            self.metadata['categories'].fillna('')
        ).str.lower().str.replace(r'[^a-zA-Z\s]', ' ', regex=True)
        
        # Create book title mapping
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
                'description': str(row['description'])[:300] + '...',
                'rating': float(row['average_rating']),
                'rating_count': int(row['rating_count'])
            }
        
        # Compute TF-IDF matrix and cache everything
        self._compute_tfidf()
        
        # Save to cache
        logger.info("Saving to cache...")
        self.metadata.to_pickle(metadata_cache)
        save_npz(tfidf_cache, self.tfidf_matrix)
        with open(vectorizer_cache, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        with open(mappings_cache, 'wb') as f:
            pickle.dump({
                'title_to_index': self.book_title_to_index,
                'index_to_book': self.index_to_book
            }, f)
        
        logger.info(f"Loaded {len(self.metadata)} books successfully!")
    
    def _compute_tfidf(self):
        logger.info("Computing TF-IDF matrix...")
        
        # Optimized TF-IDF parameters for large datasets
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,  # Increased for better accuracy
            stop_words='english',
            ngram_range=(1, 2),
            min_df=3,  # Slightly higher to reduce noise
            max_df=0.7,
            dtype=np.float32  # Use float32 to save memory
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.metadata['clean_text'])
        logger.info(f"TF-IDF matrix computed: {self.tfidf_matrix.shape}")
    
    def find_book_by_title(self, title):
        """Find book by exact or partial title match"""
        title_lower = title.lower().strip()
        
        # Exact match first
        if title_lower in self.book_title_to_index:
            idx = self.book_title_to_index[title_lower]
            return self.index_to_book[idx]
        
        # Partial match
        for book_title, idx in self.book_title_to_index.items():
            if title_lower in book_title or book_title in title_lower:
                return self.index_to_book[idx]
        
        return None
    
    def _is_duplicate_book(self, book1, book2):
        """Check if two books are duplicates based on title and author similarity"""
        title1 = book1['title'].lower().strip()
        title2 = book2['title'].lower().strip()
        author1 = book1['authors'].lower().strip()
        author2 = book2['authors'].lower().strip()
        
        # Remove common punctuation and normalize
        import re
        title1 = re.sub(r'[^\w\s]', '', title1)
        title2 = re.sub(r'[^\w\s]', '', title2)
        
        # Check for very similar titles with same author
        if author1 == author2:
            # Same author - check title similarity
            if title1 == title2:
                return True
            # Check if one title contains the other (like "Moby Dick" vs "Moby-Dick; or, The Whale")
            if title1 in title2 or title2 in title1:
                return True
            # Check for similar titles (removing common words)
            title1_words = set(title1.split())
            title2_words = set(title2.split())
            common_words = title1_words.intersection(title2_words)
            if len(common_words) >= 2 and (len(common_words) / len(title1_words) > 0.7 or len(common_words) / len(title2_words) > 0.7):
                return True
        
        return False

    def get_recommendations(self, book_title, limit=10):
        """Get book recommendations based on title using on-demand similarity computation"""
        
        # Find the source book
        source_book = self.find_book_by_title(book_title)
        if not source_book:
            # Try searching in title contains
            matching_books = []
            title_lower = book_title.lower()
            for idx, row in self.metadata.iterrows():
                if title_lower in row['title'].lower():
                    matching_books.append(self.index_to_book[idx])
            
            if not matching_books:
                return {
                    'query': book_title,
                    'found': False,
                    'message': f'No books found matching "{book_title}"',
                    'suggestions': []
                }
            
            source_book = matching_books[0]  # Use first match
        
        # Get source book index
        source_idx = None
        for idx, book in self.index_to_book.items():
            if book['id'] == source_book['id']:  # Use ID for exact match
                source_idx = idx
                break
        
        if source_idx is None:
            return {
                'query': book_title,
                'found': False,
                'message': 'Book index not found',
                'suggestions': []
            }
        
        # Compute similarity only for the source book (much faster!)
        source_vector = self.tfidf_matrix[source_idx]
        similarities = cosine_similarity(source_vector, self.tfidf_matrix).flatten()
        
        # Get top similarities (excluding source book)
        sim_scores = [(i, similarities[i]) for i in range(len(similarities)) if i != source_idx]
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get recommendations with duplicate filtering
        recommendations = []
        added_books = set()  # Track added book IDs
        
        for idx, score in sim_scores:
            candidate_book = self.index_to_book[idx].copy()
            candidate_book['similarity_score'] = round(float(score), 3)
            candidate_book['reason'] = self._get_reason(score)
            
            # Skip if this book ID is already added
            if candidate_book['id'] in added_books:
                continue
            
            # Skip if this is a duplicate of the source book
            if self._is_duplicate_book(candidate_book, source_book):
                continue
            
            # Skip if this is a duplicate of any already added book
            is_duplicate = False
            for existing_book in recommendations:
                if self._is_duplicate_book(candidate_book, existing_book):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                recommendations.append(candidate_book)
                added_books.add(candidate_book['id'])
                
                # Stop when we have enough recommendations
                if len(recommendations) >= limit:
                    break
        
        return {
            'query': book_title,
            'found': True,
            'source_book': source_book,
            'suggestions': recommendations,
            'total': len(recommendations)
        }
    
    def _get_reason(self, score):
        """Generate recommendation reason based on similarity score"""
        if score > 0.7:
            return "Very similar content and themes"
        elif score > 0.5:
            return "Similar themes and style"
        elif score > 0.3:
            return "Shared categories and topics"
        else:
            return "Related content"

# Global recommender instance
recommender = None

def get_recommender():
    global recommender
    if recommender is None:
        recommender = SimpleBookRecommender()
    return recommender

if __name__ == "__main__":
    # Test the recommender with timing
    import time
    
    print("Initializing recommender...")
    start_time = time.time()
    rec = SimpleBookRecommender()
    init_time = time.time() - start_time
    print(f"Initialization took: {init_time:.2f}s")
    
    # Test with Moby Dick
    print("\nGetting recommendations...")
    start_time = time.time()
    result = rec.get_recommendations("Moby Dick", limit=5)
    rec_time = time.time() - start_time
    print(f"Recommendations took: {rec_time:.2f}s")
    
    print(f"\nQuery: {result['query']}")
    print(f"Found: {result['found']}")
    
    if result['found']:
        print(f"Source: {result['source_book']['title']} by {result['source_book']['authors']}")
        print(f"\nRecommendations ({result['total']}):")
        for i, book in enumerate(result['suggestions'], 1):
            print(f"{i}. {book['title']}")
            print(f"   Author: {book['authors']}")
            print(f"   Similarity: {book['similarity_score']}")
            print(f"   Reason: {book['reason']}")
            print()