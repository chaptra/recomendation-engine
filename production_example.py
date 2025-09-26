"""
Example usage of the production book recommendation engine.

This demonstrates how to use the refactored, industry-standard recommendation system.
"""

import asyncio
import time
import logging
from typing import Dict, Any

from recommender import ProductionBookRecommender
from config import load_config
from exceptions import BookNotFoundError, ModelNotReadyError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_recommendations():
    """Demonstrate the production recommendation system."""
    print("🚀 Production Book Recommendation Engine Demo")
    print("=" * 50)
    
    try:
        # Initialize the recommender
        print("\n📚 Initializing recommendation engine...")
        start_time = time.time()
        
        config = load_config()
        recommender = ProductionBookRecommender(config)
        
        init_time = time.time() - start_time
        print(f"✅ Engine initialized in {init_time:.2f} seconds")
        print(f"📊 Loaded {len(recommender.metadata)} books")
        
        # Health check
        print("\n🔍 Performing health check...")
        health = recommender.health_check()
        print(f"Status: {health['status']}")
        print(f"Model Ready: {health['model_ready']}")
        print(f"Books Count: {health['books_count']}")
        
        # Test recommendations
        test_books = [
            "Moby Dick",
            "The Great Gatsby", 
            "1984",
            "Pride and Prejudice"
        ]
        
        print("\n📖 Getting recommendations for test books...")
        print("-" * 30)
        
        for book_title in test_books:
            try:
                print(f"\n🎯 Recommendations for: '{book_title}'")
                
                start_time = time.time()
                result = recommender.get_recommendations(
                    book_title=book_title,
                    limit=5,
                    similarity_threshold=0.3
                )
                
                request_time = time.time() - start_time
                
                if result['found']:
                    print(f"✅ Found source book: {result['source_book']['title']}")
                    print(f"⚡ Computed in {result['computation_time_ms']:.1f}ms")
                    print(f"🔗 {result['total']} recommendations:")
                    
                    for i, rec in enumerate(result['recommendations'], 1):
                        print(f"  {i}. {rec['title']}")
                        print(f"     📝 Author: {rec['authors']}")
                        print(f"     🎯 Similarity: {rec['similarity_score']:.3f}")
                        print(f"     💡 Reason: {rec['reason']}")
                        print()
                else:
                    print(f"❌ Book not found: {result['message']}")
                    
            except BookNotFoundError as e:
                print(f"❌ Book not found: {e}")
            except Exception as e:
                print(f"💥 Error: {e}")
        
        # Performance statistics
        print("\n📊 Performance Statistics")
        print("-" * 30)
        stats = recommender.get_stats()
        
        perf_stats = stats['performance_stats']
        print(f"Total Requests: {perf_stats.get('requests_count', 0)}")
        print(f"Success Rate: {perf_stats.get('success_rate', 0):.1f}%")
        print(f"Avg Response Time: {perf_stats.get('avg_response_time_ms', 0):.1f}ms")
        print(f"P95 Response Time: {perf_stats.get('p95_response_time_ms', 0):.1f}ms")
        print(f"Cache Hit Rate: {perf_stats.get('cache_hit_rate', 0):.1f}%")
        
        # Cache statistics
        cache_stats = stats['cache_stats']
        print(f"\n💾 Cache Statistics")
        print(f"Total Files: {cache_stats.get('total_files', 0)}")
        print(f"Total Size: {cache_stats.get('total_size_mb', 0):.1f} MB")
        print(f"Max Size: {cache_stats.get('max_size_gb', 0)} GB")
        
        # Test error handling
        print("\n🔧 Testing error handling...")
        try:
            recommender.get_recommendations("Nonexistent Book Title", limit=5)
        except BookNotFoundError:
            print("✅ BookNotFoundError handled correctly")
        
        try:
            recommender.get_recommendations("", limit=5)
        except Exception:
            print("✅ Invalid input handled correctly")
        
        print("\n🎉 Demo completed successfully!")
        
    except ModelNotReadyError as e:
        print(f"💥 Model initialization failed: {e}")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        logger.exception("Demo failed")


def benchmark_performance():
    """Benchmark the recommendation system performance."""
    print("\n⚡ Performance Benchmark")
    print("=" * 30)
    
    try:
        recommender = ProductionBookRecommender()
        
        # Benchmark parameters
        test_queries = [
            "Moby Dick", "1984", "Pride and Prejudice", 
            "The Great Gatsby", "To Kill a Mockingbird"
        ] * 4  # 20 total queries
        
        response_times = []
        successful_requests = 0
        
        print(f"🏃 Running {len(test_queries)} recommendation requests...")
        
        start_benchmark = time.time()
        
        for i, query in enumerate(test_queries, 1):
            try:
                start_time = time.time()
                result = recommender.get_recommendations(query, limit=10)
                end_time = time.time()
                
                if result['found']:
                    response_time = (end_time - start_time) * 1000
                    response_times.append(response_time)
                    successful_requests += 1
                    
                    if i % 5 == 0:
                        print(f"  ✅ Completed {i}/{len(test_queries)} requests")
                
            except Exception as e:
                print(f"  ❌ Request {i} failed: {e}")
        
        total_benchmark_time = time.time() - start_benchmark
        
        # Calculate statistics
        if response_times:
            import numpy as np
            avg_response = np.mean(response_times)
            p95_response = np.percentile(response_times, 95)
            p99_response = np.percentile(response_times, 99)
            min_response = min(response_times)
            max_response = max(response_times)
            
            print(f"\n📈 Benchmark Results:")
            print(f"Total Requests: {len(test_queries)}")
            print(f"Successful: {successful_requests}")
            print(f"Success Rate: {(successful_requests/len(test_queries))*100:.1f}%")
            print(f"Total Time: {total_benchmark_time:.2f}s")
            print(f"Requests/Second: {len(test_queries)/total_benchmark_time:.1f}")
            print(f"\n⏱️  Response Times:")
            print(f"Average: {avg_response:.1f}ms")
            print(f"P95: {p95_response:.1f}ms") 
            print(f"P99: {p99_response:.1f}ms")
            print(f"Min: {min_response:.1f}ms")
            print(f"Max: {max_response:.1f}ms")
            
            # Performance assessment
            if avg_response < 100:
                print("🚀 Excellent performance!")
            elif avg_response < 500:
                print("👍 Good performance")
            else:
                print("⚠️  Performance could be improved")
        
    except Exception as e:
        print(f"💥 Benchmark failed: {e}")
        logger.exception("Benchmark failed")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_recommendations())
    
    # Run performance benchmark
    benchmark_performance()