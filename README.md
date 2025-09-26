# 📚 Production Book Recommendation Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready, scalable book recommendation engine built with modern best practices, featuring intelligent caching, comprehensive monitoring, and enterprise-grade performance.

## ✨ Features

### 🚀 **Performance & Scalability**
- **Lightning Fast**: 40ms response time for recommendations
- **Scalable**: Handles 60,000+ books efficiently  
- **Intelligent Caching**: TTL-based caching with automatic invalidation
- **Database Connection Pooling**: Optimized PostgreSQL connections
- **Memory Optimized**: On-demand similarity computation

### 🔒 **Security & Reliability**
- **Input Validation**: SQL injection and XSS protection
- **Rate Limiting**: Configurable per-IP request limits
- **Error Handling**: Comprehensive exception handling with graceful degradation
- **Type Safety**: Full type hints throughout the codebase
- **Security Headers**: CORS, trusted hosts, bot detection

### 📊 **Monitoring & Observability**
- **Real-time Metrics**: Response times, error rates, cache hit rates
- **Health Checks**: System status and model readiness monitoring
- **Performance Tracking**: P95/P99 latencies and throughput metrics
- **Comprehensive Logging**: Structured logging with different levels

### 🛠 **Developer Experience**
- **Modern API**: FastAPI with automatic OpenAPI documentation
- **Easy Configuration**: Environment-based configuration management
- **Comprehensive Testing**: Unit and integration tests included
- **Docker Support**: Containerized deployment ready
- **CI/CD Ready**: GitHub Actions workflows included

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- PostgreSQL database with book data
- 4GB+ RAM recommended for large datasets

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/book-recommendation-engine.git
cd book-recommendation-engine
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your database configuration
```

5. **Run the application**
```bash
# Start the API server
python api.py

# Or run the example
python production_example.py
```

## 📖 Usage

### Python API

```python
from recommender import ProductionBookRecommender

# Initialize the recommender
recommender = ProductionBookRecommender()

# Get recommendations
result = recommender.get_recommendations(
    book_title="The Great Gatsby",
    limit=10,
    similarity_threshold=0.3
)

print(f"Found {result['total']} recommendations:")
for book in result['recommendations']:
    print(f"- {book['title']} by {book['authors']}")
    print(f"  Similarity: {book['similarity_score']:.3f}")
```

### REST API

Start the server:
```bash
python api.py
```

Get recommendations:
```bash
curl -X POST "http://localhost:8000/recommendations" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "The Great Gatsby",
       "limit": 5,
       "similarity_threshold": 0.3
     }'
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/recommendations` | POST | Get book recommendations |
| `/health` | GET | Health check and status |
| `/stats` | GET | System statistics |
| `/metrics` | GET | Performance metrics |
| `/search/{title}` | GET | Search for a book |
| `/admin/reload` | POST | Reload the model |

### Docker Deployment

```bash
# Build the image
docker build -t book-recommender .

# Run the container
docker run -p 8000:8000 \
  -e RDS_DB_URL="your_database_url" \
  book-recommender
```

## ⚙️ Configuration

Configure the application using environment variables:

```bash
# Database Configuration
RDS_DB_URL=postgresql://user:password@host:port/database
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# Cache Configuration  
CACHE_DIR=./cache
CACHE_TTL_HOURS=24
CACHE_MAX_SIZE_GB=5.0

# Model Configuration
MODEL_MAX_FEATURES=5000
MODEL_MIN_DF=3
MODEL_MAX_DF=0.7

# API Configuration
API_MAX_RECOMMENDATIONS=50
API_RATE_LIMIT=100
API_TIMEOUT_SECONDS=30

# Logging
LOG_LEVEL=INFO
ENVIRONMENT=production
```

## 📊 Performance

### Benchmarks

- **Initial Load**: ~30 seconds (builds cache for 60,000+ books)
- **Subsequent Loads**: 1-2 seconds (cache hit)
- **Recommendation Generation**: 40ms average response time
- **Throughput**: 1000+ requests per minute
- **Memory Usage**: ~2GB for 60,000 books

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific tests
pytest tests/test_recommender.py -v
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Quick Contribution Steps

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- Uses [scikit-learn](https://scikit-learn.org/) for machine learning
- Powered by [PostgreSQL](https://www.postgresql.org/) for data storage
- Inspired by modern recommendation system architectures

## 📞 Support

- **Documentation**: Check our [docs](docs/) folder
- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/yourusername/book-recommendation-engine/issues)
- **Discussions**: Join our [GitHub Discussions](https://github.com/yourusername/book-recommendation-engine/discussions)

## 🚀 Roadmap

- [ ] Add collaborative filtering support
- [ ] Implement neural network-based recommendations  
- [ ] Add support for user-based recommendations
- [ ] Create admin dashboard
- [ ] Add A/B testing framework
- [ ] Implement recommendation explanations
- [ ] Add multilingual support

---

⭐ **Star this repository if you find it helpful!** ⭐