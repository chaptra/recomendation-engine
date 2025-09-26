# Contributing to Book Recommendation Engine

Thank you for your interest in contributing to the Book Recommendation Engine! We welcome contributions from the community and are pleased to have you join us.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Code Style](#code-style)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)

## 🤝 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- PostgreSQL database
- Git
- Basic knowledge of machine learning and web APIs

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/yourusername/book-recommendation-engine.git
cd book-recommendation-engine
```

3. Add the original repository as a remote:
```bash
git remote add upstream https://github.com/originalowner/book-recommendation-engine.git
```

## 🛠 Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 3. Set Up Pre-commit Hooks

```bash
pre-commit install
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your database configuration
```

### 5. Run Tests

```bash
pytest
```

## 🔧 Making Changes

### Branch Naming Convention

Create a descriptive branch name:
- `feature/add-collaborative-filtering`
- `bugfix/fix-cache-invalidation`
- `docs/update-api-documentation`
- `refactor/improve-error-handling`

### Development Workflow

1. **Create a new branch**:
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**:
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Commit your changes**:
```bash
git add .
git commit -m "Add: description of your changes"
```

4. **Push to your fork**:
```bash
git push origin feature/your-feature-name
```

## ✅ Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_recommender.py

# Run tests with verbose output
pytest -v
```

### Writing Tests

- Add tests for all new functionality
- Ensure tests are isolated and repeatable
- Use descriptive test names
- Mock external dependencies (database, cache)

Example test:
```python
def test_get_recommendations_success():
    """Test successful recommendation generation."""
    recommender = ProductionBookRecommender(mock_config)
    result = recommender.get_recommendations("The Great Gatsby", limit=5)
    
    assert result['found'] is True
    assert len(result['recommendations']) <= 5
    assert all('similarity_score' in rec for rec in result['recommendations'])
```

## 📝 Submitting Changes

### Pull Request Process

1. **Update your branch**:
```bash
git fetch upstream
git rebase upstream/main
```

2. **Ensure tests pass**:
```bash
pytest
```

3. **Create Pull Request**:
   - Use a clear, descriptive title
   - Include a detailed description of changes
   - Reference any related issues
   - Add screenshots if UI changes are involved

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## 🎨 Code Style

### Python Style Guide

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://github.com/psf/black) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use [mypy](http://mypy-lang.org/) for type checking

### Formatting Commands

```bash
# Format code
black .

# Sort imports
isort .

# Type checking
mypy .

# Lint code
flake8 .
```

### Code Quality Guidelines

- **Functions**: Keep functions small and focused
- **Variables**: Use descriptive names
- **Comments**: Explain why, not what
- **Type Hints**: Add type hints to all functions
- **Docstrings**: Use Google-style docstrings

Example:
```python
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
        BookNotFoundError: If book is not found in database
        InvalidInputError: If inputs are invalid
    """
```

## 📚 Documentation

### Documentation Standards

- Update README.md for significant changes
- Add docstrings to all public functions and classes
- Update API documentation for endpoint changes
- Include examples in documentation

### Building Documentation

```bash
# Generate API docs
sphinx-build -b html docs/ docs/_build/html
```

## 🐛 Issue Reporting

### Bug Reports

Include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error messages and stack traces

### Bug Report Template

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment:**
- OS: [e.g. Ubuntu 20.04]
- Python version: [e.g. 3.9.7]
- Package version: [e.g. 1.0.0]

**Additional context**
Add any other context about the problem here.
```

## 💡 Feature Requests

### Feature Request Process

1. Check existing issues to avoid duplicates
2. Create detailed feature request with:
   - Use case description
   - Proposed solution
   - Alternative solutions considered
   - Implementation complexity estimate

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Any other context, mockups, or examples.
```

## 🏷 Labels and Milestones

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested
- `wontfix`: This will not be worked on

### Priority Labels

- `priority: high`: Critical issues
- `priority: medium`: Important issues
- `priority: low`: Nice to have

## 🎯 Contribution Areas

### High Priority Areas

1. **Performance Optimization**
   - Caching improvements
   - Query optimization
   - Memory usage reduction

2. **Algorithm Enhancements**
   - Collaborative filtering
   - Neural network recommendations
   - Hybrid approaches

3. **API Improvements**
   - Rate limiting enhancements
   - Better error handling
   - API versioning

4. **Testing & Documentation**
   - Integration tests
   - Performance tests
   - API documentation

### Good First Issues

- Documentation improvements
- Code formatting fixes
- Simple bug fixes
- Test coverage improvements

## 🔍 Review Process

### Review Criteria

- Code quality and style
- Test coverage
- Documentation completeness
- Performance impact
- Security considerations
- Backward compatibility

### Getting Reviews

- Tag relevant maintainers
- Respond to feedback promptly
- Make requested changes
- Keep discussions focused and professional

## 🎉 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Invited to join the contributors team (for significant contributions)

## 📞 Getting Help

- **GitHub Discussions**: For questions and general discussion
- **Issues**: For bug reports and feature requests
- **Discord/Slack**: [Community chat link if available]

## 📋 Checklist for Contributors

Before submitting:
- [ ] Code follows style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main
- [ ] No merge conflicts
- [ ] PR description is complete

Thank you for contributing to the Book Recommendation Engine! 🚀