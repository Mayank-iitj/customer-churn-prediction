# Contributing to Customer Churn Prediction

Thank you for your interest in contributing to the Customer Churn Prediction project! This document provides guidelines for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Submitting Changes](#submitting-changes)
- [Coding Standards](#coding-standards)
- [Testing](#testing)

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/customer-churn-prediction.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`

## Development Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

2. **Set up pre-commit hooks** (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Making Changes

### Branch Naming Convention

- Feature: `feature/description`
- Bug fix: `fix/description`
- Documentation: `docs/description`
- Performance: `perf/description`

### Commit Message Guidelines

Follow the conventional commits specification:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Example: `feat: add new feature for batch prediction optimization`

## Submitting Changes

1. **Ensure all tests pass**:
   ```bash
   pytest
   ```

2. **Lint your code**:
   ```bash
   flake8 src/ --max-line-length=127
   ```

3. **Update documentation** if needed

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: description of your changes"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

## Coding Standards

### Python Style Guide

- Follow PEP 8 guidelines
- Maximum line length: 127 characters
- Use meaningful variable and function names
- Add docstrings to all functions and classes

### Code Structure

```python
"""
Module docstring explaining the purpose.
"""

import standard_library
import third_party_library

from local_module import something


class MyClass:
    """
    Class docstring.
    
    Attributes:
        attribute1 (type): Description
    """
    
    def __init__(self):
        """Initialize the class."""
        pass
    
    def my_method(self, param1, param2):
        """
        Method description.
        
        Parameters:
        -----------
        param1 : type
            Description
        param2 : type
            Description
            
        Returns:
        --------
        type : Description
        """
        pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use descriptive test names

Example:
```python
def test_preprocessing_handles_missing_values():
    """Test that preprocessing correctly handles missing values."""
    # Test implementation
    pass
```

## Areas for Contribution

We welcome contributions in these areas:

1. **New Features**:
   - Additional ML algorithms
   - Advanced feature engineering
   - API endpoints for predictions
   - Real-time data pipeline

2. **Improvements**:
   - Performance optimization
   - Better error handling
   - Enhanced visualizations
   - UI/UX improvements

3. **Documentation**:
   - Tutorials and examples
   - API documentation
   - Deployment guides

4. **Testing**:
   - Unit tests
   - Integration tests
   - Performance tests

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about the codebase
- Suggestions for improvements

Thank you for contributing! 🎉
