# Contributing to INS-GPS Fusion Lab

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker to report bugs or suggest features
- Provide a clear description, steps to reproduce (for bugs), and your environment (OS, Python version)

### Submitting Pull Requests

1. Fork the repository and create a branch from `main`
2. Make your changes with clear, descriptive commit messages
3. Ensure all tests pass: `pytest tests/`
4. Add or update tests for new functionality
5. Submit a PR with a description of the changes

### Code Style

- Python: follow PEP 8
- Use English for all docstrings, comments, and variable/function names
- Add docstrings to public functions and classes

### Development Setup

```bash
pip install -e ".[dev]"
pytest tests/
```
