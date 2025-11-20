# Contributing to Uniform-K Compression

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/uniform-k-compression.git
   cd uniform-k-compression
   ```

2. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

   This installs the package in editable mode with development dependencies (pytest, black, ruff).

3. **Set up API key** (for testing with actual LLM calls)
   ```bash
   export ANTHROPIC_API_KEY="your-key-here"
   ```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_document.py -v

# Run with coverage (if installed)
pytest tests/ --cov=core --cov=utilities
```

## Code Style

This project uses:
- **black** for code formatting (line length: 100)
- **ruff** for linting

Before submitting a PR:

```bash
# Format code
black .

# Check for issues
ruff check .
```

## Project Structure

```
uniform-k-compression/
├── core/              # Core algorithm implementation
│   ├── document.py           # Document data model
│   ├── k_calibrator.py       # Bootstrap K selection
│   ├── layer_executor.py     # Layer execution logic
│   ├── llm_interface.py      # Anthropic API wrapper
│   ├── batch_interface.py    # Batch API support
│   ├── fractal_summarizer.py # Main orchestration
│   └── config.py             # Configuration classes
├── utilities/         # Optional preprocessing tools
│   ├── document_loader.py
│   ├── cooperative_chunker.py
│   └── ...
├── tests/             # Test suite
├── examples/          # Working examples
└── configs/           # Example configurations
```

## Adding Features

1. **Create an issue** describing the feature
2. **Fork and branch** from `main`
3. **Implement** with tests
4. **Run tests** and code formatting
5. **Submit PR** with clear description

### Guidelines

- All new code should include tests
- Maintain backwards compatibility when possible
- Update documentation (README, docstrings) as needed
- Follow existing code patterns and style

## Testing Philosophy

- **Unit tests** for individual components (document.py, config.py, etc.)
- **Integration tests** for multi-component workflows (use test fixtures)
- **No API calls in CI** - tests should run without ANTHROPIC_API_KEY
- Use mocks/fixtures for LLM responses in tests

## Common Tasks

### Adding a new configuration parameter

1. Update the appropriate config class in `core/config.py`
2. Add validation in `__post_init__`
3. Update example configs in `configs/`
4. Add tests in `tests/test_config.py`
5. Document in README.md

### Improving LLM interface

1. Core changes go in `core/llm_interface.py` or `core/batch_interface.py`
2. Add tests that don't require API calls (mock the responses)
3. Test with actual API manually before submitting
4. Update cost calculations if pricing changes

### Adding preprocessing utilities

1. Create new file in `utilities/`
2. Add to `utilities/__init__.py` exports
3. Add tests in `tests/test_*.py`
4. Add example usage in `examples/` if appropriate

## Questions?

Open an issue for:
- Feature requests
- Bug reports
- Documentation improvements
- General questions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
