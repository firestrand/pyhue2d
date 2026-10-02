# Justfile for pyhue2d

# Default recipe: run all checks and tests
default: check test

# Sync virtual environment and dependencies using uv
sync:
    uv sync

# Run linting and formatting checks
lint:
    uv run ruff check .
    uv run ruff format --check .

# Automatically fix linting and format code
format:
    uv run ruff check --fix .
    uv run ruff format .

# Run static type checking with ty
typecheck:
    uv run ty check

# Run all checks (lint + typecheck)
check: lint typecheck

# Run test suite
test *args:
    uv run pytest {{args}}

# Full verification recipe
verify: check examples-check
    uv run python scripts/check_jabcode_fixtures.py
    uv run python scripts/check_docs.py
    uv run pytest -v --cov=pyhue2d --cov-branch --cov-report=xml

# Run test suite with coverage
test-cov:
    uv run pytest -v --cov=pyhue2d --cov-branch --cov-report=xml


# Run all examples and generate repository sample assets
examples:
    uv run python examples/generate_all.py

# Verify sample assets without rewriting tracked files
examples-check:
    uv run python examples/generate_all.py --check

# Build package distributions using uv
build:
    uv build

# Clean temporary files and build artifacts
clean:
    rm -rf build/ dist/ *.egg-info src/*.egg-info .ruff_cache/ .pytest_cache/ coverage.xml .coverage
