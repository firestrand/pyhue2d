# CODING_GUIDELINES.md

These guidelines ensure code quality, maintainability, and extensibility for the PyHue2D project. All contributors must follow them.

---

## 1. General Principles
- **SOLID**: Apply SOLID object-oriented principles to code.
- **KISS**: Keep code and interfaces as simple as possible.
- **DRY**: Avoid code duplication; refactor shared logic into helpers or base classes.
- **Inventiveness**: Favor inventive, clean solutions—even if it means bending rules for clarity or safety.
- **Modularity**: Keep core logic, CLI, and symbology-specific code separate and extensible.

## 2. Project Structure
- **Core API**: `src/pyhue2d/core.py` — Main encode/decode API.
- **CLI**: `src/pyhue2d/cli.py` — Command-line interface.
- **Symbologies**: `src/pyhue2d/jabcode/` — JAB Code implementation (patterns, encoding modes, LDPC, etc.).
- **Helpers**: Place utility scripts in `/utility_scripts` (Python only).
- **Tests**: All tests in `tests/`.

## 3. Python Style
- Follow [PEP8](https://www.python.org/dev/peps/pep-0008/) for code style.
- Use [type hints](https://docs.python.org/3/library/typing.html) for all public functions and methods.
- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for all public APIs.
- Each module must have a module-level docstring.
- Imports: standard library, third-party, then local, each separated by a blank line.

## 4. Documentation
- All public functions, classes, and modules must have clear docstrings.
- Update or add documentation in `docs/` for new features or modules.
- Keep README and CLI help messages up to date with major changes.

## 5. Testing
- Use `pytest` for all tests.
- Place tests in `tests/`, mirroring the source structure where possible.
- Write tests for all new features and bug fixes.
- Aim for high coverage, especially for core logic and symbology modules.
- Test for both expected behavior and error conditions.

## 6. CLI Guidelines
- Use `argparse` for argument parsing.
- Provide clear, concise help messages for all CLI arguments.
- Handle errors gracefully and provide actionable feedback to users.

## 7. Extensibility
- New symbologies: Add as submodules under `src/pyhue2d/` (e.g., `src/pyhue2d/colorqr/`).
- New palettes or encoding modes: Add as new modules or classes, not as monolithic functions.
- Keep APIs open for extension but closed for modification (Open/Closed Principle).

## 8. File & Directory Conventions
- New modules should have an `__init__.py` and appropriate docstrings.
- Use lowercase_with_underscores for filenames and functions; CamelCase for classes.

## 9. Code Quality
- Format code with `black`.
- Lint with `flake8`.
- Type-check with `mypy`.
- Sort imports with `isort`.
- Run all checks before submitting a PR.

## 10. Commits & Pull Requests
- Write clear, descriptive commit messages.
- Reference issues or features in PRs.
- Ensure all tests pass and code quality checks succeed before requesting review.
- For more, see `CONTRIBUTING.md`

---

*Adhering to these guidelines ensures PyHue2D remains robust, readable, and a joy to contribute to.* 