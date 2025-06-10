# Contributing to PyHue2D

Thank you for your interest in contributing to PyHue2D! We welcome bug reports, feature requests, and code contributions from the community.

---

## 📚 Where to Start
- **Read the [CODING_GUIDELINES.md](CODING_GUIDELINES.md)** for code style, structure, and quality standards.
- See the [docs/](docs/) directory for API details and design rationale.

---

## 🐞 Bug Reports & Feature Requests
- **Bug reports**: Please include steps to reproduce, expected vs. actual behavior, and relevant logs or screenshots.
- **Feature requests**: Describe the problem, your proposed solution, and any alternatives considered.
- Use GitHub Issues for both.

---

## 🛠️ Development Setup
1. **Clone the repository** and create a branch for your work:
   ```bash
   git clone https://github.com/<username>/pyhue2d.git
   cd pyhue2d
   git checkout -b my-feature
   ```
2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install --upgrade pip
   pip install -r requirements-dev.txt
   pip install -e .
   ```
3. **Run tests and checks**:
   ```bash
   pytest tests/
   black src/ tests/
   flake8 src/ tests/
   mypy src/
   isort src/ tests/
   ```

---

## ✏️ Making Changes
- **Branch from `main`** for all work.
- Write clear, descriptive commit messages (imperative mood, e.g., "Add JAB Code level-H support").
- Reference related issues in your PR description.
- Ensure all tests and checks pass before submitting a pull request.

---

## ✅ Code Quality
- Follow [CODING_GUIDELINES.md](CODING_GUIDELINES.md) for style, structure, and documentation.
- Add or update tests for all new features and bug fixes.
- Update documentation as needed.

---

## 🔍 Pull Request Process
1. Open a PR against `main`.
2. The PR will be reviewed for code quality, clarity, and adherence to guidelines.
3. Respond to feedback and make requested changes.
4. Once approved, your PR will be merged.

---

## 🤝 Community & Conduct
- Be respectful and constructive in all interactions.
- For questions, open a GitHub Discussion or Issue.

---

*Thank you for helping make PyHue2D better!* 