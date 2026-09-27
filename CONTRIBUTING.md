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
   git clone https://github.com/firestrand/pyhue2d.git
   cd pyhue2d
   git checkout -b my-feature
   ```
2. **Set up environment with uv**:
   ```bash
   uv sync
   ```
3. **Run tests and checks**:
   ```bash
   # Using just (recommended)
   just check       # Static lint and type checks (ruff, ty)
   just test        # Run pytest suite
   just verify      # Full gate (check, examples, fixtures, docs, tests + coverage)

   # Or using uv directly
   uv run pytest
   uv run ruff check .
   uv run ruff format --check .
   uv run ty check
   ```

4. **Format and fix**:
   ```bash
   just format
   # or
   uv run ruff check --fix .
   uv run ruff format .
   ```

**Note:** Encoder and pipeline settings are immutable after initialization. To use different settings, create a new instance.

---

## ✏️ Making Changes
- **Branch from `main`** for all work.
- Write clear, descriptive commit messages (imperative mood, e.g., "Add JAB Code level-H support").
- Reference related issues in your PR description.
- Ensure all tests and checks pass before submitting a pull request.

### Conventional Commits
PyHue2D uses the [Conventional Commits](https://www.conventionalcommits.org/) specification so that our automated release tooling can determine the next semantic version and generate changelogs.

Prefix your commit subject with one of the following types:

* `feat:` – new user-facing feature
* `fix:` – bug fix
* `docs:` – documentation changes only
* `chore:` – CI, build system, or other internals
* `refactor:` – code change that neither fixes a bug nor adds a feature
* `test:` – adding or updating tests

Example:

```bash
git commit -m "feat: add HiQ Color-QR encoder"
```

The CI will fail if commit messages don't follow this pattern on protected branches.

---

## ✅ Code Quality
- Follow [CODING_GUIDELINES.md](CODING_GUIDELINES.md) for style, structure, and documentation.
- Add or update tests for all new features and bug fixes.
- Update documentation as needed.
- **Do not mutate encoder or pipeline settings after construction; always create a new instance for different configurations.**

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