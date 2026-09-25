## Summary

<!-- Brief description of the changes -->

## Quality Gate Checklist

- [ ] `uv sync --locked --all-extras --dev` succeeded
- [ ] `uv run ruff format --check .` passed
- [ ] `uv run ruff check .` passed
- [ ] `uv run ty check` passed
- [ ] `uv run pytest` passed
- [ ] No approved PNG fixtures or hash records were modified without review
- [ ] Branch coverage floor met (≥90% on changed behavioral code, ≥95% on new domain codec logic)
