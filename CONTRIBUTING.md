# Contributing to iso-obs

Thanks for helping make the Reliability Studio SDK better. This repo holds the public
Python packages: `iso-obs` (SDK), `iso-obs-cli`, and `iso-obs-schemas`.

## Ground rules

- Be excellent to each other — see [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md).
- Security issues go through [SECURITY.md](./SECURITY.md), never public issues.
- Open an issue before large changes so we can agree on direction first.

## Development setup

Prerequisites: Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/iso-ai/iso-obs.git
cd iso-obs
uv sync --all-packages
uv run pytest
```

## Coding standards

All Python code must pass the full gate before review:

```bash
uv run ruff check .        # lint + import order (Google docstring convention)
uv run black --check .     # formatting, line length 88
uv run mypy .              # strict type checking
uv run pytest              # tests
```

- **PEP 8**, Black-formatted, line length ≤ 88.
- **Google-style docstrings** on every module, class, and function.
- Comments explain *why*, not *what*.
- Type hints everywhere; `py.typed` is shipped and must stay honest.
- Add tests for every behavior change.

## Commits and PRs

- [Conventional Commits](https://www.conventionalcommits.org/): `feat(sdk): ...`,
  `fix(cli): ...`, `docs: ...`
- Branch from `main`, keep PRs focused and small.
- Fill in the PR template, link the issue, and confirm the checklist.
- Do not add AI co-author trailers to commits.

## Releases

Releases are cut by maintainers via tags (`sdk-v*`) and published to PyPI through
trusted publishing. Versioning follows semver; breaking changes before `1.0` are
called out in release notes.
