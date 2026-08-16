# Contributing to RF-Diffusion

Thank you for your interest in contributing.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Yangfan-Chou/RF-Diffusion.git
cd RF-Diffusion

# Set up upstream (required for running experiments)
bash scripts/setup_upstream.sh

# Install dependencies
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
```

## Making Changes

1. **Fork** the repository and create a branch from `main`.
2. **Write code** — keep changes focused. Follow the style of existing files.
3. **Add tests** for new functionality (no GPU needed; use mock data).
4. **Run tests** to ensure nothing is broken: `pytest tests/ -v`.
5. **Submit a PR** with a clear description of what changed and why.

## Guidelines

- Do **not** modify anything in `upstream/`. That directory holds vendored code from the official RF-Diffusion repository and must stay untouched.
- Keep docstrings concise — one summary line + key parameters. Avoid verbose Args/Returns lists.
- Use meaningful variable names (`sample_idx`, not `i`; `experiment_cfg`, not `cfg`).
- Import statements go at the top of files. Avoid inline imports inside functions unless there is a circular dependency.
- If you add a new script or entry point, update `setup.py` accordingly.

## Reporting Issues

- Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) for bugs.
- Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md) for suggestions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License of this project.
