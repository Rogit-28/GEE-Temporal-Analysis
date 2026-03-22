# Contributing to SatChange

Thanks for contributing.

## Development setup

```bash
git clone https://github.com/Rogit-28/GEE-Temporal-Analysis.git
cd GEE-Temporal-Analysis
python -m venv venv
# PowerShell:
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pip install -r requirements-dev.txt
```

## Before opening a PR

Run from repo root:

```bash
black --check satchange examples
flake8 satchange examples
mypy satchange
pytest -q
cd web && npm run build
```

## Contribution guidelines

- Keep changes focused and scoped.
- Update docs when behavior or flags change.
- Preserve CLI error contracts (`1` for operational failures, `130` for interrupts).
- Prefer secure defaults; never commit credentials.

## Documentation updates

If you change CLI behavior, update:

- `README.md`
- `RUN_INSTRUCTIONS.md`
- `API_REFERENCE.md`

## Security

- Do not commit service-account keys.
- Keep local credential files outside committed paths.
- Use sanitized output names and safe path handling patterns already present in the codebase.
