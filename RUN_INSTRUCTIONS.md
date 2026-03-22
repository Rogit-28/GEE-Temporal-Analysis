# SatChange Run Instructions

Setup, demo execution, validation, and troubleshooting.

## 1) Prerequisites

- Windows / macOS / Linux
- Python 3.8+
- Git
- Google Earth Engine service-account JSON key + GCP project ID
- Optional: GDAL/rasterio system dependencies for GeoTIFF export

> Security: never commit service-account keys. Keep key files local and git-ignored.

## 2) Environment setup (repo root)

```bash
git clone https://github.com/Rogit-28/GEE-Temporal-Analysis.git
cd GEE-Temporal-Analysis
python -m venv venv
# PowerShell:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
# source venv/bin/activate
pip install -r requirements.txt
pip install -e .
# For lint/type/test/release checks:
pip install -r requirements-dev.txt
```

Verify CLI entrypoints:

```bash
python -m satchange --help
satchange --help
satchange --version
```

## 3) Configure authentication

```bash
satchange config init --service-account-key /path/to/key.json --project-id your-project-id
satchange config show
```

Behavior notes:

- Key is copied to `~/.satchange/keys/`.
- Config is stored at `~/.satchange/config.yaml` by default.
- Re-run `config init` if managed key is removed or rotated.

## 4) Run core CLI workflows

### A) Dry-run (no network, no downloads)

```bash
satchange analyze --center "13.0827,80.2707" --size 100 --date-a "2022-02-04" --date-b "2024-10-26" --change-type all --output ./results --dry-run
```

Expected behavior:

- Local validation only
- Prints planned output prefix
- Exits without imagery download

### B) Inspect imagery catalog

```bash
satchange inspect --center "13.0827,80.2707" --size 100 --date-range "2022-01-01:2024-12-31" --cloud-threshold 20
```

### C) Analyze changes

```bash
satchange analyze --center "13.0827,80.2707" --size 100 --date-a "2022-02-04" --date-b "2024-10-26" --change-type all --cloud-threshold 20 --output ./results --name chennai --non-interactive
```

### D) Export visualization outputs

```bash
satchange export --result ./results --format all
# Optional legacy compatibility output:
satchange export --result ./results --format all --include-legacy-html
```

## 5) Validate web viewer

```bash
cd web
npm install
# PowerShell:
$env:SATCHANGE_RESULTS_DIR = "C:\path\to\results"
npm run build
npm run dev:reset
```

Open:

```text
http://localhost:3000/jobs/<job_id>
http://localhost:3000/api/jobs/<job_id>
```

If styles look broken, clear stale cache and restart:

```bash
cd web
npm run dev:reset
```

If CLI prints `web_viewer: not started (...)`, run the manual command shown in CLI output.

## 6) Output artifacts

For prefix `{name}_{date_a}_{date_b}`:

- `{prefix}_bands_a.npz`, `{prefix}_bands_b.npz`
- `{prefix}_classification.npy`
- `{prefix}_change_stats.json`
- `{prefix}_metadata.json`
- `{prefix}_visualization.png`
- `{prefix}_classification.tif`
- `{prefix}_job.json`
- `{prefix}_web_bundle/manifest.json`
- `{prefix}_interactive.html` (only with `--include-legacy-html`)

## 7) Quality and release verification

Run from repository root:

```bash
black --check satchange examples
flake8 satchange examples
mypy satchange
pytest -q
cd web && npm run build
```

Latest local verified baseline for this repo:

- Lint/type/tests: pass (`10 passed`)
- Web production build: pass
- CLI entrypoints: pass

## 8) Troubleshooting

- **Not authenticated**: run `satchange config init ...`.
- **No imagery found**: widen date range or raise cloud threshold.
- **Insufficient disk space**: free space in output directory.
- **GeoTIFF export failures**: install compatible rasterio/GDAL for your platform.
- **Web data not loading**: verify `SATCHANGE_RESULTS_DIR` points to the results directory with job manifest files.

## 9) Security and hygiene checklist

- Keep keys out of source control.
- Prefer managed key path under `~/.satchange/keys/`.
- Use `--dry-run` before expensive/credentialed runs.
- Do not rely on legacy pickled `*_bands_*.npy` for export; use `.npz` artifacts.

## 10) Release checklist

- [ ] `README.md`, `RUN_INSTRUCTIONS.md`, and `API_REFERENCE.md` are consistent
- [ ] Demo commands and output snippets are verified
- [ ] Internal doc links resolve
- [ ] `black`, `flake8`, `mypy`, and `pytest` pass
- [ ] Web build succeeds (`cd web && npm run build`)
- [ ] No credentials or secrets present in tracked files
- [ ] Release notes summarize user-facing CLI/output changes

## 11) References

- [README.md](README.md)
- [API_REFERENCE.md](API_REFERENCE.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
