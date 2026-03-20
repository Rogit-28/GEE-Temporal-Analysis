# RUN_INSTRUCTIONS.md

SatChange — Run instructions (setup, demo commands, and troubleshooting)

Purpose
-------
This document captures everything needed to get SatChange running locally on Windows (also applicable to macOS/Linux with small path changes). It covers environment setup, CLI installation, example commands (dry-run and real runs), quality checks, caching, and troubleshooting.

Prerequisites
-------------
- Windows / macOS / Linux
- Python 3.8+ (recommended)
- Git
- Google Earth Engine access via a service account JSON key (Earth Engine API enabled) and a GCP project id
- Optional: GDAL/rasterio for GeoTIFF export (Windows users may prefer conda for rasterio/GDAL)

Security note
-------------
Do NOT commit service-account JSON files to source control. Keep credentials outside the repository and point to them with `satchange config init` (or `GOOGLE_APPLICATION_CREDENTIALS` where applicable).

Quick setup (development)
-------------------------
Open a terminal in the repository root and run (Bash syntax shown):

```bash
git clone https://github.com/Rogit-28/GEE-Temporal-Analysis.git
cd GEE-Temporal-Analysis
python -m venv venv
# Activate venv (Git Bash on Windows)
source venv/Scripts/activate
# macOS / Linux: source venv/bin/activate
# Windows Command Prompt: venv\Scripts\activate.bat
pip install -r requirements.txt
pip install -e .
```

Verify the CLI entrypoint:

```bash
python -m satchange --help
satchange --help
satchange --version
```

Configuration and authentication
--------------------------------
Initialize configuration with a service-account key and project id (these flags are required for `config init`):

```bash
satchange config init --service-account-key /path/to/your-key.json --project-id your-project-id
satchange config show
```

`config show` prints the loaded configuration (auth state, thresholds, cache settings). The config file is stored at `~/.satchange/config.yaml` by default.

NOTE: If you do not want to authenticate (e.g., to run a dry-run), you can omit `config init` — `--dry-run` runs offline and does not require credentials.

Inspecting imagery (preview)
---------------------------
Preview available Sentinel-2 scenes for an AOI and date-range:

```bash
satchange inspect --center "13.0827,80.2707" --size 100 --date-range "2022-01-01:2024-12-31" --cloud-threshold 20
```

- `--center` accepts `lat,lon` (string)
- `--size` is pixel dimensions (NxN), default 100
- `--date-range` uses `start:end`
- `--cloud-threshold` is scene-level threshold (percent)

Dry-run (plan without network calls)
-----------------------------------
Use `--dry-run` to validate inputs and print the planned analysis without contacting GEE or downloading imagery. This works without authentication.

```bash
satchange analyze --center "13.0827,80.2707" --size 100 --date-a "2022-02-04" --date-b "2024-10-26" --change-type all --output ./results --dry-run
```

Expected behavior:
- Validates coordinates, dates, thresholds, and output path locally
- Prints the analysis plan and the output filename prefix
- Exits without network calls or downloads

Run a full analysis (credentialed)
----------------------------------
Requires `satchange config init` (service account + project). Small AOI example:

```bash
satchange analyze --center "13.0827,80.2707" --size 100 --date-a "2022-02-04" --date-b "2024-10-26" --change-type all --cloud-threshold 20 --output ./results --name chennai --non-interactive
```

Flags of interest:
- `--cloud-threshold` (local cloud acceptance %, default 15)
- `--change-type` (`vegetation`, `water`, `urban`, `all`)
- `--threshold` (change-detection threshold, default 0.2)
- `--non-interactive` auto-selects alternatives (no prompts)

Outputs (written under `--output` with prefix `{name}_{date_a}_{date_b}`):
- `{prefix}_bands_a.npy` — bands for date A
- `{prefix}_bands_b.npy` — bands for date B
- `{prefix}_classification.npy` — classification map
- `{prefix}_change_stats.json` — change statistics
- `{prefix}_metadata.json` — metadata
- `{prefix}_visualization.png` — static PNG
- `{prefix}_interactive.html` — offline interactive HTML viewer
- `{prefix}_classification.tif` — GeoTIFF (requires rasterio)

Export visualizations from an existing results directory
------------------------------------------------------
Generate the static PNG, interactive HTML, and GeoTIFFs from a completed analysis:

```bash
satchange export --result ./results --format all --emboss-intensity 1.0
```

Cache management
----------------
```bash
satchange cache status    # show cache stats
satchange cache clear     # delete all cached tiles (prompts to confirm)
satchange cache cleanup   # remove entries older than 30 days
```

Quality checks:

```bash
black --check satchange examples
flake8 satchange examples
mypy satchange
```

Notes about optional/native dependencies
---------------------------------------
- GeoTIFF export uses `rasterio` and may require GDAL system libraries on Windows. If GeoTIFF export fails, install rasterio via conda or use a system wheel appropriate for your platform.

Integration examples and artifacts
---------------------------------
- Example scripts are under `examples/` (e.g., `integration_example.py`).
- Analysis outputs are generated into your chosen output directory (for example `./results`) and are intentionally not tracked in git.

Troubleshooting
---------------
- AuthenticationError: verify the service-account key file, project id, and ensure Earth Engine API is enabled for that service account.
- NoImageryError: try a wider date range, increase `--cloud-threshold`, or run `inspect` to preview scenes.
- Download failed / network errors: check connectivity and firewall settings. Ensure at least ~50 MB free in output path.
- GeoTIFF export errors: ensure rasterio/GDAL are installed; on Windows prefer conda-forge packages.

Additional recommendations
-------------------------
- Keep service-account keys out of the working tree. If a key was added locally, remove it or move it to a secure path and re-run `satchange config init`.
- Use `--dry-run` to validate parameters before performing credentialed runs.
- For reproducibility, pin dependency versions in `requirements.txt` or use a lock file.

References
----------
- README.md (project overview and quick start)
- API_REFERENCE.md (detailed CLI reference)
- .github/copilot-instructions.md (developer guidance)

If more specific example commands (e.g., a sample AOI GeoJSON or an integration notebook) are desired, say which AOI or workflow to include and this file can be extended with ready-to-run commands and sample data paths.
