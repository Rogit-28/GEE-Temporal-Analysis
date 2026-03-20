# SatChange CLI Reference

This document is the authoritative reference for the SatChange CLI interface.

## Global Options

- `--verbose` / `-v`: Enable verbose (DEBUG) logging
- `--config-file`: Path to custom config file (default: `~/.satchange/config.yaml`)
- `--version`: Show version and exit
- `--help`: Show help text

---

## Commands

### `satchange config init`
Initialize configuration and authenticate with Google Earth Engine.

```bash
satchange config init --service-account-key /path/to/key.json --project-id your-project-id
```

**Options**
- `--service-account-key` (required): Path to the service account JSON key file
- `--project-id` (required): Google Cloud project ID

---

### `satchange config show`
Display the currently loaded configuration (auth state, thresholds, cache, analysis settings).

```bash
satchange config show
```

---

### `satchange inspect`
Preview available Sentinel-2 scenes for an AOI and date range.

```bash
satchange inspect --center "lat,lon" --size 100 --date-range "YYYY-MM-DD:YYYY-MM-DD" --cloud-threshold 20
```

**Options**
- `--center` (required): `lat,lon` coordinates
- `--size`: Pixel dimensions (NxN), default `100`
- `--date-range` (required): `start:end` dates
- `--cloud-threshold`: Scene-level cloud threshold %, default `20`

---

### `satchange analyze`
Run a complete change analysis using explicit dates. The CLI validates **local cloud coverage** for each requested date and suggests alternatives if needed.

```bash
satchange analyze \
  --center "lat,lon" \
  --size 100 \
  --date-a "YYYY-MM-DD" \
  --date-b "YYYY-MM-DD" \
  --change-type all \
  --threshold 0.2 \
  --output ./results
```

**Options**
- `--center` (required): `lat,lon` coordinates
- `--size`: Pixel dimensions (NxN), default `100`
- `--date-a` (required): First comparison date
- `--date-b` (required): Second comparison date
- `--cloud-threshold`: **Local** cloud threshold %, default `15`
- `--change-type`: `vegetation`, `water`, `urban`, `all` (default: `all`)
- `--threshold`: Change detection threshold (0.1–1.0), default `0.2`
- `--output` (required): Output directory
- `--name`: Location name used in output filenames (default: `lat_lon`)
- `--non-interactive`: Auto-select recommended alternatives without prompting
- `--dry-run`: Validate local inputs and print the analysis plan, then exit without network calls, downloading, or analyzing

**Change Types**
| Type | Index | Bands | Classes |
|------|-------|-------|---------|
| `vegetation` | NDVI | (B8 − B4) / (B8 + B4) | 1 = growth, 2 = loss |
| `water` | NDWI | (B3 − B8) / (B3 + B8) | 3 = expansion, 4 = reduction |
| `urban` | NDBI | (B11 − B8) / (B11 + B8) | 5 = development, 6 = decline |
| `all` | All three | — | Combined classification (0–7) |

**Outputs**
- `{prefix}_bands_a.npy` — Raw band arrays for date A
- `{prefix}_bands_b.npy` — Raw band arrays for date B
- `{prefix}_classification.npy` — Change classification map
- `{prefix}_change_stats.json` — Change statistics
- `{prefix}_metadata.json` — Analysis metadata
- `{prefix}_visualization.png` — Static change map
- `{prefix}_interactive.html` — Offline interactive HTML viewer
- `{prefix}_classification.tif` — GeoTIFF with classification

Where `{prefix}` = `{name}_{date_a}_{date_b}` (e.g., `chennai_2022-02-04_2024-10-26`).

**Dry Run**

With `--dry-run`, the command:
1. Validates coordinates, dates, thresholds, and output path locally
2. Reports the planned analysis settings and output prefix
3. Exits **without** network calls, downloading imagery, or running change detection

```bash
satchange analyze --center "13.0827,80.2707" --size 100 \
  --date-a "2022-02-04" --date-b "2024-10-26" \
  --change-type all --output ./results --dry-run
```

**Disk Space Check**: Before downloading, the command verifies at least 50 MB of free space is available at the output path.

---

### `satchange export`
Generate visualizations from a previous analysis.

```bash
satchange export --result ./results --format all
```

**Options**
- `--result` (required): Results directory containing analysis output files
- `--format`: `static`, `interactive`, `geotiff`, `all` (default: `all`)
- `--emboss-intensity`: Emboss effect strength (0.0–2.0), default `1.0`
- `--name`: Override output name prefix

---

### `satchange cache status`
Show cache statistics: total items, size, hit rate, evictions.

```bash
satchange cache status
```

### `satchange cache clear`
Delete all cached tiles. Prompts for confirmation.

```bash
satchange cache clear
```

### `satchange cache cleanup`
Remove expired cache entries older than 30 days.

```bash
satchange cache cleanup
```

---

## Cloud Fallback Strategies

When a requested date has high local cloud coverage, `handle_cloudy_scenes()` applies three graduated strategies:

| Strategy | Method | Details |
|----------|--------|---------|
| **1. Threshold relaxation** | Retry with higher tolerance | 20% → 40% → 60% cloud acceptance |
| **2. Temporal window expansion** | Search nearby dates | ±30 → ±60 → ±90 day windows |
| **3. Temporal compositing** | Median composite | Combines the 5 clearest scenes in ±90 days |

The return value includes `strategy_used`, `image`, `image_id`, `date`, `local_cloud_pct`, and `details`.

---

## Exception Hierarchy

All GEE-related exceptions inherit from `GEEError`:

```
GEEError (base)
├── AuthenticationError    — Invalid credentials or missing key file
├── QuotaExceededError     — GEE quota limit reached (HTTP 403)
├── RateLimitError         — GEE rate limit hit (HTTP 429)
├── NoImageryError         — No scenes found for the given parameters
└── DownloadError          — Download failed after retries
```

Import from `satchange.gee_client`:
```python
from satchange.gee_client import (
    GEEError, AuthenticationError, QuotaExceededError,
    RateLimitError, NoImageryError, DownloadError,
)
```

---

## Key Internal Methods

### `GEEClient.handle_cloudy_scenes(date, center_coords, cloud_threshold, pixel_size)`
Applies the 3-strategy cloud fallback. Returns a dict with `found`, `strategy_used`, `image`, `image_id`, `date`, `local_cloud_pct`, `details`.

### `GEEClient.create_temporal_composite(center_coords, target_date, pixel_size, window_days=90, max_scenes=5)`
Creates a median composite from the clearest scenes in a temporal window. Returns an `ee.Image` or `None` if fewer than 2 scenes are available.

### `GEEClient.check_local_cloud(date, center_coords, pixel_size)`
Checks local cloud coverage using SCL band values 3, 8, 9, 10, 11 (including snow/ice). Returns dict with `found`, `is_good`, `local_cloud_pct`, `scene_cloud_pct`, `image_id`, `date`.

### `GEEClient.find_alternative_dates(date, center_coords, cloud_threshold, pixel_size)`
Searches incrementally wider windows (±2w, ±1m, ±2m, ±3m) for clearer dates. Returns dict with `threshold_met`, `alternatives` (sorted by cloud %), `search_window`.

### `check_disk_space(path, required_mb=100.0)`
Checks available disk space. Returns `{"available_mb", "required_mb", "sufficient"}`.

---

## Quality Checks

Run from repository root:

```bash
black --check satchange examples
flake8 satchange examples
mypy satchange
```

---

## Configuration

Config file: `~/.satchange/config.yaml`

```yaml
service_account_key: /path/to/key.json
project_id: your-project-id
service_account_email: sa@project.iam.gserviceaccount.com
cloud_threshold: 20
pixel_size: 100
cache:
  max_size_gb: 5
  eviction_policy: least-recently-used
  directory: ~/.satchange/cache
analysis:
  change_threshold: 0.2
  emboss_intensity: 1.0
  min_temporal_gap_days: 180
```

**Validation Ranges**
| Field | Range |
|-------|-------|
| `cloud_threshold` | 0–100 |
| `pixel_size` | 10–1000 |
| `analysis.change_threshold` | 0.1–1.0 |
| `analysis.emboss_intensity` | 0.0–2.0 |
| `analysis.min_temporal_gap_days` | 30–365 |
