# SatChange

SatChange is a Python CLI for detecting temporal changes in satellite imagery using Google Earth Engine (Sentinel-2). It analyzes two dates, checks local cloud coverage over your AOI, and produces change maps and statistics.

## What It Does

- Detects **vegetation**, **water**, and **urban** changes using NDVI/NDWI/NDBI
- Generates static PNGs, offline interactive HTML, and GeoTIFF outputs
- Caches downloads locally to avoid repeat GEE requests
- Validates **local** cloud coverage before running analysis
- Automatic cloud fallback: finds clearer alternatives when your dates are cloudy
- Rich progress indicators for long-running operations
- Dry-run mode to preview analysis without downloading

## How To Use

### 1) Configure GEE Access

```bash
satchange config init --service-account-key /path/to/key.json --project-id your-project
```

### 2) Inspect Available Imagery

```bash
satchange inspect --center "13.0827,80.2707" --size 100 --date-range "2020-01-01:2024-12-31"
```

### 3) Run Analysis

```bash
satchange analyze \
  --center "13.0827,80.2707" \
  --size 100 \
  --date-a "2022-02-04" \
  --date-b "2024-10-26" \
  --change-type water \
  --output ./results
```

#### Dry Run (preview without downloading)

```bash
satchange analyze \
  --center "13.0827,80.2707" \
  --size 100 \
  --date-a "2022-02-04" \
  --date-b "2024-10-26" \
  --change-type all \
  --output ./results \
  --dry-run
```

The `--dry-run` flag checks cloud coverage and resolves dates, then reports what *would* happen without downloading imagery or running analysis.

#### Urban Change Detection

```bash
satchange analyze \
  --center "13.0827,80.2707" \
  --size 100 \
  --date-a "2020-01-15" \
  --date-b "2024-06-20" \
  --change-type urban \
  --output ./results
```

Detects urban development and decline using the Normalized Difference Built-up Index (NDBI) from SWIR (B11) and NIR (B8) bands.

### 4) Generate Visualizations

```bash
satchange export --result ./results --format all
```

### 5) Manage Cache

```bash
satchange cache status          # Show cache statistics
satchange cache clear           # Delete all cached tiles
satchange cache cleanup         # Remove entries older than 30 days
```

## Installation

```bash
git clone https://github.com/satchange/satchange.git
cd satchange
python -m venv venv
./venv/Scripts/activate          # Windows
# source ./venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
pip install -e .
```

## Cloud Coverage Fallback

When a requested date has high cloud coverage over your AOI, SatChange automatically applies a graduated fallback strategy:

1. **Threshold relaxation** — Retries with progressively higher cloud tolerance (20% → 40% → 60%)
2. **Temporal window expansion** — Searches nearby dates (±30 → ±60 → ±90 days)
3. **Temporal compositing** — Creates a median composite from the clearest scenes in a ±90-day window

If alternatives are found, the CLI presents them for selection (or auto-selects in `--non-interactive` mode).

## Error Handling

SatChange provides specific error messages for common failure modes:

| Error | Meaning | Suggestion |
|-------|---------|------------|
| `GEE limit reached` | Quota or rate limit exceeded | Wait and retry, or reduce area |
| `No suitable imagery` | No scenes found for your dates/area | Try different dates or higher `--cloud-threshold` |
| `Download failed` | Network or GEE download error | Check connection, retry |
| `Insufficient disk space` | Not enough free space for downloads | Free up disk space |

## Cautions

- **GEE credentials required**: You must set up a service account and enable Earth Engine API.
- **Cloud coverage**: Scene-level cloud % can be misleading; SatChange checks **local** cloud coverage over your specific AOI.
- **Data availability**: Some dates may have no imagery for your AOI.
- **B11 resolution**: SWIR band (B11) is 20m vs 10m for visible/NIR — SatChange resamples automatically using bicubic interpolation.

## Supported Services

- **Google Earth Engine** (Sentinel-2 Surface Reflectance)
- **Local disk caching** via `diskcache`
- **Rich** progress indicators (auto-installed, graceful fallback if unavailable)
