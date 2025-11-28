# SatChange

A Python CLI tool for detecting temporal changes in satellite imagery using Google Earth Engine (Sentinel-2). Analyzes two dates over a geographic area, validates local cloud coverage, and produces change maps with statistics.

## Features

- **Spectral change detection** using NDVI (vegetation), NDWI (water), and NDBI (urban)
- **Multiple output formats** — static PNG, interactive HTML with Leaflet, GeoTIFF
- **Local cloud coverage validation** over your specific AOI (not just scene-level metadata)
- **Graduated cloud fallback** — threshold relaxation, temporal window expansion, median compositing
- **Disk-based LRU cache** to avoid redundant GEE downloads
- **Dry-run mode** to preview analysis without downloading imagery
- **Rich progress indicators** with graceful fallback

## Installation

```bash
git clone https://github.com/Rogit-28/GEE-Temporal-Analysis.git
cd GEE-Temporal-Analysis
python -m venv venv
./venv/Scripts/activate          # Windows
# source ./venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
pip install -e .
```

**Prerequisites**: Python 3.8+, a GEE service account with Earth Engine API enabled.

## Quick Start

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

### 4) Export Visualizations

```bash
satchange export --result ./results --format all
```

### 5) Manage Cache

```bash
satchange cache status          # Show cache statistics
satchange cache clear           # Delete all cached tiles
satchange cache cleanup         # Remove entries older than 30 days
```

## Usage Examples

### Dry Run (preview without downloading)

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

Checks cloud coverage and resolves dates, then reports what *would* happen without downloading imagery.

### Urban Change Detection

```bash
satchange analyze \
  --center "13.0827,80.2707" \
  --size 100 \
  --date-a "2020-01-15" \
  --date-b "2024-06-20" \
  --change-type urban \
  --output ./results
```

Detects urban development using NDBI from SWIR (B11) and NIR (B8) bands.

## Project Structure

```
satchange/
  __init__.py          # Package metadata (v0.1.0)
  __main__.py          # python -m satchange entry point
  cli.py               # Click CLI commands (config, analyze, inspect, export, cache)
  config.py            # YAML configuration manager
  gee_client.py        # GEE authentication, image query, download, cloud fallback
  image_processor.py   # Cloud masking, band resampling, radiometric normalization
  change_detector.py   # NDVI/NDWI/NDBI calculation, change classification
  visualization.py     # Emboss, colorize, PNG/HTML/GeoTIFF export
  cache.py             # Disk-based LRU cache with diskcache
  utils.py             # Coordinate parsing, logging, JSON encoding
  progress.py          # Rich progress bars with fallback
tests/
  conftest.py          # Shared fixtures and mocks
  test_*.py            # 310 tests across 8 test modules
examples/
  basic_usage.py       # Simple analysis workflow
  advanced_analysis.py # Multi-region, multi-type analysis
  integration_example.py # End-to-end pipeline
```

## Cloud Coverage Fallback

When a requested date has high cloud coverage over your AOI, SatChange applies a graduated fallback:

1. **Threshold relaxation** — retries with progressively higher tolerance (20% -> 40% -> 60%)
2. **Temporal window expansion** — searches nearby dates (+/-30 -> +/-60 -> +/-90 days)
3. **Temporal compositing** — creates a median composite from the clearest scenes in a +/-90-day window

In interactive mode the CLI presents alternatives for selection; in `--non-interactive` mode it auto-selects the best option.

## Error Handling

| Error | Meaning | Suggestion |
|-------|---------|------------|
| `GEE limit reached` | Quota or rate limit exceeded | Wait and retry, or reduce area |
| `No suitable imagery` | No scenes found for dates/area | Try different dates or higher `--cloud-threshold` |
| `Download failed` | Network or GEE download error | Check connection, retry |
| `Insufficient disk space` | Not enough free space for cache | Free up disk space |

## Notes

- **GEE credentials**: Requires a service account with Earth Engine API enabled.
- **Cloud coverage**: Scene-level cloud % can be misleading; SatChange checks **local** cloud coverage over your specific AOI.
- **B11 resolution**: SWIR band is 20m vs 10m for visible/NIR — SatChange resamples automatically via bicubic interpolation.
- **Free tier**: Designed to run entirely on GEE's free tier.

## Tech Stack

- **Data source**: Google Earth Engine (Sentinel-2 Surface Reflectance)
- **Caching**: `diskcache` with LRU eviction
- **CLI**: `click` with grouped commands
- **Visualization**: `matplotlib`, `opencv-python-headless`, `jinja2` (Leaflet maps), `rasterio` (GeoTIFF)
- **Progress**: `rich` (auto-installed, graceful fallback)
- **Testing**: `pytest` (310 tests)

## License

MIT
