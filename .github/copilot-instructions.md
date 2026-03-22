# Copilot instructions for SatChange

## Build, test, and lint commands
Run commands from the repository root.

## Autopilot mode
Never consume premium requests autonomously. One pattern I have noticed is that you don't give me final summary when the implementation is done - then consume another premium request before giving me the final summary. Please give me the final summary when the implementation is done, and only consume premium requests when I ask you to.

## /fleet and subagent invokations
Always use 0x premium requests for /fleet and subagent invocations, example models include raptor-mini and gpt-5 mini. If the subagent is working on a complex write/research/refactor task, you can ask for a premium request to be used for that subagent invocation or proceed with using cheaper models like claude haiku 4.6 or gpt 5.4-mini.

### Environment and install (effective build step)
- `python -m venv venv`
- `.\venv\Scripts\activate`
- `pip install -r requirements.txt`
- `pip install -e .`
- `pip install -r requirements-dev.txt`
- `python -m satchange --help` (sanity check for CLI entrypoint)

### Lint and type checks
- Format check: `black --check satchange examples`
- Lint: `flake8 satchange examples`
- Type check: `mypy satchange`

## High-level architecture
SatChange is a Click CLI (`satchange\cli.py`) that orchestrates a full imagery-change pipeline:

1. Configuration/authentication:
   - `Config` (`satchange\config.py`) loads/saves `~\.satchange\config.yaml`.
   - `config init` extracts `client_email` from the service-account JSON and persists auth settings.

2. Imagery discovery and cloud quality:
   - `GEEClient` (`satchange\gee_client.py`) authenticates against Earth Engine and queries Sentinel-2.
   - Cloud quality decisions are AOI-local (`check_local_cloud` using SCL), not scene metadata alone.
   - Date recovery/fallback logic is in `find_alternative_dates`, `handle_cloudy_scenes`, and `create_temporal_composite`.

3. Download and cache:
   - `CacheManager`/`ImageCache` (`satchange\cache.py`) wrap download calls and persist arrays/metadata in diskcache (LRU).
   - `analyze` downloads and caches `B4`, `B3`, `B2`, `B8`, `B11`, and `QA60`.

4. Preprocessing:
   - `ImageProcessor` (`satchange\image_processor.py`) resamples B11 from 20m to 10m, applies QA60 cloud masking, checks coregistration, and conditionally applies histogram matching.

5. Detection:
   - `ChangeDetector` (`satchange\change_detector.py`) computes NDVI/NDWI/NDBI deltas and creates class maps/statistics.
   - Classification IDs are stable across processing and visualization: `0` no change, `1` vegetation growth, `2` vegetation loss, `3` water expansion, `4` water reduction, `5` urban development, `6` urban decline, `7` ambiguous.

6. Output artifacts and visualization:
   - `analyze` writes NPY/JSON artifacts with prefix `{name_or_lat_lon}_{date_a}_{date_b}`.
   - `VisualizationManager` (`satchange\visualization.py`) generates static PNG, offline interactive HTML (Leaflet assets inlined), and classification GeoTIFF.
   - `export` supports both prefixed filenames and legacy non-prefixed filenames.

## Key repository-specific conventions
- CLI error contract is explicit and tested:
  - `sys.exit(1)` for operational failures.
  - `sys.exit(130)` for `KeyboardInterrupt`.
- Use `NumpyJSONEncoder` (`satchange\utils.py`) when serializing stats/metadata containing numpy types.
- Keep output naming consistent via `generate_output_prefix`; it strips any time portion from date strings before writing files.
- In user-facing cloud guidance, prioritize `local_cloud_pct` over `scene_cloud_pct`.
- Cache-key stability matters: keys hash rounded coordinates (6 decimals), pixel size, ISO date, and sorted bands.
