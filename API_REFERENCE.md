# SatChange CLI Reference

Authoritative reference for SatChange command-line behavior.

## Global options

- `--verbose`, `-v`: enable DEBUG logs
- `--config-file PATH`: override config file location
- `--version`: print version
- `--help`: show help

## Exit codes

- `0`: success
- `1`: operational failure
- `130`: interrupted by user (`KeyboardInterrupt`)

## Commands

### `satchange config init`

Initialize authentication and persist managed key path.

```bash
satchange config init --service-account-key /path/to/key.json --project-id your-project-id
```

Options:

- `--service-account-key` (required): service account JSON path
- `--project-id` (required): Google Cloud project ID

Behavior:

- Copies the key to `~/.satchange/keys/`
- Stores config at `~/.satchange/config.yaml`

### `satchange config show`

Print loaded configuration and auth status.

```bash
satchange config show
```

### `satchange inspect`

Preview available Sentinel-2 scenes for AOI/date range.

```bash
satchange inspect --center "lat,lon" --size 100 --date-range "YYYY-MM-DD:YYYY-MM-DD" --cloud-threshold 20
```

Options:

- `--center` (required): `lat,lon`
- `--size`: AOI pixel dimensions (default `100`)
- `--date-range` (required): `start:end`
- `--cloud-threshold`: scene-level cloud threshold for catalog filtering (default `20`)

### `satchange analyze`

Run end-to-end change detection with AOI-local cloud validation and date fallback.

```bash
satchange analyze --center "lat,lon" --size 100 --date-a "YYYY-MM-DD" --date-b "YYYY-MM-DD" --change-type all --threshold 0.2 --output ./results
```

Options:

- `--center` (required)
- `--size`: default `100`
- `--date-a` (required)
- `--date-b` (required)
- `--cloud-threshold`: **local** cloud threshold %, default `15`
- `--change-type`: `vegetation | water | urban | all` (default `all`)
- `--threshold`: detection threshold `0.1..1.0` (default `0.2`)
- `--output` (required)
- `--name`: output prefix location label
- `--non-interactive`: auto-select recommended alternatives
- `--dry-run`: local validation only (no download/network/detection)
- `--include-legacy-html`: include legacy compatibility HTML output

Preflight behavior:

- validates coordinates/dates/thresholds
- checks output directory writeability
- requires at least 50 MB free disk space before download

Change classes:

| ID | Class |
|---|---|
| 0 | no change |
| 1 | vegetation growth |
| 2 | vegetation loss |
| 3 | water expansion |
| 4 | water reduction |
| 5 | urban development |
| 6 | urban decline |
| 7 | ambiguous |

### `satchange export`

Generate visualization outputs from existing analysis artifacts.

```bash
satchange export --result ./results --format all
```

Options:

- `--result` (required): result directory
- `--format`: `static | geotiff | all` (default `all`)
- `--emboss-intensity`: `0.0..2.0` (default `1.0`)
- `--name`: override output prefix
- `--include-legacy-html`: include legacy HTML output

Security constraint:

- Export expects band artifacts in `.npz` format.
- Legacy pickled `.npy` band dictionaries are rejected intentionally.

Web-first behavior:

- Emits `job_id` and `web_manifest`
- Prints `web_url_hint` (`http://localhost:3000/jobs/<job_id>`)
- Best-effort auto-start of local viewer if port 3000 is available

### `satchange cache status`

Show cache usage, hit rate, and item counts.

```bash
satchange cache status
```

### `satchange cache clear`

Clear all cached tiles (with confirmation prompt).

```bash
satchange cache clear
```

### `satchange cache cleanup`

Remove cache entries older than 30 days.

```bash
satchange cache cleanup
```

## Artifacts

Using prefix `{prefix} = {name_or_lat_lon}_{date_a}_{date_b}`:

- `{prefix}_bands_a.npz`
- `{prefix}_bands_b.npz`
- `{prefix}_classification.npy`
- `{prefix}_change_stats.json`
- `{prefix}_metadata.json`
- `{prefix}_visualization.png`
- `{prefix}_classification.tif`
- `{prefix}_job.json`
- `{prefix}_web_bundle/manifest.json`
- `{prefix}_interactive.html` (only when requested)

## Cloud fallback behavior

When local cloud checks fail, SatChange escalates through:

1. Threshold relaxation (`20 -> 40 -> 60`)
2. Temporal window expansion (`±30 -> ±60 -> ±90` days)
3. Temporal compositing (median of clearest scenes)

## Configuration schema

Default path: `~/.satchange/config.yaml`

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

Validation ranges:

| Field | Range |
|---|---|
| `cloud_threshold` | `0..100` |
| `pixel_size` | `10..1000` |
| `analysis.change_threshold` | `0.1..1.0` |
| `analysis.emboss_intensity` | `0.0..2.0` |
| `analysis.min_temporal_gap_days` | `30..365` |

## Quality commands

```bash
black --check satchange examples
flake8 satchange examples
mypy satchange
pytest -q
```
