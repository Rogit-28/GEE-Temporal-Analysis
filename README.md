# SatChange

SatChange is a Python CLI for detecting temporal changes in satellite imagery
using Google Earth Engine (Sentinel-2).

## What It Does

- Detects vegetation, water, and urban changes using spectral indices
- Generates change maps and statistics
- Caches downloads locally to avoid repeat GEE requests

## Installation

```bash
git clone https://github.com/satchange/satchange.git
cd satchange
python -m venv venv
./venv/Scripts/activate          # Windows
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from satchange import main
```

## Configuration

Set up GEE access before running:

```bash
satchange config init --service-account-key /path/to/key.json --project-id your-project
```

## License

MIT
