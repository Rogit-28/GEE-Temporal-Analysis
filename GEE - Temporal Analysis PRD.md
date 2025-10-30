# Product Requirements Document: SatChange CLI

## Executive Summary

**Product Name**: SatChange  
**Type**: Command-line interface (CLI) tool  
**Purpose**: Enable users to detect and visualize temporal changes in satellite imagery for specified geographic areas  
**Target Users**: Hobbyists, researchers, environmental enthusiasts (max 10 concurrent users)  
**Constraints**: Free-tier infrastructure only, Google Earth Engine API as primary data source

---

## Product Overview

### Problem Statement
Users need a simple way to analyze how specific geographic areas have changed over time (urbanization, deforestation, water level changes, vegetation shifts) without requiring deep remote sensing expertise or expensive commercial satellite data access.

### Solution
A Python-based CLI tool that:
1. Accepts geographic coordinates and date ranges from users
2. Fetches satellite imagery from Google Earth Engine (Sentinel-2)
3. Computes spectral index differences to identify changes
4. Generates embossed visualizations highlighting detected changes
5. Outputs interactive HTML viewers for before/after comparison

### Success Metrics
- Processing time: <30 seconds for cached 100x100 pixel analysis
- Accuracy: Correctly detects known deforestation events in validation tests
- Usability: One-command analysis from fresh installation
- Reliability: Graceful handling of cloud coverage and data availability issues

---

## Technical Architecture

### System Components

#### 1. CLI Interface Layer
**Technology**: Python `click` framework  
**Responsibilities**:
- Parse user commands and arguments
- Validate input parameters (coordinates, dates, pixel dimensions)
- Display progress indicators and status messages
- Handle errors with actionable user feedback

**Key Commands**:
- `satchange config init` - Set up GEE authentication
- `satchange inspect` - Preview available imagery for area/date
- `satchange analyze` - Execute change detection pipeline
- `satchange export` - Render final visualizations
- `satchange cache` - Manage local tile storage

#### 2. Configuration Management
**Storage**: `~/.satchange/config.yaml`  
**Contents**:
- Google Earth Engine service account credentials (JSON key path)
- Default analysis parameters (cloud threshold, pixel size)
- Cache settings (max size, eviction policy)

**Implementation Notes**:
- Use `pyyaml` for config file I/O
- Validate credentials on first run with GEE test query
- Store sensitive data (API keys) with restricted file permissions (chmod 600)

#### 3. Google Earth Engine Integration
**Data Source**: Sentinel-2 Surface Reflectance (`COPERNICUS/S2_SR_HARMONIZED`)  
**Resolution**: 10 meters/pixel (RGB + NIR bands)  
**Temporal Coverage**: 2015-present, 5-day revisit frequency

**Query Pipeline**:
1. Convert user lat/lon + pixel dimensions to bounding box (GeoJSON polygon)
2. Query GEE ImageCollection with filters:
   - Geographic bounds (from bounding box)
   - Date range (user-specified start/end)
   - Cloud coverage threshold (default: <20%)
3. Sort results by cloud coverage percentage (ascending)
4. Select "best" image pair:
   - Clearest image near start date
   - Clearest image near end date
   - Preferably same season to minimize phenological differences

**API Methods**:
- `ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(bbox).filterDate(start, end)`
- `ee.Image.getThumbURL()` for small areas (<100x100 pixels) - returns direct download URL
- `ee.batch.Export.image.toDrive()` for larger areas - asynchronous export to Google Drive

**Band Selection**:
- B4 (Red): 665nm wavelength, 10m resolution
- B3 (Green): 560nm wavelength, 10m resolution  
- B8 (NIR): 842nm wavelength, 10m resolution
- QA60 (Quality band): Cloud mask metadata

#### 4. Local Caching System
**Technology**: `diskcache` library (SQLite-backed key-value store)  
**Location**: `~/.satchange/cache/`  
**Strategy**: LRU eviction with 5GB cap

**Cache Key Structure**:
```
hash(center_lat, center_lon, pixel_size, date, satellite_id)
→ unique tile identifier
```

**Cached Data**:
- Raw band arrays (B4, B3, B8) as GeoTIFF format via `rasterio`
- Metadata: acquisition date, cloud coverage %, scene ID

**Cache Hit Logic**:
1. Generate cache key from query parameters
2. Check if key exists in cache
3. If hit: Load GeoTIFF from disk, skip GEE download
4. If miss: Fetch from GEE, save to cache, return data

**Eviction Policy**:
- Monitor total cache size after each write
- If exceeds 5GB: Remove least-recently-used tiles until under threshold
- User can manually clear cache with `satchange cache clear`

#### 5. Image Preprocessing Pipeline

**Input**: Raw Sentinel-2 bands from GEE or cache  
**Output**: Analysis-ready coregistered image pairs

**Steps**:

**Step 5.1: Cloud Masking**
- Use QA60 band (bitmask for clouds/cirrus)
- Create binary mask where cloudy pixels = 0, clear pixels = 1
- If cloud coverage in AOI exceeds threshold: Reject scene and try next clearest

**Step 5.2: Coregistration Check**
- Sentinel-2 tiles are pre-georeferenced by ESA
- Verify both images align to same UTM grid
- If using data from different satellite paths: Apply affine transformation with `rasterio.warp`

**Step 5.3: Radiometric Normalization**
- Not strictly required for Surface Reflectance products (already atmospherically corrected)
- Optional histogram matching if images show significant brightness differences

**Implementation Notes**:
- GEE's Surface Reflectance collection includes atmospheric correction via Sen2Cor
- Focus preprocessing effort on cloud masking - it's the primary failure mode

#### 6. Change Detection Engine

**Methodology**: Spectral index differencing

**Spectral Indices Computed**:

**NDVI (Normalized Difference Vegetation Index)**:
```
NDVI = (NIR - Red) / (NIR + Red)
```
- Range: -1 to +1
- High values (>0.6): Dense vegetation
- Low values (<0.2): Bare soil, water, urban
- Use case: Detect deforestation, agricultural changes, vegetation growth

**NDWI (Normalized Difference Water Index)**:
```
NDWI = (Green - NIR) / (Green + NIR)
```
- Range: -1 to +1
- High values (>0.3): Water bodies
- Low values (<0): Vegetation, dry land
- Use case: Detect flooding, drought, water body expansion/contraction

**NDBI (Normalized Difference Built-up Index)**:
```
NDBI = (SWIR - NIR) / (SWIR + NIR)
```
- Range: -1 to +1
- High values: Built-up areas, urban surfaces
- Use case: Detect urbanization, construction

**Change Detection Algorithm**:
1. Compute spectral index for Date A (earlier) image
2. Compute same spectral index for Date B (later) image
3. Calculate difference: `delta = index_B - index_A`
4. Apply threshold to identify significant changes: `abs(delta) > 0.2` (configurable)
5. Classify change direction:
   - `delta > +0.2`: Growth/increase (vegetation growth, water expansion, urban development)
   - `delta < -0.2`: Loss/decrease (deforestation, drought, demolition)

**Multi-Index Analysis** (when `--change-type all` specified):
- Compute all three indices (NDVI, NDWI, NDBI)
- Generate separate change masks for each
- Combine into multi-band change map:
  - Red channel: NDVI decreases (vegetation loss)
  - Green channel: NDVI increases (vegetation growth)
  - Blue channel: NDWI increases (water expansion)

**Output Data Structures**:
- `change_mask`: Binary array where 1 = significant change detected
- `change_magnitude`: Float array with delta values
- `change_type`: Categorical array (vegetation_loss, vegetation_growth, water_increase, etc.)

#### 7. Visualization Renderer

**Goal**: Generate embossed visual effects that emphasize detected changes

**Embossing Algorithm**:
```python
emboss_kernel = np.array([
    [-2, -1,  0],
    [-1,  1,  1],
    [ 0,  1,  2]
])
embossed = cv2.filter2D(change_mask, -1, emboss_kernel)
```
- Emboss kernel creates 3D "raised" effect by computing directional gradients
- Apply to binary change mask to create depth perception
- Normalize output to 0-255 range for image rendering

**Static Output Generation**:
1. Load base imagery (Date B RGB composite)
2. Convert embossed change mask to RGBA (alpha channel for transparency)
3. Color-code changes:
   - Red (#FF0000, 70% opacity): Vegetation loss, water decrease
   - Green (#00FF00, 70% opacity): Vegetation growth
   - Blue (#0000FF, 70% opacity): Water increase, urban development
4. Alpha blend embossed overlay onto base imagery
5. Save as PNG with `matplotlib.pyplot.imsave()`

**Interactive HTML Viewer**:
**Technology**: Leaflet.js for map interface

**Structure**:
```html
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
</head>
<body>
    <div id="map"></div>
    <div id="controls">
        <button onclick="showLayer('before')">Date A</button>
        <button onclick="showLayer('after')">Date B</button>
        <button onclick="showLayer('embossed')">Changes</button>
    </div>
    <div id="stats">
        <p>Total area changed: {percent}%</p>
        <p>Vegetation loss: {veg_loss_percent}%</p>
        <p>Water expansion: {water_increase_percent}%</p>
    </div>
    <script>
        // Initialize map with center coordinates
        // Load three image layers as overlays
        // Toggle visibility on button click
    </script>
</body>
</html>
```

**Generation Process**:
1. Save Date A, Date B, and embossed images as base64-encoded data URIs
2. Render Jinja2 template with embedded images
3. Write HTML to output file
4. Include JavaScript for layer toggling and zoom controls

**Statistics Calculation**:
- Total pixels in AOI: `width * height`
- Changed pixels: `sum(change_mask == 1)`
- Percent changed: `(changed_pixels / total_pixels) * 100`
- Breakdown by change type: Count pixels in each category

#### 8. Export Manager

**Supported Output Formats**:

**PNG** (Static visualization):
- RGB composite with embossed overlay
- Resolution: Native pixel dimensions (e.g., 100x100)
- Color depth: 8-bit per channel
- Use case: Quick preview, embedding in reports

**HTML** (Interactive viewer):
- Self-contained single-file application
- No external dependencies (all assets embedded)
- Works offline after generation
- Use case: Exploratory analysis, presentations

**GeoTIFF** (Geospatial format):
- Preserves geographic coordinates and projection
- Includes metadata: CRS, transform, acquisition dates
- Multi-band: RGB + change mask as 4th band
- Use case: Import into GIS software (QGIS, ArcGIS)

**JSON** (Statistics metadata):
```json
{
    "analysis_date": "2024-10-30T10:30:00Z",
    "aoi": {
        "center": [13.0827, 80.2707],
        "pixel_size": 100,
        "area_km2": 1.0
    },
    "dates": {
        "start": "2020-03-15",
        "end": "2024-09-22"
    },
    "changes": {
        "total_percent": 12.4,
        "vegetation_loss": 8.1,
        "vegetation_growth": 2.3,
        "water_increase": 2.0
    }
}
```

**File Naming Convention**:
```
{location_name}_{start_date}_{end_date}_{output_type}.{ext}
chennai_20200315_20240922_embossed.png
chennai_20200315_20240922_interactive.html
chennai_20200315_20240922_stats.json
```

---

## Phase-Wise Implementation Plan

### Phase 1: Foundation - GEE Authentication & Query

**Objective**: Establish connection to Google Earth Engine and execute basic imagery queries

**Deliverables**:
1. CLI entry point with `click` command structure
2. Configuration file management (`config init` command)
3. GEE authentication flow using service account
4. Basic imagery metadata query (scene list with cloud coverage)

**Detailed Tasks**:

**Task 1.1: Project Structure Setup**
```
satchange/
├── satchange/
│   ├── __init__.py
│   ├── cli.py          # Click command definitions
│   ├── config.py       # Config file I/O
│   ├── gee_client.py   # Earth Engine API wrapper
│   └── utils.py        # Helper functions
├── tests/
│   └── test_gee_auth.py
├── requirements.txt
├── setup.py
└── README.md
```

**Task 1.2: Configuration Management Implementation**
- Create `Config` class with methods:
  - `load()`: Read YAML from `~/.satchange/config.yaml`
  - `save()`: Write YAML with validated parameters
  - `get(key)`: Retrieve config value with default fallback
- Store GEE service account key path, not key contents (security)
- Validate config schema on load (required fields present)

**Task 1.3: GEE Authentication Flow**
```python
def authenticate_gee(service_account_key_path):
    """
    Initialize Earth Engine with service account credentials.
    
    Args:
        service_account_key_path: Path to JSON key file
        
    Returns:
        bool: True if authentication successful
        
    Raises:
        AuthenticationError: If credentials invalid
    """
    credentials = ee.ServiceAccountCredentials(
        email='SERVICE_ACCOUNT_EMAIL',
        key_file=service_account_key_path
    )
    ee.Initialize(credentials)
    
    # Validation test query
    test = ee.Image('COPERNICUS/S2_SR_HARMONIZED/20200101T000000_20200101T235959_T43PGP')
    test.getInfo()  # Raises exception if auth failed
    
    return True
```

**Task 1.4: Coordinate to Bounding Box Conversion**
```python
def create_bbox(center_lat, center_lon, pixel_size, resolution_meters=10):
    """
    Convert center point and pixel dimensions to geographic bounding box.
    
    Args:
        center_lat: Latitude of AOI center
        center_lon: Longitude of AOI center
        pixel_size: Number of pixels per side (e.g., 100 for 100x100)
        resolution_meters: Spatial resolution (Sentinel-2 = 10m)
        
    Returns:
        ee.Geometry.Polygon: Bounding box for GEE query
    """
    # Calculate AOI dimensions in meters
    width_meters = pixel_size * resolution_meters
    height_meters = pixel_size * resolution_meters
    
    # Convert to degrees (approximate at equator: 1 degree ≈ 111km)
    # Use geopy for accurate conversion at any latitude
    from geopy.distance import distance
    
    north = distance(meters=height_meters/2).destination((center_lat, center_lon), 0)
    south = distance(meters=height_meters/2).destination((center_lat, center_lon), 180)
    east = distance(meters=width_meters/2).destination((center_lat, center_lon), 90)
    west = distance(meters=width_meters/2).destination((center_lat, center_lon), 270)
    
    bbox = ee.Geometry.Polygon([[
        [west.longitude, south.latitude],
        [east.longitude, south.latitude],
        [east.longitude, north.latitude],
        [west.longitude, north.latitude]
    ]])
    
    return bbox
```

**Task 1.5: Implement `satchange inspect` Command**
```python
@click.command()
@click.option('--center', required=True, type=str, help='Lat,Lon (e.g., 13.0827,80.2707)')
@click.option('--size', default=100, type=int, help='Pixel dimensions (NxN)')
@click.option('--date-range', required=True, type=str, help='Start:End (YYYY-MM-DD:YYYY-MM-DD)')
@click.option('--cloud-threshold', default=20, type=int, help='Max cloud coverage %')
def inspect(center, size, date_range, cloud_threshold):
    """Preview available Sentinel-2 scenes for AOI."""
    
    # Parse inputs
    lat, lon = map(float, center.split(','))
    start_date, end_date = date_range.split(':')
    
    # Create bounding box
    bbox = create_bbox(lat, lon, size)
    
    # Query GEE
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(bbox) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
    
    # Retrieve metadata
    scenes = collection.getInfo()['features']
    
    # Display results
    click.echo(f"Found {len(scenes)} clear scenes")
    click.echo(f"\nTop 5 clearest:")
    for scene in sorted(scenes, key=lambda x: x['properties']['CLOUDY_PIXEL_PERCENTAGE'])[:5]:
        date = scene['properties']['SENSING_TIME']
        cloud_pct = scene['properties']['CLOUDY_PIXEL_PERCENTAGE']
        click.echo(f"  {date} - {cloud_pct:.1f}% clouds")
```

**Validation Criteria**:
- [ ] `satchange config init` creates valid config file
- [ ] `satchange inspect` returns scene list for valid coordinates
- [ ] Authentication fails gracefully with helpful error message
- [ ] Invalid coordinates rejected with suggestion to check format

**Required Packages** (Phase 1):
```
earthengine-api==0.1.380
click==8.1.7
pyyaml==6.0.1
geopy==2.4.0
```

---

### Phase 2: Image Acquisition - Download & Cache

**Objective**: Fetch satellite imagery from GEE and implement local caching to avoid redundant downloads

**Deliverables**:
1. Smart date pair selection algorithm (clearest images)
2. Image download via GEE API
3. Local cache storage with LRU eviction
4. Cache hit/miss logging

**Detailed Tasks**:

**Task 2.1: Smart Date Selection Algorithm**
```python
def select_best_image_pair(collection, start_date, end_date):
    """
    Select optimal image pair from collection based on cloud coverage and temporal spread.
    
    Strategy:
    1. Sort all scenes by cloud coverage (ascending)
    2. Find clearest scene within 30 days of start_date
    3. Find clearest scene within 30 days of end_date
    4. If no scenes found in windows: Expand to 60 days, then 90 days
    5. Ensure minimum 6-month gap between selected pair
    
    Args:
        collection: ee.ImageCollection (filtered by AOI and date range)
        start_date: datetime object
        end_date: datetime object
        
    Returns:
        tuple: (date_a_image, date_b_image) as ee.Image objects
    """
    from datetime import timedelta
    
    # Convert collection to sorted list
    scenes = collection.sort('CLOUDY_PIXEL_PERCENTAGE').getInfo()['features']
    
    # Find Date A (near start)
    date_a_window = [start_date, start_date + timedelta(days=30)]
    date_a_candidates = [s for s in scenes if date_a_window[0] <= parse_date(s) <= date_a_window[1]]
    
    if not date_a_candidates:
        # Expand window
        date_a_window[1] = start_date + timedelta(days=90)
        date_a_candidates = [s for s in scenes if date_a_window[0] <= parse_date(s) <= date_a_window[1]]
    
    date_a_scene = date_a_candidates[0]  # Clearest in window
    
    # Find Date B (near end, minimum 6 months after Date A)
    min_gap = parse_date(date_a_scene) + timedelta(days=180)
    date_b_window = [max(min_gap, end_date - timedelta(days=30)), end_date]
    date_b_candidates = [s for s in scenes if date_b_window[0] <= parse_date(s) <= date_b_window[1]]
    
    if not date_b_candidates:
        date_b_window[0] = min_gap
        date_b_candidates = [s for s in scenes if date_b_window[0] <= parse_date(s) <= date_b_window[1]]
    
    date_b_scene = date_b_candidates[0]
    
    return (
        ee.Image(date_a_scene['id']),
        ee.Image(date_b_scene['id'])
    )
```

**Task 2.2: Image Download Implementation**
```python
def download_image(ee_image, bbox, bands=['B4', 'B3', 'B8'], scale=10):
    """
    Download image bands from GEE as numpy arrays.
    
    Args:
        ee_image: ee.Image object
        bbox: ee.Geometry.Polygon defining AOI
        bands: List of band names to download
        scale: Resolution in meters (10m for Sentinel-2)
        
    Returns:
        dict: {band_name: numpy.ndarray}
    """
    # For small areas: Use getThumbURL (fast, synchronous)
    url = ee_image.select(bands).getThumbURL({
        'region': bbox,
        'dimensions': f'{pixel_size}x{pixel_size}',  # From user input
        'format': 'GEO_TIFF'
    })
    
    # Download GeoTIFF
    import requests
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    
    # Parse with rasterio
    import rasterio
    from io import BytesIO
    
    with rasterio.open(BytesIO(response.content)) as dataset:
        band_arrays = {
            band: dataset.read(i+1) for i, band in enumerate(bands)
        }
        metadata = {
            'transform': dataset.transform,
            'crs': dataset.crs,
            'bounds': dataset.bounds
        }
    
    return band_arrays, metadata
```

**Task 2.3: Caching System Implementation**
```python
import diskcache as dc
import hashlib
import json

class ImageCache:
    """Disk-based LRU cache for satellite imagery tiles."""
    
    def __init__(self, cache_dir='~/.satchange/cache', size_limit=5e9):
        """
        Args:
            cache_dir: Directory for cache storage
            size_limit: Maximum cache size in bytes (default 5GB)
        """
        self.cache_dir = os.path.expanduser(cache_dir)
        self.cache = dc.Cache(self.cache_dir, size_limit=int(size_limit), eviction_policy='least-recently-used')
    
    def _generate_key(self, center_lat, center_lon, pixel_size, date, bands):
        """Generate unique cache key from query parameters."""
        params = {
            'lat': round(center_lat, 6),
            'lon': round(center_lon, 6),
            'size': pixel_size,
            'date': date.isoformat(),
            'bands': sorted(bands)
        }
        key_string = json.dumps(params, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(self, center_lat, center_lon, pixel_size, date, bands):
        """Retrieve cached image data."""
        key = self._generate_key(center_lat, center_lon, pixel_size, date, bands)
        return self.cache.get(key)
    
    def set(self, center_lat, center_lon, pixel_size, date, bands, data):
        """Store image data in cache."""
        key = self._generate_key(center_lat, center_lon, pixel_size, date, bands)
        self.cache.set(key, data)
    
    def stats(self):
        """Return cache statistics."""
        return {
            'size_mb': self.cache.volume() / 1e6,
            'items': len(self.cache),
            'hits': self.cache.stats(enable=True).get('hits', 0),
            'misses': self.cache.stats(enable=True).get('misses', 0)
        }
```

**Task 2.4: Integrate Cache into Download Pipeline**
```python
def fetch_image_with_cache(ee_image, scene_date, bbox, bands, cache):
    """
    Fetch image from cache or GEE.
    
    Returns:
        tuple: (band_arrays, metadata, cache_hit: bool)
    """
    # Try cache first
    cached = cache.get(center_lat, center_lon, pixel_size, scene_date, bands)
    
    if cached is not None:
        click.echo("Cache hit - loading from disk")
        return cached['arrays'], cached['metadata'], True
    
    # Cache miss - download from GEE
    click.echo("Cache miss - downloading from GEE...")
    band_arrays, metadata = download_image(ee_image, bbox, bands)
    
    # Store in cache
    cache.set(center_lat, center_lon, pixel_size, scene_date, bands, {
        'arrays': band_arrays,
        'metadata': metadata
    })
    
    return band_arrays, metadata, False
```

**Task 2.5: Implement `satchange analyze` (Download Only)**
```python
@click.command()
@click.option('--center', required=True)
@click.option('--size', default=100)
@click.option('--start', required=True, help='Start date YYYY-MM-DD')
@click.option('--end', required=True, help='End date YYYY-MM-DD')
@click.option('--cloud-threshold', default=20)
def analyze(center, size, start, end, cloud_threshold):
    """Download and cache imagery pair for analysis."""
    
    lat, lon = map(float, center.split(','))
    bbox = create_bbox(lat, lon, size)
    
    # Query GEE
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(bbox) \
        .filterDate(start, end) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
    
    # Select best pair
    img_a, img_b = select_best_image_pair(collection, parse_date(start), parse_date(end))
    
    # Initialize cache
    cache = ImageCache()
    
    # Download both images
    click.echo("Fetching Date A imagery...")
    arrays_a, meta_a, hit_a = fetch_image_with_cache(img_a, date_a, bbox, ['B4', 'B3', 'B8'], cache)
    
    click.echo("Fetching Date B imagery...")
    arrays_b, meta_b, hit_b = fetch_image_with_cache(img_b, date_b, bbox, ['B4', 'B3', 'B8'], cache)
    
    click.echo(f"Download complete. Cache stats: {cache.stats()}")
    
    # Store for next phase (change detection)
    result = {
        'date_a': {'arrays': arrays_a, 'metadata': meta_a},
        'date_b': {'arrays': arrays_b, 'metadata': meta_b}
    }
    
    return result
```

**Validation Criteria**:
- [ ] First run downloads from GEE (cache miss)
- [ ] Second run with same parameters loads from cache (hit)
- [ ] Cache size stays under 5GB with automatic eviction
- [ ] Download completes in <30 seconds for 100x100 pixel area
- [ ] `satchange cache status` shows accurate statistics

**Required Packages** (Phase 2):
```
rasterio==1.3.9
requests==2.31.0
diskcache==5.6.3
```

---

### Phase 3: Change Detection - Spectral Analysis

**Objective**: Compute spectral indices and generate change masks from image pairs

**Deliverables**:
1. Spectral index calculation functions (NDVI, NDWI, NDBI)
2. Change detection algorithm with configurable thresholds
3. Change classification by type (vegetation, water, urban)
4. Statistics computation (percent changed, breakdown by category)

**Detailed Tasks**:

**Task 3.1: Spectral Index Calculators**
```python
import numpy as np

def calculate_ndvi(red_band, nir_band):
    """
    Calculate Normalized Difference Vegetation Index.
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Args:
        red_band: numpy array (B4 from Sentinel-2)
        nir_band: numpy array (B8 from Sentinel-2)
        
    Returns:
        numpy array: NDVI values in range [-1, 1]
    """
    # Add small epsilon to avoid division by zero
    numerator = nir_band.astype(float) - red_band.astype(float)
    denominator = nir_band.astype(float) + red_band.astype(float) + 1e-10
    
    ndvi = numerator / denominator
    
    # Clip to valid range
    ndvi = np.clip(ndvi, -1, 1)
    
    return ndvi

def calculate_ndwi(green_band, nir_band):
    """
    Calculate Normalized Difference Water Index.
    
    NDWI = (Green - NIR) / (Green + NIR)
    
    Args:
        green_band: numpy array (B3 from Sentinel-2)
        nir_band: numpy array (B8 from Sentinel-2)
        
    Returns:
        numpy array: NDWI values in range [-1, 1]
    """
    numerator = green_band.astype(float) - nir_band.astype(float)
    denominator = green_band.astype(float) + nir_band.astype(float) + 1e-10
    
    ndwi = numerator / denominator
    ndwi = np.clip(ndwi, -1, 1)
    
    return ndwi

def calculate_ndbi(swir_band, nir_band):
    """
    Calculate Normalized Difference Built-up Index.
    
    NDBI = (SWIR - NIR) / (SWIR + NIR)
    
    Args:
        swir_band: numpy array (B11 from Sentinel-2, 20m resolution)
        nir_band: numpy array (B8 from Sentinel-2)
        
    Returns:
        numpy array: NDBI values in range [-1, 1]
        
    Note: SWIR requires resampling to 10m to match NIR resolution
    """
    numerator = swir_band.astype(float) - nir_band.astype(float)
    denominator = swir_band.astype(float) + nir_band.astype(float) + 1e-10
    
    ndbi = numerator / denominator
    ndbi = np.clip(ndbi, -1, 1)
    
    return ndbi
```

**Task 3.2: Change Detection Core Algorithm**
```python
class ChangeDetector:
    """Detect changes between two satellite images using spectral indices."""
    
    def __init__(self, threshold=0.2):
        """
        Args:
            threshold: Minimum absolute difference to consider as significant change
        """
        self.threshold = threshold
    
    def detect_vegetation_change(self, bands_a, bands_b):
        """
        Detect vegetation changes using NDVI differencing.
        
        Args:
            bands_a: dict with keys 'B4' (red), 'B8' (nir) for Date A
            bands_b: dict with keys 'B4', 'B8' for Date B
            
        Returns:
            dict: {
                'ndvi_a': NDVI array for Date A,
                'ndvi_b': NDVI array for Date B,
                'delta': Difference array,
                'change_mask': Binary mask of significant changes,
                'growth_mask': Mask of vegetation growth areas,
                'loss_mask': Mask of vegetation loss areas
            }
        """
        # Calculate NDVI for both dates
        ndvi_a = calculate_ndvi(bands_a['B4'], bands_a['B8'])
        ndvi_b = calculate_ndvi(bands_b['B4'], bands_b['B8'])
        
        # Compute difference
        delta = ndvi_b - ndvi_a
        
        # Generate masks
        change_mask = np.abs(delta) > self.threshold
        growth_mask = delta > self.threshold  # Positive change = vegetation growth
        loss_mask = delta < -self.threshold   # Negative change = vegetation loss
        
        return {
            'ndvi_a': ndvi_a,
            'ndvi_b': ndvi_b,
            'delta': delta,
            'change_mask': change_mask,
            'growth_mask': growth_mask,
            'loss_mask': loss_mask
        }
    
    def detect_water_change(self, bands_a, bands_b):
        """
        Detect water body changes using NDWI differencing.
        
        Args:
            bands_a: dict with keys 'B3' (green), 'B8' (nir)
            bands_b: dict with keys 'B3', 'B8'
            
        Returns:
            dict: Similar structure to detect_vegetation_change
        """
        ndwi_a = calculate_ndwi(bands_a['B3'], bands_a['B8'])
        ndwi_b = calculate_ndwi(bands_b['B3'], bands_b['B8'])
        
        delta = ndwi_b - ndwi_a
        
        change_mask = np.abs(delta) > self.threshold
        expansion_mask = delta > self.threshold  # Water expansion/flooding
        reduction_mask = delta < -self.threshold # Drought/water loss
        
        return {
            'ndwi_a': ndwi_a,
            'ndwi_b': ndwi_b,
            'delta': delta,
            'change_mask': change_mask,
            'expansion_mask': expansion_mask,
            'reduction_mask': reduction_mask
        }
    
    def detect_all_changes(self, bands_a, bands_b):
        """
        Run all change detection algorithms and combine results.
        
        Returns:
            dict: Combined results from all detectors
        """
        veg_changes = self.detect_vegetation_change(bands_a, bands_b)
        water_changes = self.detect_water_change(bands_a, bands_b)
        
        # Combine change masks
        combined_mask = veg_changes['change_mask'] | water_changes['change_mask']
        
        return {
            'vegetation': veg_changes,
            'water': water_changes,
            'combined_mask': combined_mask
        }
```

**Task 3.3: Change Classification**
```python
def classify_changes(change_results):
    """
    Classify detected changes into categorical types.
    
    Args:
        change_results: Output from ChangeDetector.detect_all_changes()
        
    Returns:
        numpy array: Integer classification map where:
            0 = No change
            1 = Vegetation growth
            2 = Vegetation loss (deforestation)
            3 = Water expansion (flooding)
            4 = Water reduction (drought)
            5 = Ambiguous change (multiple indices triggered)
    """
    height, width = change_results['combined_mask'].shape
    classification = np.zeros((height, width), dtype=np.uint8)
    
    # Priority order: Assign most confident classification first
    
    # Vegetation growth
    classification[change_results['vegetation']['growth_mask']] = 1
    
    # Vegetation loss
    classification[change_results['vegetation']['loss_mask']] = 2
    
    # Water expansion
    classification[change_results['water']['expansion_mask']] = 3
    
    # Water reduction
    classification[change_results['water']['reduction_mask']] = 4
    
    # Ambiguous: Multiple change types at same pixel
    veg_mask = change_results['vegetation']['change_mask']
    water_mask = change_results['water']['change_mask']
    ambiguous = veg_mask & water_mask
    classification[ambiguous] = 5
    
    return classification
```

**Task 3.4: Statistics Computation**
```python
def compute_change_statistics(classification, pixel_area_m2=100):
    """
    Calculate summary statistics from classification map.
    
    Args:
        classification: Integer classification array from classify_changes()
        pixel_area_m2: Area per pixel in square meters (10m resolution = 100m²)
        
    Returns:
        dict: Statistics with counts, percentages, and areas
    """
    total_pixels = classification.size
    
    # Count pixels by class
    class_names = {
        0: 'no_change',
        1: 'vegetation_growth',
        2: 'vegetation_loss',
        3: 'water_expansion',
        4: 'water_reduction',
        5: 'ambiguous'
    }
    
    stats = {}
    for class_id, class_name in class_names.items():
        count = np.sum(classification == class_id)
        percent = (count / total_pixels) * 100
        area_km2 = (count * pixel_area_m2) / 1e6
        
        stats[class_name] = {
            'pixels': int(count),
            'percent': round(percent, 2),
            'area_km2': round(area_km2, 4)
        }
    
    # Total change percentage (all non-zero classes)
    changed_pixels = np.sum(classification > 0)
    stats['total_change'] = {
        'pixels': int(changed_pixels),
        'percent': round((changed_pixels / total_pixels) * 100, 2),
        'area_km2': round((changed_pixels * pixel_area_m2) / 1e6, 4)
    }
    
    return stats
```

**Task 3.5: Integrate Change Detection into Analyze Command**
```python
@click.command()
@click.option('--center', required=True)
@click.option('--size', default=100)
@click.option('--start', required=True)
@click.option('--end', required=True)
@click.option('--change-type', 
              type=click.Choice(['vegetation', 'water', 'all']),
              default='all')
@click.option('--threshold', default=0.2, help='Change detection threshold')
@click.option('--output', required=True, help='Output directory path')
def analyze(center, size, start, end, change_type, threshold, output):
    """Run complete change detection analysis."""
    
    # Phase 2 code: Download imagery
    lat, lon = map(float, center.split(','))
    result = fetch_imagery_pair(lat, lon, size, start, end)
    
    bands_a = result['date_a']['arrays']
    bands_b = result['date_b']['arrays']
    
    # Phase 3: Change detection
    click.echo("Running change detection...")
    detector = ChangeDetector(threshold=threshold)
    
    if change_type == 'vegetation':
        changes = detector.detect_vegetation_change(bands_a, bands_b)
    elif change_type == 'water':
        changes = detector.detect_water_change(bands_a, bands_b)
    else:  # 'all'
        changes = detector.detect_all_changes(bands_a, bands_b)
    
    # Classify changes
    classification = classify_changes(changes)
    
    # Compute statistics
    stats = compute_change_statistics(classification)
    
    # Save intermediate results for Phase 4 (visualization)
    os.makedirs(output, exist_ok=True)
    np.save(os.path.join(output, 'classification.npy'), classification)
    np.save(os.path.join(output, 'bands_a.npy'), bands_a)
    np.save(os.path.join(output, 'bands_b.npy'), bands_b)
    
    with open(os.path.join(output, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Display summary
    click.echo("\n=== Change Detection Results ===")
    click.echo(f"Total area changed: {stats['total_change']['percent']}%")
    click.echo(f"Vegetation loss: {stats['vegetation_loss']['percent']}%")
    click.echo(f"Vegetation growth: {stats['vegetation_growth']['percent']}%")
    click.echo(f"Water changes: {stats['water_expansion']['percent'] + stats['water_reduction']['percent']}%")
    click.echo(f"\nResults saved to: {output}/")
```

**Validation Criteria**:
- [ ] NDVI correctly differentiates forest (>0.6) from urban (<0.2)
- [ ] Known deforestation case (e.g., Amazon 2019-2020) shows negative NDVI delta
- [ ] Water body detection works on coastal/lake areas
- [ ] Statistics sum to 100% (all pixels classified)
- [ ] Threshold adjustment changes detection sensitivity as expected

**Required Packages** (Phase 3):
```
numpy==1.24.3
scikit-image==0.21.0  # For additional image processing utilities
```

---

### Phase 4: Visualization - Embossing & Interactive Output

**Objective**: Generate embossed visual effects and interactive HTML viewers for detected changes

**Deliverables**:
1. Emboss filter implementation
2. Static PNG output with color-coded changes
3. Interactive HTML viewer with before/after toggle
4. GeoTIFF export with spatial metadata

**Detailed Tasks**:

**Task 4.1: Emboss Filter Implementation**
```python
import cv2

def apply_emboss_effect(change_mask, intensity=1.0):
    """
    Apply emboss filter to change mask for 3D visual effect.
    
    Args:
        change_mask: Binary numpy array (0 = no change, 1 = change)
        intensity: Emboss strength multiplier (0.0 to 2.0)
        
    Returns:
        numpy array: Embossed image (0-255 range)
    """
    # Convert binary mask to uint8 for CV operations
    mask_uint8 = (change_mask * 255).astype(np.uint8)
    
    # Define emboss kernel (emphasizes edges in NE-SW direction)
    emboss_kernel = np.array([
        [-2, -1,  0],
        [-1,  1,  1],
        [ 0,  1,  2]
    ], dtype=np.float32)
    
    # Apply convolution
    embossed = cv2.filter2D(mask_uint8, cv2.CV_32F, emboss_kernel)
    
    # Normalize to 0-255 range
    embossed = cv2.normalize(embossed, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply intensity multiplier
    embossed = np.clip(embossed * intensity, 0, 255).astype(np.uint8)
    
    # Add slight blur to smooth artifacts
    embossed = cv2.GaussianBlur(embossed, (3, 3), 0)
    
    return embossed

def create_color_coded_overlay(classification, embossed):
    """
    Create RGBA overlay with color-coded change types.
    
    Args:
        classification: Integer classification map
        embossed: Embossed change mask (0-255)
        
    Returns:
        numpy array: RGBA image (H, W, 4)
    """
    height, width = classification.shape
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Color mapping (R, G, B, A)
    colors = {
        1: (0, 255, 0, 180),      # Vegetation growth - Green
        2: (255, 0, 0, 180),      # Vegetation loss - Red
        3: (0, 100, 255, 180),    # Water expansion - Blue
        4: (255, 165, 0, 180),    # Water reduction - Orange
        5: (128, 128, 128, 180)   # Ambiguous - Gray
    }
    
    # Apply colors based on classification
    for class_id, color in colors.items():
        mask = classification == class_id
        overlay[mask] = color
    
    # Modulate alpha channel by emboss intensity (creates depth)
    emboss_alpha = (embossed / 255.0) * 180
    for class_id in colors.keys():
        mask = classification == class_id
        overlay[mask, 3] = emboss_alpha[mask]
    
    return overlay
```

**Task 4.2: Static PNG Generation**
```python
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def generate_static_png(bands_b, classification, embossed, output_path):
    """
    Generate static PNG with embossed overlay on base imagery.
    
    Args:
        bands_b: Date B RGB bands dict {'B4': red, 'B3': green, 'B2': blue}
        classification: Classification map
        embossed: Embossed mask
        output_path: Path to save PNG
    """
    # Create RGB composite from Date B (most recent imagery)
    rgb = np.dstack([
        bands_b['B4'],  # Red
        bands_b['B3'],  # Green
        bands_b['B2']   # Blue (need to download B2 in Phase 2)
    ])
    
    # Normalize to 0-255 for display
    rgb_norm = np.zeros_like(rgb)
    for i in range(3):
        band = rgb[:, :, i]
        rgb_norm[:, :, i] = ((band - band.min()) / (band.max() - band.min()) * 255).astype(np.uint8)
    
    # Create overlay
    overlay = create_color_coded_overlay(classification, embossed)
    
    # Composite: blend overlay with base image
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Base imagery
    axes[0].imshow(rgb_norm)
    axes[0].set_title('Date B - Current State', fontsize=14)
    axes[0].axis('off')
    
    # Panel 2: Change mask with emboss
    axes[1].imshow(embossed, cmap='gray')
    axes[1].set_title('Detected Changes (Embossed)', fontsize=14)
    axes[1].axis('off')
    
    # Panel 3: Composite
    axes[2].imshow(rgb_norm)
    axes[2].imshow(overlay, interpolation='bilinear')
    axes[2].set_title('Changes Overlay', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    click.echo(f"Static visualization saved: {output_path}")
```

**Task 4.3: Interactive HTML Viewer**
```python
from jinja2 import Template
import base64
from io import BytesIO
from PIL import Image

def array_to_base64(array):
    """Convert numpy array to base64-encoded PNG data URI."""
    # Normalize to 0-255 if needed
    if array.max() <= 1.0:
        array = (array * 255).astype(np.uint8)
    
    # Convert to PIL Image
    if len(array.shape) == 2:  # Grayscale
        img = Image.fromarray(array)
    else:  # RGB/RGBA
        img = Image.fromarray(array)
    
    # Encode as PNG in memory
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Convert to base64
    img_base64 = base64.b64encode(buffer.read()).decode()
    
    return f"data:image/png;base64,{img_base64}"

def generate_interactive_html(bands_a, bands_b, classification, embossed, 
                              stats, center_lat, center_lon, output_path):
    """
    Generate interactive HTML viewer with Leaflet.js.
    
    Args:
        bands_a: Date A RGB bands
        bands_b: Date B RGB bands
        classification: Classification map
        embossed: Embossed mask
        stats: Statistics dict from compute_change_statistics()
        center_lat: AOI center latitude
        center_lon: AOI center longitude
        output_path: Path to save HTML file
    """
    # Create image composites
    rgb_a = create_rgb_composite(bands_a)
    rgb_b = create_rgb_composite(bands_b)
    overlay = create_color_coded_overlay(classification, embossed)
    
    # Composite overlay on base image
    composite = rgb_b.copy()
    # Alpha blend
    alpha = overlay[:, :, 3:4] / 255.0
    composite = (composite * (1 - alpha) + overlay[:, :, :3] * alpha).astype(np.uint8)
    
    # Convert to base64
    img_a_uri = array_to_base64(rgb_a)
    img_b_uri = array_to_base64(rgb_b)
    img_overlay_uri = array_to_base64(composite)
    
    # HTML template
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SatChange Analysis Results</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body { margin: 0; font-family: Arial, sans-serif; }
        #map { height: 70vh; width: 100%; }
        #controls { 
            padding: 20px; 
            background: #f0f0f0; 
            display: flex; 
            justify-content: space-around;
            align-items: center;
        }
        button { 
            padding: 12px 24px; 
            font-size: 16px; 
            cursor: pointer;
            border: none;
            border-radius: 4px;
            background: #4CAF50;
            color: white;
        }
        button:hover { background: #45a049; }
        button.active { background: #2196F3; }
        #stats {
            padding: 20px;
            background: #fff;
            border-top: 2px solid #ddd;
        }
        .stat-item { 
            display: inline-block; 
            margin-right: 30px; 
            padding: 10px;
        }
        .stat-value { 
            font-size: 24px; 
            font-weight: bold; 
            color: #2196F3;
        }
        .stat-label { 
            font-size: 14px; 
            color: #666;
        }
    </style>
</head>
<body>
    <div id="controls">
        <h2>SatChange Analysis</h2>
        <div>
            <button id="btn-before" onclick="showLayer('before')">{{ date_a }}</button>
            <button id="btn-after" onclick="showLayer('after')">{{ date_b }}</button>
            <button id="btn-changes" class="active" onclick="showLayer('changes')">Changes</button>
        </div>
    </div>
    
    <div id="map"></div>
    
    <div id="stats">
        <h3>Change Statistics</h3>
        <div class="stat-item">
            <div class="stat-value">{{ stats.total_change.percent }}%</div>
            <div class="stat-label">Total Area Changed</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{{ stats.vegetation_loss.percent }}%</div>
            <div class="stat-label">Vegetation Loss</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{{ stats.vegetation_growth.percent }}%</div>
            <div class="stat-label">Vegetation Growth</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{{ stats.water_expansion.percent }}%</div>
            <div class="stat-label">Water Expansion</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{{ stats.total_change.area_km2 }} km²</div>
            <div class="stat-label">Changed Area</div>
        </div>
    </div>
    
    <script>
        // Initialize map
        const map = L.map('map').setView([{{ center_lat }}, {{ center_lon }}], 13);
        
        // Add base tile layer
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);
        
        // Calculate bounds (approximate)
        const latOffset = 0.01;  // ~1km at equator
        const lonOffset = 0.01;
        const bounds = [
            [{{ center_lat }} - latOffset, {{ center_lon }} - lonOffset],
            [{{ center_lat }} + latOffset, {{ center_lon }} + lonOffset]
        ];
        
        // Create image overlays
        const layerBefore = L.imageOverlay('{{ img_a_uri }}', bounds);
        const layerAfter = L.imageOverlay('{{ img_b_uri }}', bounds);
        const layerChanges = L.imageOverlay('{{ img_overlay_uri }}', bounds);
        
        // Add changes layer by default
        layerChanges.addTo(map);
        
        // Layer control functions
        let currentLayer = 'changes';
        
        function showLayer(layerName) {
            // Remove all layers
            map.removeLayer(layerBefore);
            map.removeLayer(layerAfter);
            map.removeLayer(layerChanges);
            
            // Add selected layer
            if (layerName === 'before') {
                layerBefore.addTo(map);
            } else if (layerName === 'after') {
                layerAfter.addTo(map);
            } else {
                layerChanges.addTo(map);
            }
            
            // Update button states
            document.querySelectorAll('button').forEach(btn => btn.classList.remove('active'));
            document.getElementById('btn-' + layerName).classList.add('active');
            
            currentLayer = layerName;
        }
        
        // Fit map to overlay bounds
        map.fitBounds(bounds);
    </script>
</body>
</html>
    """
    
    # Render template
    template = Template(html_template)
    html_content = template.render(
        img_a_uri=img_a_uri,
        img_b_uri=img_b_uri,
        img_overlay_uri=img_overlay_uri,
        center_lat=center_lat,
        center_lon=center_lon,
        date_a="Date A",  # Replace with actual dates from metadata
        date_b="Date B",
        stats=stats
    )
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    click.echo(f"Interactive viewer saved: {output_path}")
```

**Task 4.4: GeoTIFF Export with Spatial Metadata**
```python
import rasterio
from rasterio.transform import from_bounds

def export_geotiff(classification, metadata, output_path):
    """
    Export classification map as GeoTIFF with spatial reference.
    
    Args:
        classification: Classification array
        metadata: Spatial metadata from Phase 2 download
        output_path: Path to save GeoTIFF
    """
    height, width = classification.shape
    
    # Create rasterio dataset
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=classification.dtype,
        crs=metadata['crs'],
        transform=metadata['transform'],
        compress='lzw'
    ) as dst:
        dst.write(classification, 1)
        
        # Add metadata tags
        dst.update_tags(
            1,
            change_classes='0:no_change,1:veg_growth,2:veg_loss,3:water_expand,4:water_reduce'
        )
    
    click.echo(f"GeoTIFF saved: {output_path}")
```

**Task 4.5: Implement `satchange export` Command**
```python
@click.command()
@click.option('--result', required=True, help='Path to analysis result directory')
@click.option('--format', 
              type=click.Choice(['static', 'interactive', 'geotiff', 'all']),
              default='all')
@click.option('--emboss-intensity', default=1.0, help='Emboss effect strength (0-2)')
def export(result, format, emboss_intensity):
    """Generate visualization outputs from analysis results."""
    
    # Load saved results from Phase 3
    classification = np.load(os.path.join(result, 'classification.npy'))
    bands_a = np.load(os.path.join(result, 'bands_a.npy'), allow_pickle=True).item()
    bands_b = np.load(os.path.join(result, 'bands_b.npy'), allow_pickle=True).item()
    
    with open(os.path.join(result, 'stats.json'), 'r') as f:
        stats = json.load(f)
    
    # Apply emboss effect
    change_mask = classification > 0
    embossed = apply_emboss_effect(change_mask, intensity=emboss_intensity)
    
    # Generate requested formats
    if format in ['static', 'all']:
        output_path = os.path.join(result, 'visualization_static.png')
        generate_static_png(bands_b, classification, embossed, output_path)
    
    if format in ['interactive', 'all']:
        output_path = os.path.join(result, 'visualization_interactive.html')
        generate_interactive_html(bands_a, bands_b, classification, embossed,
                                 stats, center_lat, center_lon, output_path)
    
    if format in ['geotiff', 'all']:
        output_path = os.path.join(result, 'classification.tif')
        export_geotiff(classification, metadata, output_path)
    
    click.echo("\nExport complete!")
```

**Validation Criteria**:
- [ ] Emboss effect creates visible 3D depth perception
- [ ] Color coding matches change types correctly (red=loss, green=growth)
- [ ] Interactive HTML toggles between layers without errors
- [ ] GeoTIFF opens correctly in QGIS with proper coordinates
- [ ] Static PNG shows clear before/embossed/after comparison

**Required Packages** (Phase 4):
```
opencv-python-headless==4.8.1.78
matplotlib==3.8.0
jinja2==3.1.2
pillow==10.0.1  # For image encoding
```

---

### Phase 5: Polish & Production Readiness

**Objective**: Handle edge cases, improve UX, and create documentation

**Deliverables**:
1. Cloud coverage fallback mechanisms
2. Progress indicators for long operations
3. Comprehensive error handling
4. User documentation and examples

**Detailed Tasks**:

**Task 5.1: Cloud Coverage Handling**
```python
def handle_cloudy_scenes(collection, bbox, start_date, end_date, max_cloud_threshold=20):
    """
    Handle cases where no clear scenes available.
    
    Strategies:
    1. Gradually increase cloud threshold (20% → 40% → 60%)
    2. Expand temporal window (±30 days → ±60 days)
    3. Offer temporal compositing (median of multiple scenes)
    
    Returns:
        tuple: (img_a, img_b) or None if no usable imagery
    """
    # Try increasing cloud thresholds
    for threshold in [max_cloud_threshold, 40, 60]:
        filtered = collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', threshold))
        count = filtered.size().getInfo()
        
        if count >= 2:
            click.echo(f"Found {count} scenes with <{threshold}% clouds")
            return select_best_image_pair(filtered, start_date, end_date)
    
    # If still no luck: Temporal compositing
    click.echo("No clear single scenes found. Attempting temporal composite...")
    
    # Get clearest 5 scenes for each time period
    scenes_a = collection \
        .filterDate(start_date, start_date + timedelta(days