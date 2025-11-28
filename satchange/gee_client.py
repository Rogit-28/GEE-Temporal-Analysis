"""
Google Earth Engine client for SatChange.

This module provides the interface for interacting with Google Earth Engine API,
including authentication, image queries, and data retrieval.
"""

import ee
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json

from .config import Config

logger = logging.getLogger(__name__)


class GEEError(Exception):
    """Base exception for GEE-related errors."""

    pass


class AuthenticationError(GEEError):
    """Exception raised for GEE authentication failures."""

    pass


class QuotaExceededError(GEEError):
    """Exception raised when GEE compute quota is exceeded."""

    pass


class RateLimitError(GEEError):
    """Exception raised when GEE rate limit is hit."""

    pass


class NoImageryError(GEEError):
    """Exception raised when no suitable imagery is found."""

    pass


class DownloadError(GEEError):
    """Exception raised when image download fails."""

    pass


class GEEClient:
    """Google Earth Engine client for SatChange."""

    def __init__(self, config: Config):
        """Initialize GEE client.

        Args:
            config: Configuration instance
        """
        self.config = config
        self._authenticated = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize GEE connection if authenticated."""
        if self.config.is_authenticated():
            self.authenticate()

    def authenticate(self) -> bool:
        """Authenticate with Google Earth Engine.

        Returns:
            True if authentication successful, False otherwise

        Raises:
            AuthenticationError: If authentication fails
        """
        try:
            service_account_key = self.config.get("service_account_key")
            project_id = self.config.get("project_id")

            if not service_account_key or not project_id:
                raise AuthenticationError("Service account key and project ID required")

            # Initialize GEE with service account credentials
            credentials = ee.ServiceAccountCredentials(
                email=self.config.get("service_account_email"),
                key_file=service_account_key,
            )

            ee.Initialize(credentials, project=project_id)

            # Test authentication with a simple collection query
            test_collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").limit(1)
            test_collection.size().getInfo()  # This will raise exception if auth failed

            self._authenticated = True
            logger.info("Successfully authenticated with Google Earth Engine")
            return True

        except Exception as e:
            self._authenticated = False
            error_msg = f"Authentication failed: {e}"
            logger.error(error_msg)
            raise AuthenticationError(error_msg)

    def is_authenticated(self) -> bool:
        """Check if authenticated with GEE.

        Returns:
            True if authenticated, False otherwise
        """
        return self._authenticated

    def create_bbox(
        self,
        center_lat: float,
        center_lon: float,
        pixel_size: int,
        resolution_meters: int = 10,
    ) -> ee.Geometry.Polygon:
        """Convert center point and pixel dimensions to geographic bounding box.

        Args:
            center_lat: Latitude of AOI center
            center_lon: Longitude of AOI center
            pixel_size: Number of pixels per side (e.g., 100 for 100x100)
            resolution_meters: Spatial resolution (Sentinel-2 = 10m)

        Returns:
            ee.Geometry.Polygon: Bounding box for GEE query
        """
        from geopy.distance import distance

        # Calculate AOI dimensions in meters
        width_meters = pixel_size * resolution_meters
        height_meters = pixel_size * resolution_meters

        # Calculate bounding box corners using geopy for accurate distance calculations
        center_point = (center_lat, center_lon)

        # Calculate corners
        north = distance(meters=height_meters / 2).destination(center_point, 0)
        south = distance(meters=height_meters / 2).destination(center_point, 180)
        east = distance(meters=width_meters / 2).destination(center_point, 90)
        west = distance(meters=width_meters / 2).destination(center_point, 270)

        # Create polygon coordinates
        coords = [
            [west.longitude, south.latitude],
            [east.longitude, south.latitude],
            [east.longitude, north.latitude],
            [west.longitude, north.latitude],
        ]

        return ee.Geometry.Polygon(coords)

    def query_imagery(
        self,
        bbox: ee.Geometry.Polygon,
        start_date: datetime,
        end_date: datetime,
        cloud_threshold: int = 20,
    ) -> ee.ImageCollection:
        """Query Sentinel-2 imagery for given area and date range.

        Args:
            bbox: Bounding box geometry
            start_date: Start date for query
            end_date: End date for query
            cloud_threshold: Maximum cloud coverage percentage

        Returns:
            ee.ImageCollection: Filtered image collection
        """
        # Sentinel-2 Surface Reflectance collection
        collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

        # Apply filters
        filtered = (
            collection.filterBounds(bbox)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
        )

        logger.info(f"Query found {filtered.size().getInfo()} images")
        return filtered

    def get_scenes_metadata(
        self, collection: ee.ImageCollection
    ) -> List[Dict[str, Any]]:
        """Get metadata for scenes in collection.

        Args:
            collection: GEE ImageCollection

        Returns:
            List of scene metadata dictionaries
        """
        try:
            # Get collection info
            collection_info = collection.getInfo()

            if "features" not in collection_info:
                return []

            scenes = []
            for feature in collection_info["features"]:
                properties = feature.get("properties", {})

                # Parse date from system:time_start (milliseconds epoch)
                time_start = properties.get("system:time_start")
                if time_start:
                    try:
                        date_val = datetime.fromtimestamp(time_start / 1000).strftime(
                            "%Y-%m-%d"
                        )
                    except (ValueError, OSError, OverflowError):
                        date_val = "Unknown"
                else:
                    date_val = properties.get("SENSING_TIME", "Unknown")

                # Extract relevant metadata
                scene_info = {
                    "id": feature.get("id"),
                    "date": date_val,
                    "cloud_coverage": properties.get("CLOUDY_PIXEL_PERCENTAGE", 0),
                    "tile_id": properties.get("MGRS_TILE", "Unknown"),
                    "product_id": properties.get("PRODUCT_ID", "Unknown"),
                }

                scenes.append(scene_info)

            return scenes

        except Exception as e:
            logger.error(f"Failed to get scenes metadata: {e}")
            return []

    def select_best_image_pair(
        self,
        collection: ee.ImageCollection,
        start_date: datetime,
        end_date: datetime,
        cloud_threshold: int = None,
    ) -> Tuple[ee.Image, ee.Image]:
        """Select optimal image pair from collection.

        Selects the clearest image pair that meets the cloud coverage threshold.
        If a candidate exceeds the threshold, it is rejected and the next
        clearest candidate is tried.

        Args:
            collection: ee.ImageCollection (filtered by AOI and date range)
            start_date: Target start date
            end_date: Target end date
            cloud_threshold: Maximum allowed cloud coverage percentage.
                           If None, uses config value or defaults to 20.

        Returns:
            Tuple of (date_a_image, date_b_image) as ee.Image objects

        Raises:
            ValueError: If no valid image pair can be found within thresholds
        """
        from datetime import timedelta

        # Get cloud threshold from config if not provided
        if cloud_threshold is None:
            cloud_threshold = self.config.get("cloud_threshold", 20)

        # Convert collection to sorted list (by cloud coverage, ascending)
        scenes = collection.sort("CLOUDY_PIXEL_PERCENTAGE").getInfo()["features"]

        if not scenes:
            raise ValueError("No scenes found in collection")

        # Parse dates from scene properties
        def parse_date(scene):
            # Try system:time_start first (milliseconds epoch)
            time_start = scene["properties"].get("system:time_start")
            if time_start:
                try:
                    return datetime.fromtimestamp(time_start / 1000)
                except (ValueError, OSError, OverflowError):
                    pass
            # Fallback to SENSING_TIME string
            date_str = scene["properties"].get("SENSING_TIME", "")
            try:
                return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                return datetime.min

        def get_cloud_coverage(scene):
            """Get cloud coverage percentage from scene properties."""
            return scene["properties"].get("CLOUDY_PIXEL_PERCENTAGE", 0)

        def find_valid_scene(candidates, label):
            """Find the first scene that meets cloud threshold.

            Args:
                candidates: List of candidate scenes sorted by cloud coverage
                label: Label for logging (e.g., 'Date A', 'Date B')

            Returns:
                Valid scene dict or None if no valid scene found
            """
            for scene in candidates:
                cloud_pct = get_cloud_coverage(scene)
                scene_date = parse_date(scene)
                scene_id = scene.get("id", "unknown")

                if cloud_pct <= cloud_threshold:
                    logger.info(
                        f"{label}: Selected scene {scene_id} "
                        f"(date: {scene_date.date()}, cloud: {cloud_pct:.1f}%)"
                    )
                    return scene
                else:
                    logger.warning(
                        f"{label}: Rejecting scene {scene_id} - "
                        f"cloud coverage {cloud_pct:.1f}% exceeds threshold {cloud_threshold}%"
                    )
            return None

        # Find Date A (near start)
        date_a_window = [start_date, start_date + timedelta(days=30)]
        date_a_candidates = [
            s for s in scenes if date_a_window[0] <= parse_date(s) <= date_a_window[1]
        ]

        if not date_a_candidates:
            # Expand window
            date_a_window[1] = start_date + timedelta(days=90)
            date_a_candidates = [
                s
                for s in scenes
                if date_a_window[0] <= parse_date(s) <= date_a_window[1]
            ]

        if not date_a_candidates:
            raise ValueError(f"No scenes found within {start_date} ± 90 days")

        # Find valid Date A scene (reject cloudy scenes)
        date_a_scene = find_valid_scene(date_a_candidates, "Date A")

        if date_a_scene is None:
            raise ValueError(
                f"No scenes found for Date A within threshold. "
                f"All {len(date_a_candidates)} candidates exceed {cloud_threshold}% cloud coverage."
            )

        # Find Date B (near end, minimum 6 months after Date A)
        min_gap = parse_date(date_a_scene) + timedelta(days=180)
        date_b_window = [max(min_gap, end_date - timedelta(days=30)), end_date]
        date_b_candidates = [
            s for s in scenes if date_b_window[0] <= parse_date(s) <= date_b_window[1]
        ]

        if not date_b_candidates:
            date_b_window[0] = min_gap
            date_b_candidates = [
                s
                for s in scenes
                if date_b_window[0] <= parse_date(s) <= date_b_window[1]
            ]

        if not date_b_candidates:
            raise ValueError(
                f"No scenes found within {end_date} ± 90 days with minimum 6-month gap"
            )

        # Find valid Date B scene (reject cloudy scenes)
        date_b_scene = find_valid_scene(date_b_candidates, "Date B")

        if date_b_scene is None:
            raise ValueError(
                f"No scenes found for Date B within threshold. "
                f"All {len(date_b_candidates)} candidates exceed {cloud_threshold}% cloud coverage."
            )

        logger.info(
            f"Selected image pair - Date A: {parse_date(date_a_scene).date()}, "
            f"Date B: {parse_date(date_b_scene).date()}"
        )

        return (ee.Image(date_a_scene["id"]), ee.Image(date_b_scene["id"]))

    def download_image(
        self,
        image: ee.Image,
        bbox: ee.Geometry.Polygon,
        bands: List[str] = None,
        scale: int = 10,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Download image bands from GEE as numpy arrays.

        Args:
            image: ee.Image object
            bbox: ee.Geometry.Polygon defining AOI
            bands: List of band names to download
            scale: Resolution in meters (10m for Sentinel-2)

        Returns:
            Tuple of (band_arrays, metadata)
        """
        import requests
        import rasterio
        from io import BytesIO

        if bands is None:
            bands = ["B4", "B3", "B8"]  # Default RGB + NIR

        try:
            # Get thumbnail URL for small areas
            url = image.select(bands).getThumbURL(
                {"region": bbox, "scale": scale, "format": "GEO_TIFF"}
            )

            # Download GeoTIFF with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, timeout=180)
                    response.raise_for_status()
                    break
                except (
                    requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                ) as e:
                    if attempt < max_retries - 1:
                        wait_time = 2**attempt * 5  # 5s, 10s, 20s
                        logger.warning(
                            f"Download attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        raise
                except requests.exceptions.HTTPError as e:
                    if response.status_code == 429:  # Rate limit
                        if attempt < max_retries - 1:
                            wait_time = 2**attempt * 10  # 10s, 20s, 40s
                            logger.warning(
                                f"GEE rate limit hit. Retrying in {wait_time}s..."
                            )
                            time.sleep(wait_time)
                        else:
                            raise RateLimitError(
                                f"GEE rate limit exceeded after {max_retries} retries"
                            )
                    elif response.status_code == 403:
                        raise QuotaExceededError(
                            "GEE compute quota exceeded. Try again later or request quota increase."
                        )
                    else:
                        raise DownloadError(
                            f"Image download failed with HTTP {response.status_code}: {e}"
                        )

            # Parse with rasterio
            with rasterio.open(BytesIO(response.content)) as dataset:
                band_arrays = {
                    band: dataset.read(i + 1) for i, band in enumerate(bands)
                }
                metadata = {
                    "transform": dataset.transform,
                    "crs": dataset.crs,
                    "bounds": dataset.bounds,
                    "width": dataset.width,
                    "height": dataset.height,
                }

            logger.info(f"Downloaded image with bands: {list(band_arrays.keys())}")
            return band_arrays, metadata

        except Exception as e:
            logger.error(f"Failed to download image: {e}")
            raise

    def get_image_info(self, image: ee.Image) -> Dict[str, Any]:
        """Get metadata for a single image.

        Args:
            image: ee.Image object

        Returns:
            Dictionary with image metadata
        """
        try:
            info = image.getInfo()
            properties = info.get("properties", {})

            # Get date from system:time_start (epoch milliseconds)
            time_start = properties.get("system:time_start")
            date_str = None
            if time_start:
                from datetime import datetime, timezone

                date_str = datetime.fromtimestamp(
                    time_start / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d")

            return {
                "id": info.get("id"),
                "date": date_str,
                "cloud_coverage": properties.get("CLOUDY_PIXEL_PERCENTAGE", 0),
                "bands": info.get("bands", []),
                "system:time_start": time_start,
            }

        except Exception as e:
            logger.error(f"Failed to get image info: {e}")
            return {}

    def get_available_bands(self) -> List[str]:
        """Get list of available Sentinel-2 bands.

        Returns:
            List of band names
        """
        try:
            # Get a sample image to get band information
            sample_image = ee.Image(
                "COPERNICUS/S2_SR_HARMONIZED/20200101T000000_20200101T235959_T43PGP"
            )
            info = sample_image.getInfo()

            bands = []
            for band_info in info.get("bands", []):
                bands.append(band_info.get("id"))

            return sorted(bands)

        except Exception as e:
            logger.error(f"Failed to get available bands: {e}")
            return []

    def test_connection(self) -> bool:
        """Test GEE connection with a simple query.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Simple test query
            test_collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").limit(1)
            size = test_collection.size().getInfo()

            logger.info(f"GEE connection test successful, found {size} images")
            return True

        except Exception as e:
            logger.error(f"GEE connection test failed: {e}")
            return False

    def check_local_cloud(
        self,
        date: str,
        center: Tuple[float, float],
        size: int = 150,
        resolution: int = 10,
    ) -> Dict[str, Any]:
        """Check local cloud coverage for a specific date WITHOUT downloading image.

        Uses server-side SCL (Scene Classification Layer) band computation to check
        cloud coverage at the specific area of interest, not just scene-level metadata.

        Args:
            date: Target date in 'YYYY-MM-DD' format
            center: Tuple of (latitude, longitude)
            size: AOI size in pixels (default 150 = 1.5km at 10m resolution)
            resolution: Pixel resolution in meters (default 10)

        Returns:
            Dictionary with:
                - date: The requested date
                - image_id: Full GEE image ID (or None if not found)
                - scene_cloud_pct: Scene-level cloud % from metadata
                - local_cloud_pct: Local cloud % at AOI (computed from SCL)
                - is_good: True if local_cloud_pct < 15%
                - found: True if an image was found for that date
        """
        lat, lon = center

        # Create AOI geometry
        point = ee.Geometry.Point([lon, lat])
        buffer_meters = (size * resolution) / 2  # Half the AOI width
        region = point.buffer(buffer_meters).bounds()

        # Query for images on the specific date
        date_start = date
        date_end_dt = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)
        date_end = date_end_dt.strftime("%Y-%m-%d")

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point)
            .filterDate(date_start, date_end)
        )

        count = collection.size().getInfo()

        if count == 0:
            logger.info(f"No image found for date {date}")
            return {
                "date": date,
                "image_id": None,
                "scene_cloud_pct": None,
                "local_cloud_pct": None,
                "is_good": False,
                "found": False,
            }

        # Get the first (usually only) image for that date
        image = ee.Image(collection.first())
        image_info = image.getInfo()
        image_id = image_info.get("id")
        scene_cloud = image_info.get("properties", {}).get("CLOUDY_PIXEL_PERCENTAGE", 0)

        # Compute local cloud coverage using SCL band
        # SCL values: 3=cloud shadow, 8=cloud medium prob, 9=cloud high prob, 10=cirrus
        scl = image.select("SCL")
        cloud_mask = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))

        stats = cloud_mask.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=20,  # Use 20m for SCL band (native resolution)
            maxPixels=1e6,
        )

        scl_result = stats.get("SCL")
        local_cloud_fraction = scl_result.getInfo() if scl_result is not None else None
        local_cloud_pct = (
            local_cloud_fraction * 100 if local_cloud_fraction is not None else 0
        )

        is_good = local_cloud_pct < 15.0

        logger.info(
            f"Date {date}: scene_cloud={scene_cloud:.1f}%, local_cloud={local_cloud_pct:.1f}%, good={is_good}"
        )

        return {
            "date": date,
            "image_id": image_id,
            "scene_cloud_pct": round(scene_cloud, 1),
            "local_cloud_pct": round(local_cloud_pct, 1),
            "is_good": is_good,
            "found": True,
        }

    def find_alternative_dates(
        self,
        target_date: str,
        center: Tuple[float, float],
        cloud_threshold: float = 15.0,
        size: int = 150,
        resolution: int = 10,
    ) -> Dict[str, Any]:
        """Find alternative dates with good local cloud coverage.

        Performs incremental search: ±2 weeks -> ±1 month -> ±2 months -> ±3 months
        until finding alternatives below the cloud threshold.

        Args:
            target_date: Original target date in 'YYYY-MM-DD' format
            center: Tuple of (latitude, longitude)
            cloud_threshold: Maximum acceptable local cloud % (default 15%)
            size: AOI size in pixels
            resolution: Pixel resolution in meters

        Returns:
            Dictionary with:
                - target_date: The original requested date
                - search_window: The window that found results (e.g., '±1 month')
                - threshold_met: True if alternatives below threshold were found
                - alternatives: List of alternatives sorted by local_cloud_pct, each with:
                    - date, image_id, scene_cloud_pct, local_cloud_pct, is_recommended
        """
        lat, lon = center
        point = ee.Geometry.Point([lon, lat])
        buffer_meters = (size * resolution) / 2
        region = point.buffer(buffer_meters).bounds()

        target_dt = datetime.strptime(target_date, "%Y-%m-%d")

        # Incremental search windows
        search_windows = [
            ("±2 weeks", 14),
            ("±1 month", 30),
            ("±2 months", 60),
            ("±3 months", 90),
        ]

        for window_label, days in search_windows:
            start_dt = target_dt - timedelta(days=days)
            end_dt = target_dt + timedelta(days=days)

            logger.info(
                f"Searching {window_label}: {start_dt.date()} to {end_dt.date()}"
            )

            # Query images in window
            collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(point)
                .filterDate(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
            )  # Pre-filter obvious bad ones

            count = collection.size().getInfo()
            if count == 0:
                continue

            # Get all images and compute local cloud for each
            images_info = collection.getInfo()
            alternatives = []

            for feature in images_info.get("features", []):
                image_id = feature.get("id")
                props = feature.get("properties", {})
                scene_cloud = props.get("CLOUDY_PIXEL_PERCENTAGE", 0)

                # Parse date
                time_start = props.get("system:time_start")
                if time_start:
                    img_date = datetime.fromtimestamp(time_start / 1000).strftime(
                        "%Y-%m-%d"
                    )
                else:
                    continue

                # Compute local cloud coverage
                image = ee.Image(image_id)
                scl = image.select("SCL")
                cloud_mask = (
                    scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
                )

                stats = cloud_mask.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=region, scale=20, maxPixels=1e6
                )

                local_cloud_fraction = stats.get("SCL").getInfo()
                local_cloud_pct = (
                    local_cloud_fraction * 100
                    if local_cloud_fraction is not None
                    else 100
                )

                alternatives.append(
                    {
                        "date": img_date,
                        "image_id": image_id,
                        "scene_cloud_pct": round(scene_cloud, 1),
                        "local_cloud_pct": round(local_cloud_pct, 1),
                        "is_recommended": False,
                    }
                )

            # Sort by local cloud percentage
            alternatives.sort(key=lambda x: x["local_cloud_pct"])

            # Check if any meet threshold
            good_alternatives = [
                a for a in alternatives if a["local_cloud_pct"] < cloud_threshold
            ]

            if good_alternatives:
                # Mark the best one as recommended
                alternatives[0]["is_recommended"] = True

                return {
                    "target_date": target_date,
                    "search_window": window_label,
                    "threshold_met": True,
                    "alternatives": alternatives[:10],  # Return top 10
                }

        # No good alternatives found within ±3 months
        # Return best available anyway with threshold_met=False
        logger.warning(
            f"No alternatives below {cloud_threshold}% found within ±3 months"
        )

        # Do a final search to get the best available
        start_dt = target_dt - timedelta(days=90)
        end_dt = target_dt + timedelta(days=90)

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point)
            .filterDate(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
            .sort("CLOUDY_PIXEL_PERCENTAGE")
            .limit(10)
        )

        images_info = collection.getInfo()
        alternatives = []

        for feature in images_info.get("features", []):
            image_id = feature.get("id")
            props = feature.get("properties", {})
            scene_cloud = props.get("CLOUDY_PIXEL_PERCENTAGE", 0)

            time_start = props.get("system:time_start")
            if time_start:
                img_date = datetime.fromtimestamp(time_start / 1000).strftime(
                    "%Y-%m-%d"
                )
            else:
                continue

            # Compute local cloud
            image = ee.Image(image_id)
            scl = image.select("SCL")
            cloud_mask = (
                scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
            )

            stats = cloud_mask.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=region, scale=20, maxPixels=1e6
            )

            local_cloud_fraction = stats.get("SCL").getInfo()
            local_cloud_pct = (
                local_cloud_fraction * 100 if local_cloud_fraction is not None else 100
            )

            alternatives.append(
                {
                    "date": img_date,
                    "image_id": image_id,
                    "scene_cloud_pct": round(scene_cloud, 1),
                    "local_cloud_pct": round(local_cloud_pct, 1),
                    "is_recommended": False,
                }
            )

        alternatives.sort(key=lambda x: x["local_cloud_pct"])
        if alternatives:
            alternatives[0]["is_recommended"] = True

        return {
            "target_date": target_date,
            "search_window": "±3 months (no good alternatives)",
            "threshold_met": False,
            "alternatives": alternatives,
        }

    def create_temporal_composite(
        self,
        center: Tuple[float, float],
        target_date: str,
        window_days: int = 90,
        max_scenes: int = 5,
        size: int = 150,
        resolution: int = 10,
    ) -> Optional[ee.Image]:
        """Create median temporal composite from clearest scenes in time window.

        Collects the clearest scenes within the specified window around the target
        date and produces a median composite. This reduces cloud artifacts while
        preserving spatial detail.

        Args:
            center: Tuple of (latitude, longitude)
            target_date: Center date for the compositing window in 'YYYY-MM-DD'
            window_days: Half-window size in days (±)
            max_scenes: Maximum number of scenes to include in composite
            size: AOI size in pixels
            resolution: Pixel resolution in meters

        Returns:
            ee.Image median composite, or None if fewer than 2 scenes available
        """
        lat, lon = center
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")

        start_dt = target_dt - timedelta(days=window_days)
        end_dt = target_dt + timedelta(days=window_days)

        # Create bounding box for spatial filtering
        bbox = self.create_bbox(lat, lon, size, resolution)

        # Query and filter collection: sort by cloud coverage, take clearest
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(bbox)
            .filterDate(
                start_dt.strftime("%Y-%m-%d"),
                end_dt.strftime("%Y-%m-%d"),
            )
            .sort("CLOUDY_PIXEL_PERCENTAGE")
            .limit(max_scenes)
        )

        count = collection.size().getInfo()

        if count < 2:
            logger.warning(
                f"Temporal composite: only {count} scene(s) found in "
                f"±{window_days} day window — need at least 2"
            )
            return None

        logger.info(
            f"Creating temporal composite from {count} clearest scenes "
            f"({start_dt.date()} to {end_dt.date()})"
        )

        return collection.median()

    def handle_cloudy_scenes(
        self,
        date: str,
        center: Tuple[float, float],
        size: int = 150,
        max_cloud_threshold: int = 20,
        resolution: int = 10,
    ) -> Dict[str, Any]:
        """Handle cases where no clear scenes are available using graduated fallback.

        Applies three strategies in sequence until usable imagery is found:

        1. **Graduated threshold** — Relax the cloud-cover limit step by step
           (initial → 40% → 60%) and re-check the target date.
        2. **Expanded temporal window** — Search ±30 / ±60 / ±90 days for an
           alternative date with local cloud coverage under 60%.
        3. **Temporal compositing** — Build a median composite from the five
           clearest scenes within ±90 days.

        Args:
            date: Target date in 'YYYY-MM-DD' format
            center: Tuple of (latitude, longitude)
            size: AOI size in pixels
            max_cloud_threshold: Initial maximum cloud threshold (%)
            resolution: Pixel resolution in meters

        Returns:
            Dictionary with:
                - found: bool — whether usable imagery was obtained
                - strategy_used: str — which fallback strategy succeeded
                - image: ee.Image or None — the usable image (single or composite)
                - image_id: str or None — image ID (None for composites)
                - date: str — image date, or 'composite' for composites
                - local_cloud_pct: float — cloud percentage of the result
                - details: dict — strategy-specific details
        """
        _not_found: Dict[str, Any] = {
            "found": False,
            "strategy_used": "none",
            "image": None,
            "image_id": None,
            "date": date,
            "local_cloud_pct": 100.0,
            "details": {},
        }

        # ------------------------------------------------------------------
        # Strategy 1: Graduated cloud-threshold relaxation on the target date
        # ------------------------------------------------------------------
        thresholds = sorted(set([max_cloud_threshold, 40, 60]))
        logger.info(f"Strategy 1 — Graduated threshold: trying {thresholds} on {date}")

        for threshold in thresholds:
            result = self.check_local_cloud(date, center, size, resolution)

            if not result["found"]:
                # No image exists for this date at all — skip to Strategy 2
                logger.info(f"No image found for {date}; skipping remaining thresholds")
                break

            local_pct = result["local_cloud_pct"]
            if local_pct is not None and local_pct < threshold:
                logger.info(
                    f"Strategy 1 succeeded: {date} has {local_pct:.1f}% local cloud "
                    f"(threshold relaxed to {threshold}%)"
                )
                return {
                    "found": True,
                    "strategy_used": "increased_threshold",
                    "image": ee.Image(result["image_id"]),
                    "image_id": result["image_id"],
                    "date": date,
                    "local_cloud_pct": local_pct,
                    "details": {
                        "original_threshold": max_cloud_threshold,
                        "accepted_threshold": threshold,
                        "scene_cloud_pct": result["scene_cloud_pct"],
                    },
                }

            logger.info(
                f"Threshold {threshold}%: local cloud {local_pct:.1f}% — not under threshold"
            )

        # ------------------------------------------------------------------
        # Strategy 2: Expanded temporal window via find_alternative_dates
        # ------------------------------------------------------------------
        expanded_windows = [30, 60, 90]
        logger.info(
            f"Strategy 2 — Expanded temporal window: trying ±{expanded_windows} days"
        )

        for window_days in expanded_windows:
            # find_alternative_dates uses its own incremental windows, but we
            # call it with a cloud threshold of 60% (the most relaxed
            # single-scene limit we accept) and let it search up to the window.
            alt_result = self.find_alternative_dates(
                target_date=date,
                center=center,
                cloud_threshold=60.0,
                size=size,
                resolution=resolution,
            )

            if alt_result["threshold_met"] and alt_result["alternatives"]:
                best = alt_result["alternatives"][0]
                logger.info(
                    f"Strategy 2 succeeded: alternative {best['date']} has "
                    f"{best['local_cloud_pct']:.1f}% local cloud "
                    f"(window ±{window_days} days)"
                )
                return {
                    "found": True,
                    "strategy_used": "expanded_window",
                    "image": ee.Image(best["image_id"]),
                    "image_id": best["image_id"],
                    "date": best["date"],
                    "local_cloud_pct": best["local_cloud_pct"],
                    "details": {
                        "original_date": date,
                        "window_days": window_days,
                        "search_window": alt_result["search_window"],
                        "alternatives_checked": len(alt_result["alternatives"]),
                        "scene_cloud_pct": best["scene_cloud_pct"],
                    },
                }

            # If alternatives exist but none met the 60% threshold, the next
            # wider window won't help since find_alternative_dates already
            # searches up to ±3 months internally. Break early.
            if alt_result["alternatives"]:
                logger.info(
                    "Alternatives found but none under 60% local cloud — "
                    "moving to Strategy 3"
                )
                break

        # ------------------------------------------------------------------
        # Strategy 3: Temporal compositing (median of clearest scenes)
        # ------------------------------------------------------------------
        logger.info(
            "Strategy 3 — Temporal compositing: median of up to 5 clearest "
            "scenes within ±90 days"
        )

        composite = self.create_temporal_composite(
            center=center,
            target_date=date,
            window_days=90,
            max_scenes=5,
            size=size,
            resolution=resolution,
        )

        if composite is not None:
            logger.info(
                "Strategy 3 succeeded: temporal composite created from ±90 day window"
            )
            return {
                "found": True,
                "strategy_used": "temporal_composite",
                "image": composite,
                "image_id": None,
                "date": "composite",
                "local_cloud_pct": 0.0,  # composite has no single-scene cloud metric
                "details": {
                    "original_date": date,
                    "window_days": 90,
                    "max_scenes": 5,
                    "method": "median",
                },
            }

        # ------------------------------------------------------------------
        # All strategies exhausted
        # ------------------------------------------------------------------
        logger.warning(
            f"All fallback strategies exhausted for {date} at "
            f"{center}. No usable imagery found."
        )
        return _not_found
