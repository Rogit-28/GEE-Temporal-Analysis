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

