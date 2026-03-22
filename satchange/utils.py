"""
Utility functions for SatChange.

This module provides helper functions for common operations used across
the SatChange CLI tool.
"""

import logging
import sys
import json
import numpy as np
from typing import Any, Dict, Tuple, Optional
from datetime import datetime, timedelta
import os
import re

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)

    # Suppress overly verbose loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def parse_coordinates(coord_string: str) -> Tuple[float, float]:
    """Parse coordinate string into latitude and longitude.

    Args:
        coord_string: Coordinate string in format "lat,lon" or "lat lon"

    Returns:
        Tuple of (latitude, longitude)

    Raises:
        ValueError: If coordinate string is invalid
    """
    try:
        # Remove whitespace and split
        parts = re.split(r"[,\s]+", coord_string.strip())

        if len(parts) != 2:
            raise ValueError("Coordinates must be in format 'lat,lon' or 'lat lon'")

        lat, lon = map(float, parts)

        # Validate coordinate ranges
        if not -90 <= lat <= 90:
            raise ValueError(f"Latitude must be between -90 and 90, got {lat}")

        if not -180 <= lon <= 180:
            raise ValueError(f"Longitude must be between -180 and 180, got {lon}")

        return lat, lon

    except ValueError as e:
        if "could not convert" in str(e):
            raise ValueError(f"Invalid coordinate format: {coord_string}")
        raise


def parse_date(date_string: str) -> Tuple[datetime, datetime]:
    """Parse date string into start and end datetime objects.

    Args:
        date_string: Date string in format "YYYY-MM-DD:YYYY-MM-DD" or "YYYY-MM-DD"

    Returns:
        Tuple of (start_date, end_date)

    Raises:
        ValueError: If date string is invalid
    """
    try:
        if ":" in date_string:
            # Date range format
            start_str, end_str = date_string.split(":", 1)
            start_date = datetime.strptime(start_str.strip(), "%Y-%m-%d")
            end_date = datetime.strptime(end_str.strip(), "%Y-%m-%d")
        else:
            # Single date - use as start date, end date is same day
            start_date = datetime.strptime(date_string.strip(), "%Y-%m-%d")
            end_date = start_date

        # Validate date range
        if start_date > end_date:
            raise ValueError(
                f"Start date ({start_date}) cannot be after end date ({end_date})"
            )

        # Limit date range to reasonable maximum (10 years)
        max_range = datetime.now() - timedelta(days=365 * 10)
        if start_date < max_range:
            raise ValueError("Start date too far in the past (max 10 years)")

        return start_date, end_date

    except ValueError as e:
        error_msg = str(e).lower()
        if "could not convert" in error_msg or "does not match format" in error_msg:
            raise ValueError(
                f"Invalid date format: {date_string}. Use YYYY-MM-DD:YYYY-MM-DD"
            )
        raise


def validate_pixel_size(pixel_size: int) -> None:
    """Validate pixel size parameter.

    Args:
        pixel_size: Pixel size value

    Raises:
        ValueError: If pixel size is invalid
    """
    if not isinstance(pixel_size, int) or pixel_size <= 0:
        raise ValueError(f"Pixel size must be a positive integer, got {pixel_size}")

    if pixel_size > 1000:
        raise ValueError(f"Pixel size too large (max 1000), got {pixel_size}")


def validate_cloud_threshold(cloud_threshold: int) -> None:
    """Validate cloud threshold parameter.

    Args:
        cloud_threshold: Cloud threshold percentage

    Raises:
        ValueError: If cloud threshold is invalid
    """
    if not isinstance(cloud_threshold, int) or not 0 <= cloud_threshold <= 100:
        raise ValueError(
            f"Cloud threshold must be between 0 and 100, got {cloud_threshold}"
        )


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    if size_bytes == 0:
        return "0 B"

    if size_bytes < 0:
        return f"-{format_file_size(-size_bytes)}"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)

    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1

    return f"{size:.1f} {size_names[i]}"


def validate_threshold(
    threshold: float, min_val: float = 0.0, max_val: float = 1.0
) -> None:
    """Validate threshold parameter.

    Args:
        threshold: Threshold value
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Raises:
        ValueError: If threshold is invalid
    """
    if not isinstance(threshold, (int, float)):
        raise ValueError(f"Threshold must be a number, got {type(threshold)}")

    if not min_val <= threshold <= max_val:
        raise ValueError(
            f"Threshold must be between {min_val} and {max_val}, got {threshold}"
        )


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_output_name(name: Optional[str]) -> Optional[str]:
    """Sanitize user-provided output name for filesystem-safe artifact prefixes."""
    if name is None:
        return None
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Name cannot be empty")

    if os.path.sep in cleaned or (os.path.altsep and os.path.altsep in cleaned):
        raise ValueError("Name must not contain path separators")
    if cleaned in {".", ".."} or ".." in cleaned:
        raise ValueError("Name must not contain relative path segments")

    safe = _SAFE_NAME_RE.sub("_", cleaned).strip("._-")
    if not safe:
        raise ValueError(
            "Name contains no valid filename characters (use letters, numbers, dot, underscore, hyphen)"
        )
    return safe


def safe_join(base_dir: str, *parts: str) -> str:
    """Join path components and ensure final path stays within base_dir."""
    base_abs = os.path.abspath(base_dir)
    candidate = os.path.abspath(os.path.join(base_abs, *parts))
    if os.path.commonpath([base_abs, candidate]) != base_abs:
        raise ValueError(f"Unsafe path escape attempt: {candidate}")
    return candidate


def check_disk_space(path: str, required_mb: float = 100.0) -> Dict[str, Any]:
    """Check available disk space at the given path.

    Args:
        path: Directory path to check
        required_mb: Required free space in megabytes

    Returns:
        Dictionary with:
            - available_mb: float - available space in MB
            - required_mb: float - required space in MB
            - sufficient: bool - whether space is sufficient
    """
    import shutil

    try:
        usage = shutil.disk_usage(path)
        available_mb = usage.free / (1024 * 1024)
        return {
            "available_mb": round(available_mb, 1),
            "required_mb": required_mb,
            "sufficient": available_mb >= required_mb,
        }
    except OSError as e:
        logger.warning(f"Could not check disk space: {e}")
        return {
            "available_mb": -1,
            "required_mb": required_mb,
            "sufficient": False,
        }


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.datetime64):
            return str(obj)
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        return super().default(obj)
