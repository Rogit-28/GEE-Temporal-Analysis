"""Helpers for JobID web bundle and Next.js viewer payload generation."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import datetime
from io import BytesIO
from typing import Any, Dict

import numpy as np
from PIL import Image

from .utils import NumpyJSONEncoder, sanitize_output_name, safe_join

WEB_MANIFEST_VERSION = "1.1.0"

CLASS_MAPPING = {
    "0": "no_change",
    "1": "vegetation_growth",
    "2": "vegetation_loss",
    "3": "water_expansion",
    "4": "water_reduction",
    "5": "urban_development",
    "6": "urban_decline",
    "7": "ambiguous",
}

LEGEND_ITEMS = [
    {"id": "1", "label": "Vegetation Growth", "color": "rgba(0,255,0,0.7)"},
    {"id": "2", "label": "Vegetation Loss", "color": "rgba(255,0,0,0.7)"},
    {"id": "3", "label": "Water Expansion", "color": "rgba(0,100,255,0.7)"},
    {"id": "4", "label": "Water Reduction", "color": "rgba(255,165,0,0.7)"},
    {"id": "5", "label": "Urban Development", "color": "rgba(128,128,128,0.7)"},
    {"id": "6", "label": "Urban Decline", "color": "rgba(64,64,64,0.7)"},
]


def build_job_id(output_prefix: str, center_lat: float, center_lon: float) -> str:
    """Build stable JobID from output prefix and center coordinates."""
    raw = f"{output_prefix}|{center_lat:.6f}|{center_lon:.6f}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{output_prefix}-{digest}"


def _to_base64_png_uri(array: np.ndarray) -> str:
    """Convert numpy array to base64-encoded PNG data URI."""
    if array.dtype != np.uint8 and array.max() <= 1.0 and array.min() >= 0.0:
        array = (array * 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = array.astype(np.uint8)

    if len(array.shape) == 2:
        image = Image.fromarray(array, mode="L")
    else:
        image = Image.fromarray(array)

    buffer = BytesIO()
    try:
        image.save(buffer, format="PNG")
        buffer.seek(0)
        image_b64 = base64.b64encode(buffer.read()).decode("utf-8")
    finally:
        buffer.close()
    return f"data:image/png;base64,{image_b64}"


def _normalize_rgb(image: np.ndarray) -> np.ndarray:
    """Normalize an RGB image to uint8 [0, 255]."""
    img_min = image.min()
    img_max = image.max()
    if img_max == img_min:
        return np.zeros_like(image, dtype=np.uint8)
    if img_max <= 1.0 and img_min >= 0.0:
        return (image * 255).astype(np.uint8)
    normalized = image.astype(float)
    normalized = (normalized - img_min) / (img_max - img_min)
    return (normalized * 255).astype(np.uint8)


def _create_rgb_composite(bands: Dict[str, np.ndarray]) -> np.ndarray:
    """Create RGB composite from Sentinel bands."""
    if "B2" in bands:
        rgb = np.dstack([bands["B4"], bands["B3"], bands["B2"]])
    else:
        rgb = np.dstack([bands["B4"], bands["B3"], bands["B4"]])
    return _normalize_rgb(rgb)


def _build_changes_composite(
    rgb_after: np.ndarray, overlay_rgba: np.ndarray
) -> np.ndarray:
    """Blend change overlay over Date B RGB image."""
    alpha = overlay_rgba[:, :, 3:4] / 255.0
    return (rgb_after * (1 - alpha) + overlay_rgba[:, :, :3] * alpha).astype(np.uint8)


def _extract_date(metadata: Dict[str, Any], key: str, fallback: str) -> str:
    """Extract date string (YYYY-MM-DD) from metadata date block."""
    value = metadata.get(key, {})
    if isinstance(value, dict):
        raw = value.get("date", fallback)
    else:
        raw = fallback
    if isinstance(raw, str):
        return raw.split("T")[0]
    return fallback


def _stats_percent(stats: Dict[str, Any], key: str) -> str:
    """Format percent value for a stats key."""
    value = stats.get(key, {}).get("percent", 0.0)
    return f"{value}%"


def _build_viewer_payload(
    center_lat: float,
    center_lon: float,
    stats: Dict[str, Any],
    metadata: Dict[str, Any],
    rgb_before_uri: str,
    rgb_after_uri: str,
    changes_uri: str,
) -> Dict[str, Any]:
    """Build viewer payload consumed by Next.js job map page."""
    lat_offset = 0.01
    lon_offset = 0.01
    date_a = _extract_date(metadata, "date_a", "Date A")
    date_b = _extract_date(metadata, "date_b", "Date B")

    return {
        "map": {
            "center": {"lat": center_lat, "lon": center_lon},
            "zoom": 13,
            "bounds": [
                [center_lat - lat_offset, center_lon - lon_offset],
                [center_lat + lat_offset, center_lon + lon_offset],
            ],
            "tile_url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "tile_attribution": "© OpenStreetMap contributors",
        },
        "controls": {"default_layer": "changes"},
        "layers": {
            "before": {"label": "Date A", "date": date_a, "image_uri": rgb_before_uri},
            "after": {"label": "Date B", "date": date_b, "image_uri": rgb_after_uri},
            "changes": {"label": "Changes", "image_uri": changes_uri},
        },
        "legend": LEGEND_ITEMS,
        "stats_cards": [
            {
                "key": "total_area_changed",
                "label": "Total Area Changed",
                "value": _stats_percent(stats, "total_change"),
            },
            {
                "key": "vegetation_growth",
                "label": "Vegetation Growth",
                "value": _stats_percent(stats, "vegetation_growth"),
            },
            {
                "key": "vegetation_loss",
                "label": "Vegetation Loss",
                "value": _stats_percent(stats, "vegetation_loss"),
            },
            {
                "key": "water_expansion",
                "label": "Water Expansion",
                "value": _stats_percent(stats, "water_expansion"),
            },
            {
                "key": "water_reduction",
                "label": "Water Reduction",
                "value": _stats_percent(stats, "water_reduction"),
            },
            {
                "key": "changed_area",
                "label": "Changed Area",
                "value": f"{stats.get('total_change', {}).get('area_km2', 0)} km²",
            },
        ],
    }


def export_web_bundle(
    output_dir: str,
    output_prefix: str,
    center_lat: float,
    center_lon: float,
    classification: np.ndarray,
    stats: Dict[str, Any],
    metadata: Dict[str, Any],
    generated_files: Dict[str, str],
    bands_a: Dict[str, np.ndarray],
    bands_b: Dict[str, np.ndarray],
    overlay_rgba: np.ndarray,
) -> Dict[str, str]:
    """Export JobID-linked manifest with viewer-ready rendering payload."""
    abs_output_dir = os.path.abspath(output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    safe_prefix = sanitize_output_name(output_prefix) or "output"

    job_id = build_job_id(safe_prefix, center_lat, center_lon)
    bundle_dir = safe_join(abs_output_dir, f"{safe_prefix}_web_bundle")
    os.makedirs(bundle_dir, exist_ok=True)

    manifest_path = safe_join(bundle_dir, "manifest.json")
    index_path = safe_join(abs_output_dir, f"{safe_prefix}_job.json")
    height, width = classification.shape

    static_png = generated_files.get("static")
    if static_png and not os.path.isabs(static_png):
        static_png = os.path.abspath(static_png)
    legacy_html = generated_files.get("interactive")
    if legacy_html and not os.path.isabs(legacy_html):
        legacy_html = os.path.abspath(legacy_html)
    geotiff = generated_files.get("geotiff")
    if geotiff and not os.path.isabs(geotiff):
        geotiff = os.path.abspath(geotiff)

    rgb_before = _create_rgb_composite(bands_a)
    rgb_after = _create_rgb_composite(bands_b)
    changes_composite = _build_changes_composite(rgb_after, overlay_rgba)
    viewer_payload = _build_viewer_payload(
        center_lat=center_lat,
        center_lon=center_lon,
        stats=stats,
        metadata=metadata,
        rgb_before_uri=_to_base64_png_uri(rgb_before),
        rgb_after_uri=_to_base64_png_uri(rgb_after),
        changes_uri=_to_base64_png_uri(changes_composite),
    )

    manifest = {
        "schema_version": WEB_MANIFEST_VERSION,
        "job_id": job_id,
        "output_prefix": safe_prefix,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "center": {"lat": center_lat, "lon": center_lon},
        "dimensions": {"height": height, "width": width},
        "artifacts": {
            "static_png": static_png,
            "legacy_interactive_html": legacy_html,
            "classification_geotiff": geotiff,
            "classification_npy": safe_join(
                abs_output_dir, f"{safe_prefix}_classification.npy"
            ),
            "bands_a_npz": safe_join(abs_output_dir, f"{safe_prefix}_bands_a.npz"),
            "bands_b_npz": safe_join(abs_output_dir, f"{safe_prefix}_bands_b.npz"),
            "change_stats_json": safe_join(
                abs_output_dir, f"{safe_prefix}_change_stats.json"
            ),
            "metadata_json": safe_join(abs_output_dir, f"{safe_prefix}_metadata.json"),
        },
        "stats": stats,
        "metadata": metadata,
        "class_mapping": CLASS_MAPPING,
        "viewer": viewer_payload,
    }

    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, cls=NumpyJSONEncoder)

    with open(index_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "job_id": job_id,
                "manifest_path": manifest_path,
                "bundle_path": bundle_dir,
            },
            file,
            indent=2,
        )

    return {"job_id": job_id, "manifest_path": manifest_path, "bundle_path": bundle_dir}
