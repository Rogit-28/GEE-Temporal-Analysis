"""
Visualization module for SatChange.

This module handles the generation of embossed visual effects and interactive outputs
for detected changes in satellite imagery.
"""

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, Any, Tuple, Optional, List
import logging
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import json

logger = logging.getLogger(__name__)


class VisualizationError(Exception):
    """Exception raised for visualization errors."""

    pass


class EmbossRenderer:
    """Render embossed effects for change visualization."""

    def __init__(self, intensity: float = 1.0):
        """Initialize emboss renderer.

        Args:
            intensity: Emboss effect strength (0.0 to 2.0)
        """
        self.intensity = intensity
        self.emboss_kernel = np.array(
            [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32
        )

    def apply_emboss_effect(self, change_mask: np.ndarray) -> np.ndarray:
        """Apply emboss filter to change mask for 3D visual effect.

        Args:
            change_mask: Binary numpy array (0 = no change, 1 = change)

        Returns:
            Embossed image (0-255 range)
        """
        try:
            # Convert binary mask to uint8 for CV operations
            mask_uint8 = (change_mask * 255).astype(np.uint8)

            # Apply convolution
            embossed = cv2.filter2D(mask_uint8, cv2.CV_32F, self.emboss_kernel)

            # Normalize to 0-255 range
            embossed = cv2.normalize(embossed, None, 0, 255, cv2.NORM_MINMAX)

            # Apply intensity multiplier
            embossed = np.clip(embossed * self.intensity, 0, 255).astype(np.uint8)

            # Add slight blur to smooth artifacts
            embossed = cv2.GaussianBlur(embossed, (3, 3), 0)

            return embossed

        except Exception as e:
            logger.error(f"Emboss effect application failed: {e}")
            raise VisualizationError(f"Emboss effect application failed: {e}")

    def create_color_coded_overlay(
        self, classification: np.ndarray, embossed: np.ndarray
    ) -> np.ndarray:
        """Create RGBA overlay with color-coded change types.

        Args:
            classification: Integer classification map
            embossed: Embossed change mask (0-255)

        Returns:
            RGBA image (H, W, 4)
        """
        try:
            height, width = classification.shape
            overlay = np.zeros((height, width, 4), dtype=np.uint8)

            # Color mapping (R, G, B, A)
            colors = {
                0: (0, 0, 0, 0),  # No change - Transparent
                1: (0, 255, 0, 180),  # Vegetation growth - Green
                2: (255, 0, 0, 180),  # Vegetation loss - Red
                3: (0, 100, 255, 180),  # Water expansion - Blue
                4: (255, 165, 0, 180),  # Water reduction - Orange
                5: (128, 128, 128, 180),  # Urban development - Gray
                6: (64, 64, 64, 180),  # Urban decline - Dark gray
                7: (255, 255, 0, 180),  # Ambiguous - Yellow
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

        except Exception as e:
            logger.error(f"Color-coded overlay creation failed: {e}")
            raise VisualizationError(f"Color-coded overlay creation failed: {e}")


class StaticVisualizer:
    """Generate static visualizations of change detection results."""

    def __init__(self, emboss_intensity: float = 1.0):
        """Initialize static visualizer.

        Args:
            emboss_intensity: Emboss effect strength
        """
        self.emboss_renderer = EmbossRenderer(emboss_intensity)

    def create_rgb_composite(self, bands: Dict[str, np.ndarray]) -> np.ndarray:
        """Create RGB composite from satellite bands.

        Args:
            bands: Dictionary with band arrays

        Returns:
            RGB composite array (H, W, 3)
        """
        try:
            # Create RGB composite (B4=Red, B3=Green, B2=Blue)
            # Note: B2 (Blue) might not be available in all cases
            if "B2" in bands:
                rgb = np.dstack(
                    [
                        bands["B4"],  # Red
                        bands["B3"],  # Green
                        bands["B2"],  # Blue
                    ]
                )
            else:
                # If B2 not available, create grayscale from RGB bands
                rgb = np.dstack(
                    [
                        bands["B4"],  # Red
                        bands["B3"],  # Green
                        bands["B4"],  # Use Red as Blue substitute
                    ]
                )

            # Normalize to 0-255 for display
            rgb_norm = self._normalize_image(rgb)

            return rgb_norm

        except Exception as e:
            logger.error(f"RGB composite creation failed: {e}")
            raise VisualizationError(f"RGB composite creation failed: {e}")

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 range.

        Args:
            image: Input image array

        Returns:
            Normalized image array
        """
        try:
            img_min = image.min()
            img_max = image.max()

            if img_max == img_min:
                # Constant image — return uniform array
                return np.zeros_like(image, dtype=np.uint8)
            elif img_max <= 1.0 and img_min >= 0.0:
                # Already normalized to [0, 1]
                return (image * 255).astype(np.uint8)
            else:
                # Normalize based on min/max
                image_norm = image.astype(float)
                image_norm = (image_norm - img_min) / (img_max - img_min)
                return (image_norm * 255).astype(np.uint8)

        except Exception as e:
            logger.error(f"Image normalization failed: {e}")
            raise VisualizationError(f"Image normalization failed: {e}")

    def generate_comparison_plot(
        self,
        bands_a: Dict[str, np.ndarray],
        bands_b: Dict[str, np.ndarray],
        classification: np.ndarray,
        embossed: np.ndarray,
        output_path: str,
    ) -> None:
        """Generate static comparison plot with before/after/changes.

        Args:
            bands_a: Band arrays for Date A
            bands_b: Band arrays for Date B
            classification: Change classification map
            embossed: Embossed change mask
            output_path: Path to save the plot
        """
        try:
            logger.info(f"Generating static comparison plot: {output_path}")

            # Create RGB composites
            rgb_a = self.create_rgb_composite(bands_a)
            rgb_b = self.create_rgb_composite(bands_b)

            # Create overlay
            overlay = self.emboss_renderer.create_color_coded_overlay(
                classification, embossed
            )

            # Composite: blend overlay with base image
            composite = rgb_b.copy()
            # Alpha blend
            alpha = overlay[:, :, 3:4] / 255.0
            composite = (composite * (1 - alpha) + overlay[:, :, :3] * alpha).astype(
                np.uint8
            )

            # Create figure with subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Panel 1: Date A (Before)
            axes[0].imshow(rgb_a)
            axes[0].set_title("Date A - Before", fontsize=14, fontweight="bold")
            axes[0].axis("off")

            # Panel 2: Date B (After)
            axes[1].imshow(rgb_b)
            axes[1].set_title("Date B - After", fontsize=14, fontweight="bold")
            axes[1].axis("off")

            # Panel 3: Changes overlay
            axes[2].imshow(rgb_b)
            axes[2].imshow(overlay, interpolation="bilinear")
            axes[2].set_title("Detected Changes", fontsize=14, fontweight="bold")
            axes[2].axis("off")

            # Add legend
            self._add_change_legend(axes[2])

            plt.tight_layout()
            try:
                plt.savefig(
                    output_path, dpi=300, bbox_inches="tight", facecolor="white"
                )
                logger.info(f"Static plot saved: {output_path}")
            finally:
                plt.close(fig)

        except Exception as e:
            logger.error(f"Static plot generation failed: {e}")
            raise VisualizationError(f"Static plot generation failed: {e}")

    def _add_change_legend(self, ax) -> None:
        """Add legend to change detection plot.

        Args:
            ax: Matplotlib axis object
        """
        try:
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor=(0, 1, 0, 0.7), label="Vegetation Growth"),
                Patch(facecolor=(1, 0, 0, 0.7), label="Vegetation Loss"),
                Patch(facecolor=(0, 0.4, 1, 0.7), label="Water Expansion"),
                Patch(facecolor=(1, 0.65, 0, 0.7), label="Water Reduction"),
                Patch(facecolor=(0.5, 0.5, 0.5, 0.7), label="Urban Development"),
                Patch(facecolor=(0.25, 0.25, 0.25, 0.7), label="Urban Decline"),
                Patch(facecolor=(1, 1, 0, 0.7), label="Ambiguous"),
            ]

            ax.legend(
                handles=legend_elements,
                loc="upper left",
                bbox_to_anchor=(1, 1),
                fontsize=10,
                framealpha=0.9,
            )

        except Exception as e:
            logger.warning(f"Failed to add legend: {e}")


class InteractiveVisualizer:
    """Generate interactive HTML viewers for change detection results."""

    def __init__(self, emboss_intensity: float = 1.0):
        """Initialize interactive visualizer.

        Args:
            emboss_intensity: Emboss effect strength
        """
        self.emboss_renderer = EmbossRenderer(emboss_intensity)

    def array_to_base64(self, array: np.ndarray) -> str:
        """Convert numpy array to base64-encoded PNG data URI.

        Args:
            array: Input array

        Returns:
            Base64-encoded data URI
        """
        try:
            # Normalize to 0-255 if needed
            if array.dtype != np.uint8 and array.max() <= 1.0 and array.min() >= 0.0:
                array = (array * 255).astype(np.uint8)
            elif array.dtype != np.uint8:
                array = array.astype(np.uint8)

            # Convert to PIL Image
            if len(array.shape) == 2:  # Grayscale
                img = Image.fromarray(array, mode="L")
            else:  # RGB/RGBA
                img = Image.fromarray(array)

            # Encode as PNG in memory
            buffer = BytesIO()
            try:
                img.save(buffer, format="PNG")
                buffer.seek(0)

                # Convert to base64
                img_base64 = base64.b64encode(buffer.read()).decode()

                return f"data:image/png;base64,{img_base64}"
            finally:
                buffer.close()

        except Exception as e:
            logger.error(f"Base64 conversion failed: {e}")
            raise VisualizationError(f"Base64 conversion failed: {e}")

    def create_rgb_composite(self, bands: Dict[str, np.ndarray]) -> np.ndarray:
        """Create RGB composite from satellite bands.

        Args:
            bands: Dictionary with band arrays

        Returns:
            RGB composite array (H, W, 3)
        """
        try:
            # Create RGB composite (B4=Red, B3=Green, B2=Blue)
            if "B2" in bands:
                rgb = np.dstack(
                    [
                        bands["B4"],  # Red
                        bands["B3"],  # Green
                        bands["B2"],  # Blue
                    ]
                )
            else:
                # If B2 not available, create grayscale from RGB bands
                rgb = np.dstack(
                    [
                        bands["B4"],  # Red
                        bands["B3"],  # Green
                        bands["B4"],  # Use Red as Blue substitute
                    ]
                )

            # Normalize to 0-255
            rgb_norm = self._normalize_image(rgb)

            return rgb_norm

        except Exception as e:
            logger.error(f"RGB composite creation failed: {e}")
            raise VisualizationError(f"RGB composite creation failed: {e}")

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 range.

        Args:
            image: Input image array

        Returns:
            Normalized image array
        """
        try:
            img_min = image.min()
            img_max = image.max()

            if img_max == img_min:
                # Constant image — return uniform array
                return np.zeros_like(image, dtype=np.uint8)
            elif img_max <= 1.0 and img_min >= 0.0:
                # Already normalized to [0, 1]
                return (image * 255).astype(np.uint8)
            else:
                image_norm = image.astype(float)
                image_norm = (image_norm - img_min) / (img_max - img_min)
                return (image_norm * 255).astype(np.uint8)

        except Exception as e:
            logger.error(f"Image normalization failed: {e}")
            raise VisualizationError(f"Image normalization failed: {e}")

    def generate_interactive_html(
        self,
        bands_a: Dict[str, np.ndarray],
        bands_b: Dict[str, np.ndarray],
        classification: np.ndarray,
        embossed: np.ndarray,
        stats: Dict[str, Any],
        center_lat: float,
        center_lon: float,
        output_path: str,
    ) -> None:
        """Generate interactive HTML viewer with Leaflet.js.

        Args:
            bands_a: Band arrays for Date A
            bands_b: Band arrays for Date B
            classification: Classification map
            embossed: Embossed mask
            stats: Statistics dictionary
            center_lat: AOI center latitude
            center_lon: AOI center longitude
            output_path: Path to save HTML file
        """
        try:
            logger.info(f"Generating interactive HTML: {output_path}")

            # Create image composites
            rgb_a = self.create_rgb_composite(bands_a)
            rgb_b = self.create_rgb_composite(bands_b)
            overlay = self.emboss_renderer.create_color_coded_overlay(
                classification, embossed
            )

            # Composite overlay on base image
            composite = rgb_b.copy()
            # Alpha blend
            alpha = overlay[:, :, 3:4] / 255.0
            composite = (composite * (1 - alpha) + overlay[:, :, :3] * alpha).astype(
                np.uint8
            )

            # Convert to base64
            img_a_uri = self.array_to_base64(rgb_a)
            img_b_uri = self.array_to_base64(rgb_b)
            img_overlay_uri = self.array_to_base64(composite)

            # Get dates from stats if available
            date_a = stats.get("date_a", {}).get("date", "Unknown")
            date_b = stats.get("date_b", {}).get("date", "Unknown")

            # HTML template
            html_template = self._get_html_template()

            # Render template
            from jinja2 import Environment

            env = Environment(autoescape=True)
            template = env.from_string(html_template)
            html_content = template.render(
                img_a_uri=img_a_uri,
                img_b_uri=img_b_uri,
                img_overlay_uri=img_overlay_uri,
                center_lat=center_lat,
                center_lon=center_lon,
                date_a=date_a,
                date_b=date_b,
                stats=stats,
            )

            # Write to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"Interactive HTML saved: {output_path}")

        except Exception as e:
            logger.error(f"Interactive HTML generation failed: {e}")
            raise VisualizationError(f"Interactive HTML generation failed: {e}")

    def _get_html_template(self) -> str:
        """Get HTML template for interactive viewer.

        Returns:
            HTML template string
        """
        return r"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SatChange Analysis Results</title>
    <!-- Leaflet 1.9.4 CSS (inlined for offline use) -->
    <style>
/* required styles */



.leaflet-pane,

.leaflet-tile,

.leaflet-marker-icon,

.leaflet-marker-shadow,

.leaflet-tile-container,

.leaflet-pane > svg,

.leaflet-pane > canvas,

.leaflet-zoom-box,

.leaflet-image-layer,

.leaflet-layer {

	position: absolute;

	left: 0;

	top: 0;

	}

.leaflet-container {

	overflow: hidden;

	}

.leaflet-tile,

.leaflet-marker-icon,

.leaflet-marker-shadow {

	-webkit-user-select: none;

	   -moz-user-select: none;

	        user-select: none;

	  -webkit-user-drag: none;

	}

/* Prevents IE11 from highlighting tiles in blue */

.leaflet-tile::selection {

	background: transparent;

}

/* Safari renders non-retina tile on retina better with this, but Chrome is worse */

.leaflet-safari .leaflet-tile {

	image-rendering: -webkit-optimize-contrast;

	}

/* hack that prevents hw layers "stretching" when loading new tiles */

.leaflet-safari .leaflet-tile-container {

	width: 1600px;

	height: 1600px;

	-webkit-transform-origin: 0 0;

	}

.leaflet-marker-icon,

.leaflet-marker-shadow {

	display: block;

	}

/* .leaflet-container svg: reset svg max-width decleration shipped in Joomla! (joomla.org) 3.x */

/* .leaflet-container img: map is broken in FF if you have max-width: 100% on tiles */

.leaflet-container .leaflet-overlay-pane svg {

	max-width: none !important;

	max-height: none !important;

	}

.leaflet-container .leaflet-marker-pane img,

.leaflet-container .leaflet-shadow-pane img,

.leaflet-container .leaflet-tile-pane img,

.leaflet-container img.leaflet-image-layer,

.leaflet-container .leaflet-tile {

	max-width: none !important;

	max-height: none !important;

	width: auto;

	padding: 0;

	}



.leaflet-container img.leaflet-tile {

	/* See: https://bugs.chromium.org/p/chromium/issues/detail?id=600120 */

	mix-blend-mode: plus-lighter;

}



.leaflet-container.leaflet-touch-zoom {

	-ms-touch-action: pan-x pan-y;

	touch-action: pan-x pan-y;

	}

.leaflet-container.leaflet-touch-drag {

	-ms-touch-action: pinch-zoom;

	/* Fallback for FF which doesn't support pinch-zoom */

	touch-action: none;

	touch-action: pinch-zoom;

}

.leaflet-container.leaflet-touch-drag.leaflet-touch-zoom {

	-ms-touch-action: none;

	touch-action: none;

}

.leaflet-container {

	-webkit-tap-highlight-color: transparent;

}

.leaflet-container a {

	-webkit-tap-highlight-color: rgba(51, 181, 229, 0.4);

}

.leaflet-tile {

	filter: inherit;

	visibility: hidden;

	}

.leaflet-tile-loaded {

	visibility: inherit;

	}

.leaflet-zoom-box {

	width: 0;

	height: 0;

	-moz-box-sizing: border-box;

	     box-sizing: border-box;

	z-index: 800;

	}

/* workaround for https://bugzilla.mozilla.org/show_bug.cgi?id=888319 */

.leaflet-overlay-pane svg {

	-moz-user-select: none;

	}



.leaflet-pane         { z-index: 400; }



.leaflet-tile-pane    { z-index: 200; }

.leaflet-overlay-pane { z-index: 400; }

.leaflet-shadow-pane  { z-index: 500; }

.leaflet-marker-pane  { z-index: 600; }

.leaflet-tooltip-pane   { z-index: 650; }

.leaflet-popup-pane   { z-index: 700; }



.leaflet-map-pane canvas { z-index: 100; }

.leaflet-map-pane svg    { z-index: 200; }



.leaflet-vml-shape {

	width: 1px;

	height: 1px;

	}

.lvml {

	behavior: url(#default#VML);

	display: inline-block;

	position: absolute;

	}





/* control positioning */



.leaflet-control {

	position: relative;

	z-index: 800;

	pointer-events: visiblePainted; /* IE 9-10 doesn't have auto */

	pointer-events: auto;

	}

.leaflet-top,

.leaflet-bottom {

	position: absolute;

	z-index: 1000;

	pointer-events: none;

	}

.leaflet-top {

	top: 0;

	}

.leaflet-right {

	right: 0;

	}

.leaflet-bottom {

	bottom: 0;

	}

.leaflet-left {

	left: 0;

	}

.leaflet-control {

	float: left;

	clear: both;

	}

.leaflet-right .leaflet-control {

	float: right;

	}

.leaflet-top .leaflet-control {

	margin-top: 10px;

	}

.leaflet-bottom .leaflet-control {

	margin-bottom: 10px;

	}

.leaflet-left .leaflet-control {

	margin-left: 10px;

	}

.leaflet-right .leaflet-control {

	margin-right: 10px;

	}





/* zoom and fade animations */



.leaflet-fade-anim .leaflet-popup {

	opacity: 0;

	-webkit-transition: opacity 0.2s linear;

	   -moz-transition: opacity 0.2s linear;

	        transition: opacity 0.2s linear;

	}

.leaflet-fade-anim .leaflet-map-pane .leaflet-popup {

	opacity: 1;

	}

.leaflet-zoom-animated {

	-webkit-transform-origin: 0 0;

	    -ms-transform-origin: 0 0;

	        transform-origin: 0 0;

	}

svg.leaflet-zoom-animated {

	will-change: transform;

}

