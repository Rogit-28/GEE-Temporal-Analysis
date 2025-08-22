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

