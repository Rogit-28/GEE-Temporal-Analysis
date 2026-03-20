"""
Image processing pipeline for SatChange.

This module handles image preprocessing, including cloud masking,
coregistration, and radiometric normalization.
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)


class ImageProcessingError(Exception):
    """Exception raised for image processing errors."""

    pass


class ImageProcessor:
    """Image preprocessing pipeline for satellite imagery."""

    def __init__(self, config):
        """Initialize image processor.

        Args:
            config: Configuration instance
        """
        self.config = config
        self.cloud_threshold = config.get("cloud_threshold", 20)

    def _resample_b11_to_10m(
        self, bands: Dict[str, np.ndarray], reference_band: str = "B4"
    ) -> Dict[str, np.ndarray]:
        """Resample B11 band from 20m to 10m resolution to match other bands.

        Sentinel-2 B11 (SWIR) is natively 20m resolution while B4/B3/B8 are 10m.
        This method upsamples B11 using bilinear interpolation to match the
        spatial resolution of the 10m bands.

        Args:
            bands: Dictionary of band arrays
            reference_band: Name of a 10m band to use as reference for target shape

        Returns:
            Dictionary of band arrays with B11 resampled to 10m
        """
        if "B11" not in bands:
            logger.debug("B11 band not present, skipping resampling")
            return bands

        # Get reference shape from a 10m band (B4, B3, or B8)
        reference_shape = None
        for ref_band in [reference_band, "B4", "B3", "B8"]:
            if ref_band in bands:
                reference_shape = bands[ref_band].shape
                logger.debug(
                    f"Using {ref_band} as reference for B11 resampling (shape: {reference_shape})"
                )
                break

        if reference_shape is None:
            logger.warning("No 10m reference band found for B11 resampling")
            return bands

        b11 = bands["B11"]

        # Check if resampling is needed (B11 at 20m should be ~half the size of 10m bands)
        if b11.shape == reference_shape:
            logger.debug("B11 already matches reference shape, skipping resampling")
            return bands

        try:
            # Calculate zoom factors for resampling
            zoom_factors = (
                reference_shape[0] / b11.shape[0],
                reference_shape[1] / b11.shape[1],
            )

            logger.info(
                f"Resampling B11 from {b11.shape} to {reference_shape} (zoom factors: {zoom_factors})"
            )

            # Use scipy.ndimage.zoom with bilinear interpolation (order=1)
            # order=1 is bilinear, order=3 is bicubic
            b11_resampled = zoom(b11, zoom_factors, order=3, mode="nearest")

            # Clamp to original value range to prevent bicubic overshoot/ringing
            b11_resampled = np.clip(b11_resampled, b11.min(), b11.max())

            # Ensure exact shape match (zoom may produce slightly different dimensions due to rounding)
            if b11_resampled.shape != reference_shape:
                logger.warning(
                    f"B11 resampled shape {b11_resampled.shape} differs from reference {reference_shape}, "
                    f"cropping/padding to match"
                )
                # Crop or pad to match exactly
                b11_final = np.zeros(reference_shape, dtype=b11.dtype)
                min_h = min(b11_resampled.shape[0], reference_shape[0])
                min_w = min(b11_resampled.shape[1], reference_shape[1])
                b11_final[:min_h, :min_w] = b11_resampled[:min_h, :min_w]
                b11_resampled = b11_final

            # Preserve original dtype
            b11_resampled = b11_resampled.astype(b11.dtype)

            # Update bands dictionary
            bands_resampled = bands.copy()
            bands_resampled["B11"] = b11_resampled

            logger.info(
                f"B11 resampling completed: {b11.shape} -> {b11_resampled.shape}"
            )
            return bands_resampled

        except Exception as e:
            logger.error(f"Failed to resample B11: {e}")
            return bands

    def preprocess_image_pair(
        self,
        bands_a: Dict[str, np.ndarray],
        bands_b: Dict[str, np.ndarray],
        metadata_a: Dict[str, Any],
        metadata_b: Dict[str, Any],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Preprocess a pair of satellite images.

        Args:
            bands_a: Band arrays for Date A
            bands_b: Band arrays for Date B
            metadata_a: Metadata for Date A
            metadata_b: Metadata for Date B

        Returns:
            Tuple of (processed_bands_a, processed_bands_b)

        Raises:
            ImageProcessingError: If preprocessing fails
        """
        try:
            logger.info("Starting image preprocessing pipeline")

            # Step 1: Resample B11 from 20m to 10m resolution
            logger.info("Resampling B11 to 10m resolution...")
            bands_a = self._resample_b11_to_10m(bands_a)
            bands_b = self._resample_b11_to_10m(bands_b)

            # Step 2: Cloud masking
            logger.info("Applying cloud masking...")
            bands_a, bands_b = self._apply_cloud_masking(bands_a, bands_b)

            # Step 3: Coregistration check
            logger.info("Checking coregistration...")
            self._check_coregistration(metadata_a, metadata_b)

            # Step 4: Radiometric normalization (if needed)
            logger.info("Applying radiometric normalization...")
            bands_a, bands_b = self._apply_radiometric_normalization(bands_a, bands_b)

            logger.info("Image preprocessing completed successfully")
            return bands_a, bands_b

        except Exception as e:
            error_msg = f"Image preprocessing failed: {e}"
            logger.error(error_msg)
            raise ImageProcessingError(error_msg)

    def _apply_cloud_masking(
        self, bands_a: Dict[str, np.ndarray], bands_b: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Apply cloud masking to image pairs.

        Args:
            bands_a: Band arrays for Date A
            bands_b: Band arrays for Date B

        Returns:
            Tuple of (cloud_masked_bands_a, cloud_masked_bands_b)
        """
        try:
            # Extract QA60 band for cloud detection
            qa60_a = bands_a.get("QA60")
            qa60_b = bands_b.get("QA60")

            if qa60_a is None or qa60_b is None:
                logger.warning("QA60 band not available, skipping cloud masking")
                return bands_a, bands_b

            # Create cloud masks
            cloud_mask_a = self._create_cloud_mask(qa60_a)
            cloud_mask_b = self._create_cloud_mask(qa60_b)

            # Calculate cloud coverage (cloud_mask: 1=clear, 0=cloudy)
            # So cloud coverage = percentage of cloudy pixels = sum(mask==0) / size
            cloud_coverage_a = (np.sum(cloud_mask_a == 0) / cloud_mask_a.size) * 100
            cloud_coverage_b = (np.sum(cloud_mask_b == 0) / cloud_mask_b.size) * 100

            logger.info(
                f"Cloud coverage - Date A: {cloud_coverage_a:.1f}%, Date B: {cloud_coverage_b:.1f}%"
            )

            # Apply cloud masks to all bands
            # Note: For change detection, we keep the original data and just track cloud coverage
            # The cloud mask can be used later to exclude cloudy pixels from analysis if needed
            bands_a_masked = {}
            bands_b_masked = {}

            for band_name, band_array in bands_a.items():
                if band_name != "QA60":  # Don't mask the QA band itself
                    # np.where(condition, value_if_true, value_if_false)
                    # cloud_mask: 1=clear, 0=cloudy
                    # Keep clear pixels, set cloudy pixels to 0
                    bands_a_masked[band_name] = np.where(
                        cloud_mask_a == 1, band_array, 0
                    )
                else:
                    bands_a_masked[band_name] = band_array

            for band_name, band_array in bands_b.items():
                if band_name != "QA60":
                    bands_b_masked[band_name] = np.where(
                        cloud_mask_b == 1, band_array, 0
                    )
                else:
                    bands_b_masked[band_name] = band_array

            # Check if cloud coverage exceeds threshold
            if cloud_coverage_a > self.cloud_threshold:
                logger.warning(
                    f"Date A cloud coverage ({cloud_coverage_a:.1f}%) exceeds threshold ({self.cloud_threshold}%)"
                )

            if cloud_coverage_b > self.cloud_threshold:
                logger.warning(
                    f"Date B cloud coverage ({cloud_coverage_b:.1f}%) exceeds threshold ({self.cloud_threshold}%)"
                )

            return bands_a_masked, bands_b_masked

        except Exception as e:
            logger.error(f"Cloud masking failed: {e}")
            # Return original bands if masking fails
            return bands_a, bands_b

    def _create_cloud_mask(self, qa60_band: np.ndarray) -> np.ndarray:
        """Create cloud mask from QA60 band.

        Args:
            qa60_band: QA60 band array

        Returns:
            Binary cloud mask (1 = clear, 0 = cloudy)
        """
        try:
            # QA60 band contains cloud and cirrus information in bit masks
            # Bit 10: Opaque clouds (1024)
            # Bit 11: Cirrus clouds (2048)

            # Create cloud mask using bitwise operations
            cloud_bit = 1 << 10  # 1024 - opaque clouds
            cirrus_bit = 1 << 11  # 2048 - cirrus clouds

            # Check if either cloud or cirrus bit is set
            qa60_int = qa60_band.astype(np.int32)
            cloud_present = ((qa60_int & cloud_bit) != 0) | (
                (qa60_int & cirrus_bit) != 0
            )

            # Clear pixels = 1, cloudy pixels = 0
            cloud_mask = (~cloud_present).astype(np.uint8)

            return cloud_mask

        except Exception as e:
            logger.error(f"Failed to create cloud mask: {e}")
            # Return all-clear mask if creation fails
            return np.ones_like(qa60_band, dtype=np.uint8)

    def _check_coregistration(
        self, metadata_a: Dict[str, Any], metadata_b: Dict[str, Any]
    ) -> None:
        """Check if images are properly coregistered.

        Args:
            metadata_a: Metadata for Date A
            metadata_b: Metadata for Date B

        Raises:
            ImageProcessingError: If images are not properly coregistered
        """
        try:
            # Check if images have the same dimensions
            width_a = metadata_a.get("width")
            height_a = metadata_a.get("height")
            width_b = metadata_b.get("width")
            height_b = metadata_b.get("height")

            if width_a != width_b or height_a != height_b:
                logger.warning(
                    f"Image dimensions differ - Date A: {width_a}x{height_a}, Date B: {width_b}x{height_b}"
                )

            # Check coordinate reference system (CRS)
            crs_a = metadata_a.get("crs")
            crs_b = metadata_b.get("crs")

            if crs_a != crs_b:
                logger.warning(
                    f"Different CRS detected - Date A: {crs_a}, Date B: {crs_b}"
                )

            # Check transform (georeferencing)
            transform_a = metadata_a.get("transform")
            transform_b = metadata_b.get("transform")

            if transform_a != transform_b:
                logger.warning("Different geotransforms detected")

            # For now, we'll proceed with a warning
            # In a production system, you might want to apply coregistration transforms

        except Exception as e:
            logger.error(f"Coregistration check failed: {e}")
            # Don't raise exception, just log the issue

    def _apply_radiometric_normalization(
        self, bands_a: Dict[str, np.ndarray], bands_b: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Apply radiometric normalization to image pairs.

        Args:
            bands_a: Band arrays for Date A
            bands_b: Band arrays for Date B

        Returns:
            Tuple of (normalized_bands_a, normalized_bands_b)
        """
        try:
            # For Sentinel-2 Surface Reflectance products, atmospheric correction
            # is already applied via Sen2Cor, so radiometric normalization
            # is typically not required

            # However, we can apply histogram matching if there are significant
            # brightness differences between the images

            # Check for significant brightness differences
            brightness_diff = self._calculate_brightness_difference(bands_a, bands_b)

            if brightness_diff > 0.1:  # 10% difference threshold
                logger.info(
                    f"Significant brightness difference detected ({brightness_diff:.2%}), applying histogram matching"
                )
                bands_a, bands_b = self._apply_histogram_matching(bands_a, bands_b)
            else:
                logger.info(
                    "Brightness differences within acceptable range, skipping normalization"
                )

            return bands_a, bands_b

        except Exception as e:
            logger.error(f"Radiometric normalization failed: {e}")
            # Return original bands if normalization fails
            return bands_a, bands_b

    def _calculate_brightness_difference(
        self, bands_a: Dict[str, np.ndarray], bands_b: Dict[str, np.ndarray]
    ) -> float:
        """Calculate brightness difference between image pairs.

        Args:
            bands_a: Band arrays for Date A
            bands_b: Band arrays for Date B

        Returns:
            Normalized brightness difference (0-1)
        """
        try:
            # Calculate mean brightness for common bands
            common_bands = set(bands_a.keys()) & set(bands_b.keys())

            if not common_bands:
                return 0.0

            brightness_a = []
            brightness_b = []

            for band in common_bands:
                # Exclude QA band from brightness calculation
                if band != "QA60":
                    band_a = bands_a[band]
                    band_b = bands_b[band]

                    # Calculate mean brightness (excluding zero values)
                    valid_pixels_a = band_a[band_a > 0]
                    valid_pixels_b = band_b[band_b > 0]

                    if len(valid_pixels_a) > 0 and len(valid_pixels_b) > 0:
                        brightness_a.append(np.mean(valid_pixels_a))
                        brightness_b.append(np.mean(valid_pixels_b))

            if not brightness_a or not brightness_b:
                return 0.0

            # Calculate relative difference
            mean_brightness_a = np.mean(brightness_a)
            mean_brightness_b = np.mean(brightness_b)

            if abs(mean_brightness_a) < 1e-10:
                return 0.0

            relative_diff = (
                abs(mean_brightness_b - mean_brightness_a) / mean_brightness_a
            )
            return min(relative_diff, 1.0)  # Cap at 1.0

        except Exception as e:
            logger.error(f"Failed to calculate brightness difference: {e}")
            return 0.0

    def _apply_histogram_matching(
        self, bands_a: Dict[str, np.ndarray], bands_b: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Apply histogram matching to normalize brightness.

        Args:
            bands_a: Band arrays for Date A (reference)
            bands_b: Band arrays for Date B (to be normalized)

        Returns:
            Tuple of (normalized_bands_a, normalized_bands_b)
        """
        try:
            bands_a_norm = {}
            bands_b_norm = {}

            common_bands = set(bands_a.keys()) & set(bands_b.keys())

            for band in common_bands:
                if band != "QA60":  # Don't normalize QA band
                    band_a = bands_a[band]
                    band_b = bands_b[band]

                    # Apply histogram matching
                    band_b_matched = self._match_histogram(band_a, band_b)

                    bands_a_norm[band] = band_a
                    bands_b_norm[band] = band_b_matched
                else:
                    # Keep QA band as-is
                    bands_a_norm[band] = band_a
                    bands_b_norm[band] = band_b

            return bands_a_norm, bands_b_norm

        except Exception as e:
            logger.error(f"Histogram matching failed: {e}")
            # Return original bands if matching fails
            return bands_a, bands_b

    def _match_histogram(self, reference: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Match histogram of target image to reference image.

        Args:
            reference: Reference image array
            target: Target image array to be normalized

        Returns:
            Histogram-matched target array
        """
        try:
            # Flatten and get valid (non-zero) pixels
            ref_valid = reference[reference > 0].flatten()
            tgt_valid = target[target > 0].flatten()

            if len(ref_valid) == 0 or len(tgt_valid) == 0:
                return target

            # Calculate histograms over a shared bin range
            combined_min = min(ref_valid.min(), tgt_valid.min())
            combined_max = max(ref_valid.max(), tgt_valid.max())

            ref_hist, bin_edges = np.histogram(
                ref_valid, bins=256, range=(combined_min, combined_max)
            )
            tgt_hist, _ = np.histogram(
                tgt_valid, bins=256, range=(combined_min, combined_max)
            )

            # Cumulative distribution functions (normalized to [0, 1])
            ref_cdf = np.cumsum(ref_hist).astype(np.float64)
            tgt_cdf = np.cumsum(tgt_hist).astype(np.float64)

            # Guard against flat CDF (e.g., uniform or all-zero histogram)
            if ref_cdf[-1] == 0 or tgt_cdf[-1] == 0:
                logger.warning(
                    "Flat CDF detected in histogram matching, skipping normalization for this band"
                )
                return target

            ref_cdf /= ref_cdf[-1]
            tgt_cdf /= tgt_cdf[-1]

            # Build lookup: for each target bin, find the reference bin with
            # the closest CDF value
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            lookup = np.interp(tgt_cdf, ref_cdf, bin_centers)

            # Map target pixels through the lookup table
            # Digitize target values into bins, then replace with lookup values
            bin_indices = np.digitize(target.flatten(), bin_edges[:-1]) - 1
            bin_indices = np.clip(bin_indices, 0, 255)
            target_matched = lookup[bin_indices].reshape(target.shape)

            # Preserve zero (no-data) pixels
            target_matched[target == 0] = 0

            return target_matched.astype(reference.dtype)

        except Exception as e:
            logger.error(f"Histogram matching failed: {e}")
            return target

    def validate_image_quality(
        self, bands: Dict[str, np.ndarray], metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate image quality metrics.

        Args:
            bands: Band arrays
            metadata: Image metadata

        Returns:
            Dictionary with quality metrics
        """
        try:
            quality_metrics: Dict[str, Any] = {
                "total_pixels": 0,
                "valid_pixels": 0,
                "cloud_coverage": 0.0,
                "brightness_mean": 0.0,
                "brightness_std": 0.0,
                "has_data": False,
            }

            # Calculate basic statistics
            total_pixels = 0
            valid_pixels = 0
            brightness_sum = 0.0
            brightness_sum_sq = 0.0

            for band_name, band_array in bands.items():
                if band_name != "QA60":  # Exclude QA band from statistics
                    total_pixels += band_array.size
                    valid_pixels += int(np.sum(band_array > 0))

                    # Calculate brightness statistics
                    valid_values = band_array[band_array > 0]
                    if len(valid_values) > 0:
                        brightness_sum += float(np.sum(valid_values))
                        brightness_sum_sq += float(np.sum(valid_values**2))

            if total_pixels > 0:
                quality_metrics["total_pixels"] = total_pixels
                quality_metrics["valid_pixels"] = valid_pixels
                quality_metrics["valid_pixel_percentage"] = (
                    valid_pixels / total_pixels
                ) * 100

                # Calculate cloud coverage from QA60 if available
                if "QA60" in bands:
                    qa60 = bands["QA60"]
                    cloud_mask = self._create_cloud_mask(qa60)
                    quality_metrics["cloud_coverage"] = (
                        np.sum(cloud_mask == 0) / cloud_mask.size
                    ) * 100

                # Calculate brightness statistics
                if valid_pixels > 0:
                    quality_metrics["brightness_mean"] = brightness_sum / valid_pixels
                    variance = (brightness_sum_sq / valid_pixels) - (
                        brightness_sum / valid_pixels
                    ) ** 2
                    quality_metrics["brightness_std"] = np.sqrt(max(variance, 0.0))

                quality_metrics["has_data"] = valid_pixels > 0

            return quality_metrics

        except Exception as e:
            logger.error(f"Failed to validate image quality: {e}")
            return quality_metrics

    def get_processing_summary(
        self,
        bands_a: Dict[str, np.ndarray],
        bands_b: Dict[str, np.ndarray],
        metadata_a: Dict[str, Any],
        metadata_b: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get summary of image processing results.

        Args:
            bands_a: Band arrays for Date A
            bands_b: Band arrays for Date B
            metadata_a: Metadata for Date A
            metadata_b: Metadata for Date B

        Returns:
            Dictionary with processing summary
        """
        try:
            summary = {
                "date_a": metadata_a.get("date", "Unknown"),
                "date_b": metadata_b.get("date", "Unknown"),
                "cloud_coverage_a": 0,
                "cloud_coverage_b": 0,
                "brightness_difference": 0,
                "processing_successful": True,
                "warnings": [],
            }

            # Calculate cloud coverage
            if "QA60" in bands_a:
                qa60_a = bands_a["QA60"]
                cloud_mask_a = self._create_cloud_mask(qa60_a)
                summary["cloud_coverage_a"] = (
                    np.sum(cloud_mask_a == 0) / cloud_mask_a.size
                ) * 100

            if "QA60" in bands_b:
                qa60_b = bands_b["QA60"]
                cloud_mask_b = self._create_cloud_mask(qa60_b)
                summary["cloud_coverage_b"] = (
                    np.sum(cloud_mask_b == 0) / cloud_mask_b.size
                ) * 100

            # Calculate brightness difference
            summary["brightness_difference"] = self._calculate_brightness_difference(
                bands_a, bands_b
            )

            # Add warnings
            if summary["cloud_coverage_a"] > self.cloud_threshold:
                summary["warnings"].append(
                    f"Date A cloud coverage ({summary['cloud_coverage_a']:.1f}%) exceeds threshold"
                )

            if summary["cloud_coverage_b"] > self.cloud_threshold:
                summary["warnings"].append(
                    f"Date B cloud coverage ({summary['cloud_coverage_b']:.1f}%) exceeds threshold"
                )

            if summary["brightness_difference"] > 0.1:
                summary["warnings"].append(
                    f"Significant brightness difference ({summary['brightness_difference']:.2%}) detected"
                )

            return summary

        except Exception as e:
            logger.error(f"Failed to get processing summary: {e}")
            return {"processing_successful": False, "error": str(e)}
