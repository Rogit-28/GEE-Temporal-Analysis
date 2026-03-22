import numpy as np
import pytest

from satchange.config import Config
from satchange.image_processor import ImageProcessor, ImageProcessingError


def test_low_valid_coverage_raises() -> None:
    cfg = Config()
    processor = ImageProcessor(cfg)

    shape = (10, 10)
    # Set all pixels cloudy by using QA60 cloud bit (bit 10)
    qa_all_cloud = np.full(shape, 1024, dtype=np.uint16)
    bands_a = {
        "B4": np.ones(shape, dtype=np.float32),
        "B3": np.ones(shape, dtype=np.float32),
        "B8": np.ones(shape, dtype=np.float32),
        "B11": np.ones(shape, dtype=np.float32),
        "QA60": qa_all_cloud,
    }
    bands_b = {k: v.copy() for k, v in bands_a.items()}

    metadata = {
        "width": 10,
        "height": 10,
        "crs": "EPSG:4326",
        "transform": (1, 0, 0, 0, -1, 0),
    }

    with pytest.raises(ImageProcessingError):
        processor.preprocess_image_pair(bands_a, bands_b, metadata, metadata)

def test_valid_ratio_not_modified_by_histogram_matching() -> None:
    cfg = Config()
    processor = ImageProcessor(cfg)

    shape = (50, 50)
    qa = np.zeros(shape, dtype=np.uint16)
    qa[:25, :] = 1024  # 50% cloudy

    bands_a = {
        "B4": np.full(shape, 0.2, dtype=np.float32),
        "B3": np.full(shape, 0.2, dtype=np.float32),
        "B8": np.full(shape, 0.2, dtype=np.float32),
        "B11": np.full(shape, 0.2, dtype=np.float32),
        "QA60": qa,
    }
    # Brightness-shift Date B to force histogram matching path
    bands_b = {
        "B4": np.full(shape, 0.9, dtype=np.float32),
        "B3": np.full(shape, 0.9, dtype=np.float32),
        "B8": np.full(shape, 0.9, dtype=np.float32),
        "B11": np.full(shape, 0.9, dtype=np.float32),
        "QA60": qa.copy(),
    }

    metadata = {
        "width": 50,
        "height": 50,
        "crs": "EPSG:4326",
        "transform": (1, 0, 0, 0, -1, 0),
    }

    processed_a, processed_b = processor.preprocess_image_pair(
        bands_a, bands_b, metadata, metadata
    )

    assert float(processed_a["VALID_RATIO"]) == pytest.approx(0.5, abs=1e-6)
    assert float(processed_b["VALID_RATIO"]) == pytest.approx(0.5, abs=1e-6)
    assert float(np.mean(processed_a["VALID_MASK"])) == pytest.approx(0.5, abs=1e-6)
    assert float(np.mean(processed_b["VALID_MASK"])) == pytest.approx(0.5, abs=1e-6)

