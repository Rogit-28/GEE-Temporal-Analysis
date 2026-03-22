import numpy as np

from satchange.change_detector import ChangeDetector


def _mock_bands(value_a: float, value_b: float):
    size = (4, 4)
    a = np.full(size, value_a, dtype=np.float32)
    b = np.full(size, value_b, dtype=np.float32)
    qa = np.zeros(size, dtype=np.uint16)
    valid = np.ones(size, dtype=np.uint8)
    valid[0, 0] = 0
    return (
        {"B4": a, "B3": a, "B8": a, "B11": a, "QA60": qa, "VALID_MASK": valid},
        {"B4": b, "B3": b, "B8": b, "B11": b, "QA60": qa, "VALID_MASK": valid},
    )


def test_invalid_pixels_excluded_from_change_masks() -> None:
    bands_a, bands_b = _mock_bands(0.2, 0.8)
    detector = ChangeDetector(threshold=0.1)
    summary = detector.get_change_summary(bands_a, bands_b, "all")
    classification = summary["classification"]
    assert classification[0, 0] == 0


def test_scalar_aux_key_order_does_not_break_mask_broadcast() -> None:
    size = (5, 5)
    qa = np.zeros(size, dtype=np.uint16)
    # Intentionally place VALID_RATIO first to mimic scalar-first NPZ load order edge case
    bands_a = {
        "VALID_RATIO": np.array(0.8, dtype=np.float32),
        "B4": np.full(size, 0.2, dtype=np.float32),
        "B3": np.full(size, 0.2, dtype=np.float32),
        "B8": np.full(size, 0.2, dtype=np.float32),
        "B11": np.full(size, 0.2, dtype=np.float32),
        "QA60": qa,
        "VALID_MASK": np.ones(size, dtype=np.uint8),
    }
    bands_b = {
        "VALID_RATIO": np.array(0.9, dtype=np.float32),
        "B4": np.full(size, 0.7, dtype=np.float32),
        "B3": np.full(size, 0.7, dtype=np.float32),
        "B8": np.full(size, 0.7, dtype=np.float32),
        "B11": np.full(size, 0.7, dtype=np.float32),
        "QA60": qa,
        "VALID_MASK": np.ones(size, dtype=np.uint8),
    }

    detector = ChangeDetector(threshold=0.1)
    summary = detector.get_change_summary(bands_a, bands_b, "all")
    assert summary["classification"].shape == size
