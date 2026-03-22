import os

import pytest

from satchange.utils import sanitize_output_name, safe_join, check_disk_space


def test_sanitize_output_name_rejects_traversal() -> None:
    with pytest.raises(ValueError):
        sanitize_output_name("..\\evil")
    with pytest.raises(ValueError):
        sanitize_output_name("../evil")


def test_sanitize_output_name_normalizes_symbols() -> None:
    assert sanitize_output_name("chennai center@2024") == "chennai_center_2024"


def test_safe_join_confines_to_base(tmp_path) -> None:
    base = str(tmp_path)
    child = safe_join(base, "a", "b.txt")
    assert child.startswith(base)
    with pytest.raises(ValueError):
        safe_join(base, "..", "escape.txt")


def test_check_disk_space_invalid_path_fails_safe() -> None:
    # Use a path that should not exist and cannot be resolved for disk usage.
    result = check_disk_space(
        os.path.join("Z:\\", "nonexistent", "path"), required_mb=1.0
    )
    assert result["sufficient"] is False
