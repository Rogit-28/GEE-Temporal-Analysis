import json
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from satchange.cli import main


def test_analyze_dry_run_rejects_traversal_name(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("{}", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--config-file",
            str(cfg),
            "analyze",
            "--center",
            "13.08,80.27",
            "--size",
            "10",
            "--date-a",
            "2022-01-01",
            "--date-b",
            "2024-01-01",
            "--output",
            str(tmp_path),
            "--name",
            "..\\evil",
            "--dry-run",
        ],
    )
    assert result.exit_code == 1
    assert "Name must not contain path separators" in result.output


def test_export_rejects_legacy_pickle_band_files(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("{}", encoding="utf-8")

    prefix = "sample_2022-01-01_2024-01-01"
    (tmp_path / f"{prefix}_metadata.json").write_text(
        json.dumps(
            {
                "center_lat": 13.08,
                "center_lon": 80.27,
                "date_a": {"date": "2022-01-01T00:00:00"},
                "date_b": {"date": "2024-01-01T00:00:00"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / f"{prefix}_change_stats.json").write_text(
        json.dumps({"total_change": {"percent": 0.0, "pixels": 0, "area_km2": 0.0}}),
        encoding="utf-8",
    )
    np.save(tmp_path / f"{prefix}_classification.npy", np.zeros((4, 4), dtype=np.uint8))

    legacy_a = {
        "B4": np.ones((4, 4), dtype=np.float32),
        "B3": np.ones((4, 4), dtype=np.float32),
        "B8": np.ones((4, 4), dtype=np.float32),
        "B11": np.ones((4, 4), dtype=np.float32),
    }
    legacy_b = {
        "B4": np.ones((4, 4), dtype=np.float32),
        "B3": np.ones((4, 4), dtype=np.float32),
        "B8": np.ones((4, 4), dtype=np.float32),
        "B11": np.ones((4, 4), dtype=np.float32),
    }
    np.save(
        tmp_path / f"{prefix}_bands_a.npy",
        np.array(legacy_a, dtype=object),
        allow_pickle=True,
    )
    np.save(
        tmp_path / f"{prefix}_bands_b.npy",
        np.array(legacy_b, dtype=object),
        allow_pickle=True,
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--config-file",
            str(cfg),
            "export",
            "--result",
            str(tmp_path),
            "--format",
            "static",
        ],
    )
    assert result.exit_code == 1
    assert "Legacy .npy dict format is unsupported" in result.output
