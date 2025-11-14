"""
Comprehensive tests for satchange CLI interface.

Tests all CLI commands, error handling, and helper functions using
Click's CliRunner and mocked external dependencies.
"""

import json
import os
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from click.testing import CliRunner

from satchange.cli import main, format_location_name, generate_output_prefix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    """Provide a Click test runner."""
    return CliRunner()


@contextmanager
def _noop_spinner(desc):
    yield None


@contextmanager
def _noop_progress_bar(desc, total):
    yield lambda step=1: None


def _make_cloud_check(date, is_good=True, found=True, local_pct=5.0, scene_pct=10.0):
    """Helper to build a cloud-check result dict."""
    return {
        "found": found,
        "is_good": is_good,
        "local_cloud_pct": local_pct,
        "scene_cloud_pct": scene_pct,
        "date": date,
        "image_id": f"COPERNICUS/S2_SR_HARMONIZED/{date.replace('-', '')}T000000",
    }


def _make_band_arrays():
    """Helper: small band-array dict matching Sentinel-2 download shape."""
    return {
        "B4": np.random.rand(100, 100).astype(np.float32) * 1000,
        "B3": np.random.rand(100, 100).astype(np.float32) * 1000,
        "B2": np.random.rand(100, 100).astype(np.float32) * 1000,
        "B8": np.random.rand(100, 100).astype(np.float32) * 1000,
        "B11": np.random.rand(50, 50).astype(np.float32) * 1000,
        "QA60": np.zeros((100, 100), dtype=np.uint16),
    }


def _make_change_summary_all():
    """Return a change summary dict for change_type='all'."""
    classification = np.zeros((100, 100), dtype=np.uint8)
    return {
        "classification": classification,
        "statistics": {
            "total_change": {"pixels": 100, "percent": 1.0, "area_km2": 0.01},
            "change_types": {
                "vegetation": {"pixels": 40, "percent": 0.4},
                "water": {"pixels": 30, "percent": 0.3},
                "urban": {"pixels": 30, "percent": 0.3},
            },
        },
        "summary": "Test summary",
    }


def _make_change_summary_single(change_type):
    """Return a change summary dict for a single change type."""
    size = 100
    result = {
        "change_results": {
            "growth_mask": np.zeros((size, size), dtype=bool),
            "loss_mask": np.zeros((size, size), dtype=bool),
            "expansion_mask": np.zeros((size, size), dtype=bool),
            "reduction_mask": np.zeros((size, size), dtype=bool),
            "development_mask": np.zeros((size, size), dtype=bool),
            "decline_mask": np.zeros((size, size), dtype=bool),
        },
        "statistics": {
            "changed_pixels": 50,
            "change_percentage": 0.5,
            "growth_pixels": 30,
            "loss_pixels": 20,
            "expansion_pixels": 25,
            "reduction_pixels": 25,
            "development_pixels": 10,
            "decline_pixels": 5,
        },
        "summary": f"Test {change_type} summary",
    }
    return result


@pytest.fixture
def mock_gee_client():
    """A fully wired GEEClient mock."""
    client = MagicMock()
    client.check_local_cloud.side_effect = lambda date, *a, **kw: _make_cloud_check(
        date
    )
    client.create_bbox.return_value = MagicMock()
    client.get_image_info.return_value = {"date": "2022-02-04", "cloud_coverage": 5.0}
    client.download_image.return_value = (
        _make_band_arrays(),
        {"crs": "EPSG:4326", "date": "2022-02-04"},
    )
    client.query_imagery.return_value = MagicMock()
    client.get_scenes_metadata.return_value = [
        {"date": "2022-02-04", "cloud_coverage": 5.0},
        {"date": "2022-02-10", "cloud_coverage": 8.2},
        {"date": "2022-02-15", "cloud_coverage": 12.0},
    ]
    return client


# ======================================================================
# Helper function tests
# ======================================================================


class TestFormatLocationName:
    """Tests for format_location_name helper."""

    def test_basic(self):
        assert format_location_name(13.0827, 80.2707) == "13.0827_80.2707"

    def test_trailing_zeros_stripped(self):
        assert format_location_name(13.0, 80.0) == "13_80"

    def test_negative_coordinates(self):
        result = format_location_name(-33.8688, 151.2093)
        assert result == "-33.8688_151.2093"

    def test_zero_coordinates(self):
        assert format_location_name(0.0, 0.0) == "0_0"

    def test_high_precision_truncated(self):
        # 4 decimal places max, trailing zeros stripped
        result = format_location_name(13.08270000, 80.27070000)
        assert result == "13.0827_80.2707"


class TestGenerateOutputPrefix:
    """Tests for generate_output_prefix helper."""

    def test_with_name(self):
        prefix = generate_output_prefix(
            "chennai", 13.0, 80.0, "2022-01-01", "2022-06-01"
        )
        assert prefix == "chennai_2022-01-01_2022-06-01"

    def test_without_name(self):
        prefix = generate_output_prefix(
            None, 13.0827, 80.2707, "2022-01-01", "2022-06-01"
        )
        assert "13.0827_80.2707" in prefix
        assert "2022-01-01" in prefix
        assert "2022-06-01" in prefix

    def test_datetime_with_T(self):
        prefix = generate_output_prefix(
            "loc", 0, 0, "2022-01-01T12:00:00", "2022-06-01T00:00:00"
        )
        assert prefix == "loc_2022-01-01_2022-06-01"


# ======================================================================
# main group tests
# ======================================================================


class TestMainGroup:
    """Tests for the top-level CLI group."""

    @patch("satchange.cli.Config")
    def test_help(self, MockConfig, runner):
        MockConfig.return_value = MagicMock(
            is_authenticated=MagicMock(return_value=False)
        )
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "SatChange" in result.output

    @patch("satchange.cli.Config")
    def test_version(self, MockConfig, runner):
        MockConfig.return_value = MagicMock(
            is_authenticated=MagicMock(return_value=False)
        )
        result = runner.invoke(main, ["--version"])
        # click.version_option() may fail if package is not installed via pip;
        # in that case exit_code=1 and the error message mentions the package.
        # When installed, exit_code=0 and output contains the version string.
        if result.exit_code == 0:
            assert "version" in result.output.lower() or "0.1.0" in result.output
        else:
            # Not installed — version_option raises RuntimeError about package name
            assert result.exit_code == 1

    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_verbose_flag(self, MockConfig, MockGEE, runner):
        cfg = MagicMock(is_authenticated=MagicMock(return_value=False))
        MockConfig.return_value = cfg
        # --verbose with a subcommand shouldn't crash
        result = runner.invoke(main, ["--verbose", "--help"])
        assert result.exit_code == 0
        assert "SatChange" in result.output


# ======================================================================
# config init tests
# ======================================================================


class TestConfigInit:
    """Tests for the 'config init' command."""

    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_success(self, MockConfig, MockGEE, runner, tmp_path):
        cfg = MagicMock(is_authenticated=MagicMock(return_value=False))
        cfg.get.return_value = "test@sa.iam.gserviceaccount.com"
        MockConfig.return_value = cfg

        # Create a fake key file
        key_file = tmp_path / "key.json"
        key_file.write_text('{"client_email": "test@sa.iam.gserviceaccount.com"}')

        result = runner.invoke(
            main,
            [
                "config",
                "init",
                "--service-account-key",
                str(key_file),
                "--project-id",
                "my-proj",
            ],
        )
        assert result.exit_code == 0
        assert "[OK]" in result.output
        cfg.initialize_auth.assert_called_once_with(str(key_file), "my-proj")

    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_missing_params_exits_1(self, MockConfig, MockGEE, runner):
        cfg = MagicMock(is_authenticated=MagicMock(return_value=False))
        MockConfig.return_value = cfg

        result = runner.invoke(main, ["config", "init"])
        assert result.exit_code == 1
        assert "provide both" in result.output

    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_exception_exits_1(self, MockConfig, MockGEE, runner, tmp_path):
        cfg = MagicMock(is_authenticated=MagicMock(return_value=False))
        cfg.initialize_auth.side_effect = Exception("disk full")
        MockConfig.return_value = cfg

        key_file = tmp_path / "key.json"
        key_file.write_text("{}")

        result = runner.invoke(
            main,
            [
                "config",
                "init",
                "--service-account-key",
                str(key_file),
                "--project-id",
                "proj",
            ],
        )
        assert result.exit_code == 1
        assert "failed" in result.output.lower() or "ERROR" in result.output

    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_keyboard_interrupt_exits_130(self, MockConfig, MockGEE, runner, tmp_path):
        cfg = MagicMock(is_authenticated=MagicMock(return_value=False))
        cfg.initialize_auth.side_effect = KeyboardInterrupt()
        MockConfig.return_value = cfg

        key_file = tmp_path / "key.json"
        key_file.write_text("{}")

        result = runner.invoke(
            main,
            [
                "config",
                "init",
                "--service-account-key",
                str(key_file),
                "--project-id",
                "proj",
            ],
        )
        assert result.exit_code == 130


# ======================================================================
# inspect command tests
# ======================================================================


class TestInspect:
    """Tests for the 'inspect' command."""

    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_success_shows_scene_count(
        self, MockConfig, MockGEE, runner, mock_gee_client
    ):
        cfg = MagicMock(is_authenticated=MagicMock(return_value=True))
        MockConfig.return_value = cfg
        MockGEE.return_value = mock_gee_client

        result = runner.invoke(
            main,
            [
                "inspect",
                "--center",
                "13.0827,80.2707",
                "--date-range",
                "2022-01-01:2022-12-31",
            ],
        )
        assert result.exit_code == 0
        assert "Found 3 clear scenes" in result.output

    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_not_authenticated_exits_1(self, MockConfig, MockGEE, runner):
        cfg = MagicMock(is_authenticated=MagicMock(return_value=False))
        MockConfig.return_value = cfg

        result = runner.invoke(
            main,
            [
                "inspect",
                "--center",
                "13.0827,80.2707",
                "--date-range",
                "2022-01-01:2022-12-31",
            ],
        )
        assert result.exit_code == 1
        assert "Not authenticated" in result.output

    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_keyboard_interrupt_exits_130(
        self, MockConfig, MockGEE, runner, mock_gee_client
    ):
        cfg = MagicMock(is_authenticated=MagicMock(return_value=True))
        MockConfig.return_value = cfg
        mock_gee_client.query_imagery.side_effect = KeyboardInterrupt()
        MockGEE.return_value = mock_gee_client

        result = runner.invoke(
            main,
            [
                "inspect",
                "--center",
                "13.0827,80.2707",
                "--date-range",
                "2022-01-01:2022-12-31",
            ],
        )
        assert result.exit_code == 130

    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_generic_exception_exits_1(
        self, MockConfig, MockGEE, runner, mock_gee_client
    ):
        cfg = MagicMock(is_authenticated=MagicMock(return_value=True))
        MockConfig.return_value = cfg
        mock_gee_client.query_imagery.side_effect = RuntimeError("timeout")
        MockGEE.return_value = mock_gee_client

        result = runner.invoke(
            main,
            [
                "inspect",
                "--center",
                "13.0827,80.2707",
                "--date-range",
                "2022-01-01:2022-12-31",
            ],
        )
        assert result.exit_code == 1
        assert "Inspection failed" in result.output


# ======================================================================
# analyze command tests — shared helpers
# ======================================================================

# Common CLI args used for analyze invocations
_ANALYZE_BASE_ARGS = [
    "analyze",
    "--center",
    "13.0827,80.2707",
    "--size",
    "100",
    "--date-a",
    "2022-02-04",
    "--date-b",
    "2024-10-26",
    "--output",
]


def _patch_analyze_deps(
    config_authenticated=True,
    gee_client=None,
    cloud_check_a=None,
    cloud_check_b=None,
    change_type="all",
    disk_space_ok=True,
):
    """
    Return a list of patch context-managers for analyze dependencies.

    Usage:
        with ExitStack() as stack:
            mocks = {k: stack.enter_context(v) for k, v in _patch_analyze_deps(...).items()}
    """
    if gee_client is None:
        gee_client = MagicMock()
        # Default cloud checks: both good
        if cloud_check_a is None:
            cloud_check_a = _make_cloud_check("2022-02-04")
        if cloud_check_b is None:
            cloud_check_b = _make_cloud_check("2024-10-26")

        def _check_cloud(date, *a, **kw):
            if date == "2022-02-04":
                return cloud_check_a
            return cloud_check_b

        gee_client.check_local_cloud.side_effect = _check_cloud
        gee_client.create_bbox.return_value = MagicMock()
        gee_client.get_image_info.return_value = {
            "date": "2022-02-04",
            "cloud_coverage": 5.0,
        }
        gee_client.download_image.return_value = (
            _make_band_arrays(),
            {"crs": "EPSG:4326", "date": "2022-02-04"},
        )

    # Config
    cfg = MagicMock()
    cfg.is_authenticated.return_value = config_authenticated

    # Cache
    mock_cache_inst = MagicMock()
    mock_cache_inst.get_image_with_cache.return_value = (
        {
            "arrays": _make_band_arrays(),
            "metadata": {"crs": "EPSG:4326", "date": "2022-02-04"},
        },
        False,
    )
    mock_cache_inst.get_cache_stats.return_value = {
        "hit_rate": 50.0,
        "usage_percent": 10.0,
        "total_items": 5,
        "size_formatted": "120 MB",
        "hits": 10,
        "misses": 10,
        "evictions": 0,
    }

    # Image processor
    mock_ip_inst = MagicMock()
    mock_ip_inst.preprocess_image_pair.return_value = (
        _make_band_arrays(),
        _make_band_arrays(),
    )
    mock_ip_inst.get_processing_summary.return_value = {
        "processing_successful": True,
        "warnings": [],
    }

    # Change detector
    mock_cd_inst = MagicMock()
    if change_type == "all":
        mock_cd_inst.get_change_summary.return_value = _make_change_summary_all()
    else:
        mock_cd_inst.get_change_summary.return_value = _make_change_summary_single(
            change_type
        )

    # Visualization manager
    mock_viz_inst = MagicMock()
    mock_viz_inst.generate_all_outputs.return_value = {
        "static": "/tmp/out/static.png",
        "interactive": "/tmp/out/interactive.html",
        "geotiff": "/tmp/out/output.tif",
    }

    patches = {
        "Config": patch("satchange.cli.Config", return_value=cfg),
        "GEEClient": patch("satchange.cli.GEEClient", return_value=gee_client),
        "CacheManager": patch(
            "satchange.cli.CacheManager", return_value=mock_cache_inst
        ),
        "ImageProcessor": patch(
            "satchange.cli.ImageProcessor", return_value=mock_ip_inst
        ),
        "ChangeDetector": patch(
            "satchange.cli.ChangeDetector", return_value=mock_cd_inst
        ),
        "VisualizationManager": patch(
            "satchange.cli.VisualizationManager", return_value=mock_viz_inst
        ),
        "spinner": patch("satchange.cli.spinner", side_effect=_noop_spinner),
        "progress_bar": patch(
            "satchange.cli.progress_bar", side_effect=_noop_progress_bar
        ),
        "check_disk_space": patch(
            "satchange.utils.check_disk_space",
            return_value={
                "available_mb": 500.0 if disk_space_ok else 10.0,
                "required_mb": 50.0,
                "sufficient": disk_space_ok,
            },
        ),
    }
    return patches


# ======================================================================
# analyze command tests
# ======================================================================


class TestAnalyzeHappyPath:
    """Tests for analyze command — successful runs."""

    def test_basic_analyze_all_steps(self, runner, tmp_path):
        """Full pipeline: cloud-check → download → detect → visualize."""
        from contextlib import ExitStack

        with ExitStack() as stack:
            mocks = {
                k: stack.enter_context(v) for k, v in _patch_analyze_deps().items()
            }
            result = runner.invoke(main, _ANALYZE_BASE_ARGS + [str(tmp_path)])

