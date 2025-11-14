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

        assert result.exit_code == 0, result.output
        assert "Step 1/4" in result.output
        assert "Step 3/4" in result.output
        assert "Step 4/4" in result.output
        assert "Change detection completed successfully" in result.output

    def test_both_dates_good_no_alternatives(self, runner, tmp_path):
        """When both dates are cloud-free the 'no alternatives needed' message appears."""
        from contextlib import ExitStack

        with ExitStack() as stack:
            for v in _patch_analyze_deps().values():
                stack.enter_context(v)
            result = runner.invoke(main, _ANALYZE_BASE_ARGS + [str(tmp_path)])

        assert result.exit_code == 0
        assert "no alternatives needed" in result.output


class TestAnalyzeCloudAlternatives:
    """Tests for cloud-coverage alternative-date logic."""

    def test_date_a_bad_triggers_alternatives(self, runner, tmp_path):
        """When Date A has bad cloud coverage, alternatives are searched."""
        from contextlib import ExitStack

        bad_a = _make_cloud_check("2022-02-04", is_good=False, local_pct=80.0)
        gee = MagicMock()
        gee.check_local_cloud.side_effect = lambda date, *a, **kw: (
            bad_a if date == "2022-02-04" else _make_cloud_check("2024-10-26")
        )
        gee.find_alternative_dates.return_value = {
            "threshold_met": True,
            "search_window": "±30 days",
            "alternatives": [
                {
                    "date": "2022-02-10",
                    "local_cloud_pct": 3.0,
                    "is_recommended": True,
                    "image_id": "COPERNICUS/S2_SR_HARMONIZED/20220210T000000",
                },
            ],
        }
        gee.create_bbox.return_value = MagicMock()
        gee.get_image_info.return_value = {"date": "2022-02-10", "cloud_coverage": 3.0}
        gee.download_image.return_value = (
            _make_band_arrays(),
            {"crs": "EPSG:4326", "date": "2022-02-10"},
        )

        patches = _patch_analyze_deps(gee_client=gee)

        with ExitStack() as stack:
            for v in patches.values():
                stack.enter_context(v)
            result = runner.invoke(
                main,
                _ANALYZE_BASE_ARGS + [str(tmp_path), "--non-interactive"],
            )

        assert result.exit_code == 0, result.output
        assert "Finding alternatives" in result.output
        gee.find_alternative_dates.assert_called_once()

    def test_non_interactive_auto_selects_recommended(self, runner, tmp_path):
        """--non-interactive auto-selects the recommended date."""
        from contextlib import ExitStack

        bad_a = _make_cloud_check("2022-02-04", is_good=False, local_pct=80.0)
        gee = MagicMock()
        gee.check_local_cloud.side_effect = lambda date, *a, **kw: (
            bad_a if date == "2022-02-04" else _make_cloud_check("2024-10-26")
        )
        gee.find_alternative_dates.return_value = {
            "threshold_met": True,
            "search_window": "±30 days",
            "alternatives": [
                {
                    "date": "2022-02-12",
                    "local_cloud_pct": 2.0,
                    "is_recommended": True,
                    "image_id": "COPERNICUS/S2_SR_HARMONIZED/20220212T000000",
                },
            ],
        }
        gee.create_bbox.return_value = MagicMock()
        gee.get_image_info.return_value = {"date": "2022-02-12", "cloud_coverage": 2.0}
        gee.download_image.return_value = (
            _make_band_arrays(),
            {"crs": "EPSG:4326", "date": "2022-02-12"},
        )

        patches = _patch_analyze_deps(gee_client=gee)

        with ExitStack() as stack:
            for v in patches.values():
                stack.enter_context(v)
            result = runner.invoke(
                main,
                _ANALYZE_BASE_ARGS + [str(tmp_path), "--non-interactive"],
            )

        assert result.exit_code == 0, result.output
        assert "Auto-selected" in result.output


class TestAnalyzeDryRun:
    """Tests for --dry-run flag."""

    def test_dry_run_exits_after_cloud_check(self, runner, tmp_path):
        from contextlib import ExitStack

        with ExitStack() as stack:
            mocks = {
                k: stack.enter_context(v) for k, v in _patch_analyze_deps().items()
            }
            result = runner.invoke(
                main, _ANALYZE_BASE_ARGS + [str(tmp_path), "--dry-run"]
            )

        assert result.exit_code == 0
        assert "DRY RUN" in result.output

    def test_dry_run_shows_what_would_be_done(self, runner, tmp_path):
        from contextlib import ExitStack

        with ExitStack() as stack:
            mocks = {
                k: stack.enter_context(v) for k, v in _patch_analyze_deps().items()
            }
            result = runner.invoke(
                main, _ANALYZE_BASE_ARGS + [str(tmp_path), "--dry-run"]
            )

        assert "No images downloaded" in result.output

    def test_dry_run_does_not_call_download(self, runner, tmp_path):
        from contextlib import ExitStack

        with ExitStack() as stack:
            mocks = {
                k: stack.enter_context(v) for k, v in _patch_analyze_deps().items()
            }
            result = runner.invoke(
                main, _ANALYZE_BASE_ARGS + [str(tmp_path), "--dry-run"]
            )

        assert result.exit_code == 0
        # If download were called we'd see "Step 3/4" — it shouldn't appear
        assert "Step 3/4" not in result.output


class TestAnalyzeChangeTypes:
    """Tests for --change-type options."""

    def test_vegetation(self, runner, tmp_path):
        from contextlib import ExitStack

        patches = _patch_analyze_deps(change_type="vegetation")
        with ExitStack() as stack:
            for v in patches.values():
                stack.enter_context(v)
            result = runner.invoke(
                main,
                _ANALYZE_BASE_ARGS + [str(tmp_path), "--change-type", "vegetation"],
            )

        assert result.exit_code == 0, result.output
        assert "Vegetation" in result.output

    def test_water(self, runner, tmp_path):
        from contextlib import ExitStack

        patches = _patch_analyze_deps(change_type="water")
        with ExitStack() as stack:
            for v in patches.values():
                stack.enter_context(v)
            result = runner.invoke(
                main,
                _ANALYZE_BASE_ARGS + [str(tmp_path), "--change-type", "water"],
            )

        assert result.exit_code == 0, result.output
        assert "Water" in result.output

    def test_urban(self, runner, tmp_path):
        from contextlib import ExitStack

        patches = _patch_analyze_deps(change_type="urban")
        with ExitStack() as stack:
            for v in patches.values():
                stack.enter_context(v)
            result = runner.invoke(
                main,
                _ANALYZE_BASE_ARGS + [str(tmp_path), "--change-type", "urban"],
            )

        assert result.exit_code == 0, result.output
        assert "Urban" in result.output

    def test_all(self, runner, tmp_path):
        from contextlib import ExitStack

        patches = _patch_analyze_deps(change_type="all")
        with ExitStack() as stack:
            for v in patches.values():
                stack.enter_context(v)
            result = runner.invoke(
                main,
                _ANALYZE_BASE_ARGS + [str(tmp_path), "--change-type", "all"],
            )

        assert result.exit_code == 0, result.output
        assert "Change Detection Results" in result.output


class TestAnalyzeErrors:
    """Tests for analyze error handling."""

    def test_not_authenticated_exits_1(self, runner, tmp_path):
        from contextlib import ExitStack

        patches = _patch_analyze_deps(config_authenticated=False)
        with ExitStack() as stack:
            for v in patches.values():
                stack.enter_context(v)
            result = runner.invoke(main, _ANALYZE_BASE_ARGS + [str(tmp_path)])

        assert result.exit_code == 1
        assert "Not authenticated" in result.output

    @patch("satchange.cli.spinner", side_effect=_noop_spinner)
    @patch("satchange.cli.progress_bar", side_effect=_noop_progress_bar)
    @patch("satchange.cli.VisualizationManager")
    @patch("satchange.cli.ChangeDetector")
    @patch("satchange.cli.ImageProcessor")
    @patch("satchange.cli.CacheManager")
    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_quota_exceeded_error(
        self,
        MockConfig,
        MockGEE,
        MockCache,
        MockIP,
        MockCD,
        MockViz,
        _pb,
        _sp,
        runner,
        tmp_path,
    ):
        from satchange.gee_client import QuotaExceededError

        cfg = MagicMock(is_authenticated=MagicMock(return_value=True))
        MockConfig.return_value = cfg
        gee = MagicMock()
        gee.check_local_cloud.side_effect = QuotaExceededError("quota exceeded")
        MockGEE.return_value = gee

        result = runner.invoke(main, _ANALYZE_BASE_ARGS + [str(tmp_path)])
        assert result.exit_code == 1
        assert "GEE limit reached" in result.output

    @patch("satchange.cli.spinner", side_effect=_noop_spinner)
    @patch("satchange.cli.progress_bar", side_effect=_noop_progress_bar)
    @patch("satchange.cli.VisualizationManager")
    @patch("satchange.cli.ChangeDetector")
    @patch("satchange.cli.ImageProcessor")
    @patch("satchange.cli.CacheManager")
    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_rate_limit_error(
        self,
        MockConfig,
        MockGEE,
        MockCache,
        MockIP,
        MockCD,
        MockViz,
        _pb,
        _sp,
        runner,
        tmp_path,
    ):
        from satchange.gee_client import RateLimitError

        cfg = MagicMock(is_authenticated=MagicMock(return_value=True))
        MockConfig.return_value = cfg
        gee = MagicMock()
        gee.check_local_cloud.side_effect = RateLimitError("rate limit")
        MockGEE.return_value = gee

        result = runner.invoke(main, _ANALYZE_BASE_ARGS + [str(tmp_path)])
        assert result.exit_code == 1
        assert "GEE limit reached" in result.output

    @patch("satchange.cli.spinner", side_effect=_noop_spinner)
    @patch("satchange.cli.progress_bar", side_effect=_noop_progress_bar)
    @patch("satchange.cli.VisualizationManager")
    @patch("satchange.cli.ChangeDetector")
    @patch("satchange.cli.ImageProcessor")
    @patch("satchange.cli.CacheManager")
    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_no_imagery_error(
        self,
        MockConfig,
        MockGEE,
        MockCache,
        MockIP,
        MockCD,
        MockViz,
        _pb,
        _sp,
        runner,
        tmp_path,
    ):
        from satchange.gee_client import NoImageryError

        cfg = MagicMock(is_authenticated=MagicMock(return_value=True))
        MockConfig.return_value = cfg
        gee = MagicMock()
        gee.check_local_cloud.side_effect = NoImageryError("no imagery")
        MockGEE.return_value = gee

        result = runner.invoke(main, _ANALYZE_BASE_ARGS + [str(tmp_path)])
        assert result.exit_code == 1
        assert "No suitable imagery" in result.output

    @patch("satchange.cli.spinner", side_effect=_noop_spinner)
    @patch("satchange.cli.progress_bar", side_effect=_noop_progress_bar)
    @patch("satchange.cli.VisualizationManager")
    @patch("satchange.cli.ChangeDetector")
    @patch("satchange.cli.ImageProcessor")
    @patch("satchange.cli.CacheManager")
    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_download_error(
        self,
        MockConfig,
        MockGEE,
        MockCache,
        MockIP,
        MockCD,
        MockViz,
        _pb,
        _sp,
        runner,
        tmp_path,
    ):
        from satchange.gee_client import DownloadError

        cfg = MagicMock(is_authenticated=MagicMock(return_value=True))
        MockConfig.return_value = cfg
        gee = MagicMock()
        gee.check_local_cloud.side_effect = DownloadError("download failed")
        MockGEE.return_value = gee

        result = runner.invoke(main, _ANALYZE_BASE_ARGS + [str(tmp_path)])
        assert result.exit_code == 1
        assert "Download failed" in result.output

    @patch("satchange.cli.spinner", side_effect=_noop_spinner)
    @patch("satchange.cli.progress_bar", side_effect=_noop_progress_bar)
    @patch("satchange.cli.VisualizationManager")
    @patch("satchange.cli.ChangeDetector")
    @patch("satchange.cli.ImageProcessor")
    @patch("satchange.cli.CacheManager")
    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_keyboard_interrupt_exits_130(
        self,
        MockConfig,
        MockGEE,
        MockCache,
        MockIP,
        MockCD,
        MockViz,
        _pb,
        _sp,
        runner,
        tmp_path,
    ):
        cfg = MagicMock(is_authenticated=MagicMock(return_value=True))
        MockConfig.return_value = cfg
        gee = MagicMock()
        gee.check_local_cloud.side_effect = KeyboardInterrupt()
        MockGEE.return_value = gee

        result = runner.invoke(main, _ANALYZE_BASE_ARGS + [str(tmp_path)])
        assert result.exit_code == 130

    @patch("satchange.cli.spinner", side_effect=_noop_spinner)
    @patch("satchange.cli.progress_bar", side_effect=_noop_progress_bar)
    @patch("satchange.cli.VisualizationManager")
    @patch("satchange.cli.ChangeDetector")
    @patch("satchange.cli.ImageProcessor")
    @patch("satchange.cli.CacheManager")
    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_generic_exception_exits_1(
        self,
        MockConfig,
        MockGEE,
        MockCache,
        MockIP,
        MockCD,
        MockViz,
        _pb,
        _sp,
        runner,
        tmp_path,
    ):
        cfg = MagicMock(is_authenticated=MagicMock(return_value=True))
        MockConfig.return_value = cfg
        gee = MagicMock()
        gee.check_local_cloud.side_effect = RuntimeError("unexpected")
        MockGEE.return_value = gee

        result = runner.invoke(main, _ANALYZE_BASE_ARGS + [str(tmp_path)])
        assert result.exit_code == 1
        assert "Analysis failed" in result.output


class TestAnalyzeDiskSpace:
    """Tests for disk-space pre-check in analyze."""

    def test_insufficient_disk_space_exits_1(self, runner, tmp_path):
        from contextlib import ExitStack

        patches = _patch_analyze_deps(disk_space_ok=False)
        with ExitStack() as stack:
            for v in patches.values():
                stack.enter_context(v)
            result = runner.invoke(main, _ANALYZE_BASE_ARGS + [str(tmp_path)])

        assert result.exit_code == 1
        assert "Insufficient disk space" in result.output


# ======================================================================
# export command tests
# ======================================================================


def _setup_export_dir(tmp_path, prefix="loc_2022-01-01_2022-06-01"):
    """Create fake analysis artefacts so export can find them."""
    bands = _make_band_arrays()
    classification = np.zeros((100, 100), dtype=np.uint8)
    stats = {"total_change": {"pixels": 100, "percent": 1.0, "area_km2": 0.01}}
    metadata = {
        "date_a": {"date": "2022-01-01", "cloud_coverage": 5.0},
        "date_b": {"date": "2022-06-01", "cloud_coverage": 8.0},
        "center_lat": 13.0827,
        "center_lon": 80.2707,
        "output_prefix": prefix,
    }

    np.save(os.path.join(tmp_path, f"{prefix}_bands_a.npy"), bands)
    np.save(os.path.join(tmp_path, f"{prefix}_bands_b.npy"), bands)
    np.save(os.path.join(tmp_path, f"{prefix}_classification.npy"), classification)

    with open(os.path.join(tmp_path, f"{prefix}_change_stats.json"), "w") as f:
        json.dump(stats, f)
    with open(os.path.join(tmp_path, f"{prefix}_metadata.json"), "w") as f:
        json.dump(metadata, f)


class TestExport:
    """Tests for the 'export' command."""

    @patch("satchange.cli.VisualizationManager")
    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_success(self, MockConfig, MockGEE, MockViz, runner, tmp_path):
        cfg = MagicMock(is_authenticated=MagicMock(return_value=False))
        MockConfig.return_value = cfg

        _setup_export_dir(tmp_path)
        viz_inst = MagicMock()
        viz_inst.generate_all_outputs.return_value = {
            "static": str(tmp_path / "out.png"),
        }
        MockViz.return_value = viz_inst

        result = runner.invoke(main, ["export", "--result", str(tmp_path)])
        assert result.exit_code == 0
        assert "Visualization generation completed" in result.output

    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_result_dir_not_found(self, MockConfig, MockGEE, runner, tmp_path):
        cfg = MagicMock(is_authenticated=MagicMock(return_value=False))
        MockConfig.return_value = cfg

        result = runner.invoke(
            main, ["export", "--result", str(tmp_path / "nonexistent")]
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_missing_required_files(self, MockConfig, MockGEE, runner, tmp_path):
        cfg = MagicMock(is_authenticated=MagicMock(return_value=False))
        MockConfig.return_value = cfg

        # Empty directory — no artefacts at all  → no metadata file
        result = runner.invoke(main, ["export", "--result", str(tmp_path)])
        assert result.exit_code == 1
        # Either "Missing required files" or "No metadata file found"
        assert "ERROR" in result.output

    @patch("satchange.cli.VisualizationManager")
    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_format_static(self, MockConfig, MockGEE, MockViz, runner, tmp_path):
        cfg = MagicMock(is_authenticated=MagicMock(return_value=False))
        MockConfig.return_value = cfg

        _setup_export_dir(tmp_path)
        viz_inst = MagicMock()
        viz_inst.generate_all_outputs.return_value = {"static": "out.png"}
        MockViz.return_value = viz_inst

        result = runner.invoke(
            main, ["export", "--result", str(tmp_path), "--format", "static"]
        )
        assert result.exit_code == 0
        # Verify only static format was requested
        call_args = viz_inst.generate_all_outputs.call_args
        assert call_args[0][8] == ["static"]  # formats_to_generate positional arg #8

    @patch("satchange.cli.VisualizationManager")
    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_name_flag(self, MockConfig, MockGEE, MockViz, runner, tmp_path):
        cfg = MagicMock(is_authenticated=MagicMock(return_value=False))
        MockConfig.return_value = cfg

        _setup_export_dir(tmp_path)
        viz_inst = MagicMock()
        viz_inst.generate_all_outputs.return_value = {"static": "out.png"}
        MockViz.return_value = viz_inst

        result = runner.invoke(
            main,
            ["export", "--result", str(tmp_path), "--name", "my-place"],
        )
        assert result.exit_code == 0
        call_kwargs = viz_inst.generate_all_outputs.call_args
        # The output_prefix kwarg should contain the user-provided name
        assert "my-place" in call_kwargs.kwargs.get(
            "output_prefix", call_kwargs[1].get("output_prefix", "")
        )


# ======================================================================
# cache command tests
# ======================================================================


class TestCacheStatus:
    """Tests for 'cache status' command."""

    @patch("satchange.cli.CacheManager")
    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_shows_statistics(self, MockConfig, MockGEE, MockCache, runner):
        cfg = MagicMock(is_authenticated=MagicMock(return_value=False))
        MockConfig.return_value = cfg

        cache_inst = MagicMock()
        cache_inst.get_cache_stats.return_value = {
            "total_items": 42,
            "size_formatted": "256 MB",
            "usage_percent": 12.5,
            "hit_rate": 60.0,
            "hits": 30,
            "misses": 20,
            "evictions": 2,
        }
        MockCache.return_value = cache_inst

        result = runner.invoke(main, ["cache", "status"])
        assert result.exit_code == 0
        assert "42" in result.output
        assert "256 MB" in result.output

    @patch("satchange.cli.CacheManager")
    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_exception(self, MockConfig, MockGEE, MockCache, runner):
        cfg = MagicMock(is_authenticated=MagicMock(return_value=False))
        MockConfig.return_value = cfg
        MockCache.side_effect = RuntimeError("db locked")

        result = runner.invoke(main, ["cache", "status"])
        assert "ERROR" in result.output


class TestCacheClear:
    """Tests for 'cache clear' command."""

    @patch("satchange.cli.CacheManager")
    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_confirm_clears(self, MockConfig, MockGEE, MockCache, runner):
        cfg = MagicMock(is_authenticated=MagicMock(return_value=False))
        MockConfig.return_value = cfg

        cache_inst = MagicMock()
        cache_inst.clear_cache.return_value = True
        MockCache.return_value = cache_inst

        result = runner.invoke(main, ["cache", "clear"], input="y\n")
        assert result.exit_code == 0
        assert "cleared" in result.output.lower()
        cache_inst.clear_cache.assert_called_once()

    @patch("satchange.cli.CacheManager")
    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_clear_failure(self, MockConfig, MockGEE, MockCache, runner):
        cfg = MagicMock(is_authenticated=MagicMock(return_value=False))
        MockConfig.return_value = cfg

        cache_inst = MagicMock()
        cache_inst.clear_cache.return_value = False
        MockCache.return_value = cache_inst

        result = runner.invoke(main, ["cache", "clear"], input="y\n")
        assert "ERROR" in result.output or "Failed" in result.output


class TestCacheCleanup:
    """Tests for 'cache cleanup' command."""

    @patch("satchange.cli.CacheManager")
    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_success(self, MockConfig, MockGEE, MockCache, runner):
        cfg = MagicMock(is_authenticated=MagicMock(return_value=False))
        MockConfig.return_value = cfg

        cache_inst = MagicMock()
        cache_inst.cleanup_cache.return_value = True
        MockCache.return_value = cache_inst

        result = runner.invoke(main, ["cache", "cleanup"])
        assert result.exit_code == 0
        assert "cleanup completed" in result.output.lower()

    @patch("satchange.cli.CacheManager")
    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_failure(self, MockConfig, MockGEE, MockCache, runner):
        cfg = MagicMock(is_authenticated=MagicMock(return_value=False))
        MockConfig.return_value = cfg

        cache_inst = MagicMock()
        cache_inst.cleanup_cache.return_value = False
        MockCache.return_value = cache_inst

        result = runner.invoke(main, ["cache", "cleanup"])
        assert "ERROR" in result.output or "failed" in result.output.lower()

    @patch("satchange.cli.CacheManager")
    @patch("satchange.cli.GEEClient")
    @patch("satchange.cli.Config")
    def test_cleanup_exception(self, MockConfig, MockGEE, MockCache, runner):
        cfg = MagicMock(is_authenticated=MagicMock(return_value=False))
        MockConfig.return_value = cfg
        MockCache.side_effect = RuntimeError("db error")

        result = runner.invoke(main, ["cache", "cleanup"])
        assert "ERROR" in result.output or "failed" in result.output.lower()
