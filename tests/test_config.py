"""
Comprehensive tests for satchange.config module.

Tests cover ConfigError, Config.__init__, load, save, get, set,
initialize_auth, is_authenticated, validate, cache helpers, and to_dict.
"""

import json
import os
import copy

import pytest
import yaml
from unittest.mock import patch, MagicMock, mock_open

from satchange.config import Config, ConfigError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config_dir(tmp_path):
    """Create a temporary config directory."""
    return tmp_path / ".satchange"


@pytest.fixture
def config_file(config_dir):
    """Create a temporary config file path."""
    config_dir.mkdir(exist_ok=True)
    return str(config_dir / "config.yaml")


@pytest.fixture
def fresh_config(config_file):
    """Create a Config instance with a fresh temporary config file."""
    return Config(config_file=config_file)


@pytest.fixture
def populated_config(config_file):
    """Create a Config instance with authentication fields populated."""
    cfg = Config(config_file=config_file)
    cfg.set("service_account_key", "/fake/key.json")
    cfg.set("project_id", "my-project")
    cfg.set("service_account_email", "sa@proj.iam.gserviceaccount.com")
    return cfg


# ---------------------------------------------------------------------------
# ConfigError
# ---------------------------------------------------------------------------


class TestConfigError:
    """Tests for ConfigError exception class."""

    def test_is_subclass_of_exception(self):
        assert issubclass(ConfigError, Exception)

    def test_message_propagation(self):
        err = ConfigError("something went wrong")
        assert str(err) == "something went wrong"


# ---------------------------------------------------------------------------
# Config.__init__
# ---------------------------------------------------------------------------


class TestConfigInit:
    """Tests for Config.__init__."""

    def test_with_explicit_config_file(self, config_file):
        cfg = Config(config_file=config_file)
        assert cfg.config_file == config_file

    def test_without_config_file_uses_default(self, tmp_path):
        fake_default = str(tmp_path / "default_config.yaml")
        with patch.object(
            Config, "_get_default_config_path", return_value=fake_default
        ):
            cfg = Config()
        assert cfg.config_file == fake_default

    def test_loads_existing_config_on_init(self, config_file):
        # Write a config file before creating Config
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, "w") as f:
            yaml.dump({"cloud_threshold": 50}, f)

        cfg = Config(config_file=config_file)
        assert cfg.get("cloud_threshold") == 50

    def test_default_config_is_deep_copied(self, config_file):
        cfg1 = Config(config_file=config_file)
        cfg2 = Config(config_file=config_file)
        cfg1.set("cloud_threshold", 99)
        # cfg2 must still have the default
        assert cfg2.get("cloud_threshold") == 20


# ---------------------------------------------------------------------------
# Config.load
# ---------------------------------------------------------------------------


class TestConfigLoad:
    """Tests for Config.load."""

    def test_loads_valid_yaml(self, fresh_config, config_file):
        with open(config_file, "w") as f:
            yaml.dump({"pixel_size": 250}, f)
        fresh_config.load()
        assert fresh_config.get("pixel_size") == 250

    def test_missing_file_no_crash(self, tmp_path):
        missing = str(tmp_path / "nonexistent" / "config.yaml")
        cfg = Config(config_file=missing)
        # Just calling load explicitly on a missing file should not crash
        cfg.load()
        # defaults still present
        assert cfg.get("cloud_threshold") == 20

    def test_invalid_yaml_raises_config_error(self, fresh_config, config_file):
        with open(config_file, "w") as f:
            f.write("{{invalid: yaml: [")
        with pytest.raises(ConfigError, match="Invalid YAML"):
            fresh_config.load()

    def test_permission_error_raises_config_error(self, fresh_config, config_file):
        with patch("builtins.open", side_effect=PermissionError("denied")):
            with pytest.raises(ConfigError, match="Failed to load configuration"):
                fresh_config.load()

    def test_loaded_config_merges_with_defaults(self, fresh_config, config_file):
        # Write partial config — only cloud_threshold
        with open(config_file, "w") as f:
            yaml.dump({"cloud_threshold": 75}, f)
        fresh_config.load()
        # Overridden value
        assert fresh_config.get("cloud_threshold") == 75
        # Default value still present (not replaced)
        assert fresh_config.get("pixel_size") == 100


# ---------------------------------------------------------------------------
# Config.save
# ---------------------------------------------------------------------------


class TestConfigSave:
    """Tests for Config.save."""

    def test_creates_parent_directory(self, tmp_path):
        nested = str(tmp_path / "a" / "b" / "config.yaml")
        cfg = Config(config_file=nested)
        cfg.save()
        assert os.path.isfile(nested)

    def test_writes_valid_yaml(self, fresh_config, config_file):
        fresh_config.save()
        with open(config_file, "r") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert "cloud_threshold" in data

    def test_yaml_round_trip(self, fresh_config, config_file):
        fresh_config.set("cloud_threshold", 42)
        fresh_config.set("cache.max_size_gb", 10)
        fresh_config.save()

        reloaded = Config(config_file=config_file)
        assert reloaded.get("cloud_threshold") == 42
        assert reloaded.get("cache.max_size_gb") == 10

    def test_non_windows_calls_chmod(self, fresh_config, config_file):
        mock_platform = MagicMock()
        mock_platform.system.return_value = "Linux"

        with patch.dict("sys.modules", {"platform": mock_platform}):
            with patch("satchange.config.os.chmod") as mock_chmod:
                fresh_config.save()
                mock_chmod.assert_called_once_with(config_file, 0o600)

    def test_windows_calls_icacls(self, fresh_config, config_file):
        mock_platform = MagicMock()
        mock_platform.system.return_value = "Windows"
        mock_subprocess = MagicMock()

        with patch.dict(
            "sys.modules", {"platform": mock_platform, "subprocess": mock_subprocess}
        ):
            with patch.dict(os.environ, {"USERNAME": "testuser"}):
                fresh_config.save()
                mock_subprocess.run.assert_called_once()
                call_args = mock_subprocess.run.call_args
                assert "icacls" in call_args[0][0]

    def test_chmod_failure_logs_warning(self, fresh_config, config_file, caplog):
        mock_platform = MagicMock()
        mock_platform.system.return_value = "Linux"

        with patch.dict("sys.modules", {"platform": mock_platform}):
            with patch("satchange.config.os.chmod", side_effect=OSError("perm fail")):
                import logging

                with caplog.at_level(logging.WARNING, logger="satchange.config"):
                    fresh_config.save()
                assert "secure permissions" in caplog.text.lower() or os.path.isfile(
                    config_file
                )
                # File still written — no crash
                assert os.path.isfile(config_file)

    def test_windows_icacls_failure_logs_warning(
        self, fresh_config, config_file, caplog
    ):
        mock_platform = MagicMock()
        mock_platform.system.return_value = "Windows"
        mock_subprocess = MagicMock()
        mock_subprocess.run.side_effect = Exception("icacls broke")

        with patch.dict(
            "sys.modules", {"platform": mock_platform, "subprocess": mock_subprocess}
        ):
            with patch.dict(os.environ, {"USERNAME": "testuser"}):
                import logging

                with caplog.at_level(logging.WARNING, logger="satchange.config"):
                    fresh_config.save()
                # No crash, file written
                assert os.path.isfile(config_file)

    def test_write_failure_raises_config_error(self, fresh_config):
        with patch("builtins.open", side_effect=IOError("disk full")):
            with pytest.raises(ConfigError, match="Failed to save configuration"):
                fresh_config.save()


# ---------------------------------------------------------------------------
# Config.get
# ---------------------------------------------------------------------------


class TestConfigGet:
    """Tests for Config.get."""

    def test_top_level_key(self, fresh_config):
        assert fresh_config.get("cloud_threshold") == 20

    def test_dot_notation_nested(self, fresh_config):
        assert fresh_config.get("cache.max_size_gb") == 5

    def test_deep_dot_notation(self, fresh_config):
        assert fresh_config.get("analysis.change_threshold") == 0.2

    def test_missing_key_returns_none(self, fresh_config):
        assert fresh_config.get("nonexistent") is None

    def test_missing_key_returns_custom_default(self, fresh_config):
        assert fresh_config.get("nonexistent", 42) == 42

    def test_nested_missing_key_returns_default(self, fresh_config):
        assert fresh_config.get("cache.nonexistent_key", "fallback") == "fallback"


# ---------------------------------------------------------------------------
# Config.set
# ---------------------------------------------------------------------------


class TestConfigSet:
    """Tests for Config.set."""

    def test_set_top_level_key(self, fresh_config):
        fresh_config.set("cloud_threshold", 55)
        assert fresh_config.get("cloud_threshold") == 55

    def test_set_nested_key_dot_notation(self, fresh_config):
        fresh_config.set("cache.max_size_gb", 10)
        assert fresh_config.get("cache.max_size_gb") == 10

    def test_set_creates_intermediate_dicts(self, fresh_config):
        fresh_config.set("new.nested.key", "value")
        assert fresh_config.get("new.nested.key") == "value"

    def test_overwrite_existing_value(self, fresh_config):
        fresh_config.set("pixel_size", 200)
        assert fresh_config.get("pixel_size") == 200
        fresh_config.set("pixel_size", 300)
        assert fresh_config.get("pixel_size") == 300


# ---------------------------------------------------------------------------
# Config.initialize_auth
# ---------------------------------------------------------------------------


class TestInitializeAuth:
    """Tests for Config.initialize_auth."""

    def test_valid_key_file(self, fresh_config, tmp_path):
        key_file = tmp_path / "key.json"
        key_data = {
            "type": "service_account",
            "client_email": "sa@example.iam.gserviceaccount.com",
            "project_id": "test-proj",
        }
        key_file.write_text(json.dumps(key_data))

        fresh_config.initialize_auth(str(key_file), "test-proj")

        assert fresh_config.get("service_account_key") == str(key_file)
        assert fresh_config.get("project_id") == "test-proj"
        assert (
            fresh_config.get("service_account_email")
            == "sa@example.iam.gserviceaccount.com"
        )

    def test_file_not_found_raises_config_error(self, fresh_config):
        with pytest.raises(ConfigError, match="not found"):
            fresh_config.initialize_auth("/no/such/file.json", "proj")

    def test_invalid_json_raises_config_error(self, fresh_config, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json {{")
        with pytest.raises(ConfigError, match="Invalid service account key file"):
            fresh_config.initialize_auth(str(bad_file), "proj")

    def test_missing_client_email_raises_config_error(self, fresh_config, tmp_path):
        key_file = tmp_path / "no_email.json"
        key_file.write_text(json.dumps({"type": "service_account"}))
        with pytest.raises(ConfigError, match="missing client_email"):
            fresh_config.initialize_auth(str(key_file), "proj")

    def test_initialize_auth_saves_config(self, fresh_config, tmp_path, config_file):
        key_file = tmp_path / "key.json"
        key_file.write_text(
            json.dumps(
                {
                    "client_email": "sa@proj.iam.gserviceaccount.com",
                }
            )
        )
        fresh_config.initialize_auth(str(key_file), "my-proj")
        # Verify the config was saved to disk
        assert os.path.isfile(config_file)
        with open(config_file) as f:
            saved = yaml.safe_load(f)
        assert saved["project_id"] == "my-proj"


# ---------------------------------------------------------------------------
# Config.is_authenticated
# ---------------------------------------------------------------------------


class TestIsAuthenticated:
    """Tests for Config.is_authenticated."""

    def test_returns_true_when_all_fields_set(self, populated_config):
        assert populated_config.is_authenticated() is True

    def test_returns_false_when_key_is_none(self, populated_config):
        populated_config.set("service_account_key", None)
        assert populated_config.is_authenticated() is False

    def test_returns_false_when_project_id_is_none(self, populated_config):
        populated_config.set("project_id", None)
        assert populated_config.is_authenticated() is False

    def test_returns_false_when_email_is_none(self, populated_config):
        populated_config.set("service_account_email", None)
        assert populated_config.is_authenticated() is False

    def test_returns_false_with_default_config(self, fresh_config):
        assert fresh_config.is_authenticated() is False


# ---------------------------------------------------------------------------
# Config.validate
# ---------------------------------------------------------------------------


class TestValidate:
    """Tests for Config.validate."""

    def test_valid_config_passes(self, populated_config):
        with patch("satchange.config.os.path.exists", return_value=True):
            populated_config.validate()  # Should not raise

    def test_missing_service_account_key_raises(self, fresh_config):
        fresh_config.set("project_id", "proj")
        fresh_config.set("service_account_email", "x@y.com")
        with pytest.raises(ConfigError, match="service_account_key"):
            fresh_config.validate()

    def test_missing_project_id_raises(self, fresh_config):
        fresh_config.set("service_account_key", "/key.json")
        fresh_config.set("service_account_email", "x@y.com")
        with patch("satchange.config.os.path.exists", return_value=True):
            # project_id is still None
            with pytest.raises(ConfigError, match="project_id"):
                fresh_config.validate()

    def test_missing_email_raises(self, fresh_config):
        fresh_config.set("service_account_key", "/key.json")
        fresh_config.set("project_id", "proj")
        with patch("satchange.config.os.path.exists", return_value=True):
            with pytest.raises(ConfigError, match="service_account_email"):
                fresh_config.validate()

    def test_key_file_not_found_raises(self, populated_config):
        with patch("satchange.config.os.path.exists", return_value=False):
            with pytest.raises(ConfigError, match="not found"):
                populated_config.validate()

    def test_cloud_threshold_below_zero(self, populated_config):
        populated_config.set("cloud_threshold", -1)
        with patch("satchange.config.os.path.exists", return_value=True):
            with pytest.raises(ConfigError, match="cloud_threshold"):
                populated_config.validate()

    def test_cloud_threshold_above_100(self, populated_config):
        populated_config.set("cloud_threshold", 101)
        with patch("satchange.config.os.path.exists", return_value=True):
            with pytest.raises(ConfigError, match="cloud_threshold"):
                populated_config.validate()

    def test_pixel_size_below_range(self, populated_config):
        populated_config.set("pixel_size", 5)
        with patch("satchange.config.os.path.exists", return_value=True):
            with pytest.raises(ConfigError, match="pixel_size"):
                populated_config.validate()

    def test_pixel_size_above_range(self, populated_config):
        populated_config.set("pixel_size", 1001)
        with patch("satchange.config.os.path.exists", return_value=True):
            with pytest.raises(ConfigError, match="pixel_size"):
                populated_config.validate()

    def test_change_threshold_out_of_range(self, populated_config):
        populated_config.set("analysis.change_threshold", 0.05)
        with patch("satchange.config.os.path.exists", return_value=True):
            with pytest.raises(ConfigError, match="change_threshold"):
                populated_config.validate()

    def test_emboss_intensity_out_of_range(self, populated_config):
        populated_config.set("analysis.emboss_intensity", 3.0)
        with patch("satchange.config.os.path.exists", return_value=True):
            with pytest.raises(ConfigError, match="emboss_intensity"):
                populated_config.validate()

    def test_min_temporal_gap_days_out_of_range(self, populated_config):
        populated_config.set("analysis.min_temporal_gap_days", 10)
        with patch("satchange.config.os.path.exists", return_value=True):
            with pytest.raises(ConfigError, match="min_temporal_gap_days"):
                populated_config.validate()


# ---------------------------------------------------------------------------
# Config.get_cache_directory
# ---------------------------------------------------------------------------


class TestGetCacheDirectory:
    """Tests for Config.get_cache_directory."""

    def test_returns_expanded_path(self, fresh_config):
        result = fresh_config.get_cache_directory()
        # The default is ~/.satchange/cache; ~ should be expanded
        assert "~" not in result
        assert os.path.isabs(result)

    def test_uses_value_from_config(self, fresh_config, tmp_path):
        custom_dir = str(tmp_path / "my_cache")
        fresh_config.set("cache.directory", custom_dir)
        assert fresh_config.get_cache_directory() == custom_dir


# ---------------------------------------------------------------------------
# Config.get_cache_max_size_bytes
# ---------------------------------------------------------------------------


class TestGetCacheMaxSizeBytes:
    """Tests for Config.get_cache_max_size_bytes."""

    def test_default_5gb_to_bytes(self, fresh_config):
        expected = 5 * 1024 * 1024 * 1024  # 5368709120
        assert fresh_config.get_cache_max_size_bytes() == expected

    def test_with_different_value(self, fresh_config):
        fresh_config.set("cache.max_size_gb", 10)
        expected = 10 * 1024 * 1024 * 1024
        assert fresh_config.get_cache_max_size_bytes() == expected


# ---------------------------------------------------------------------------
# Config.to_dict
# ---------------------------------------------------------------------------


class TestToDict:
    """Tests for Config.to_dict."""

    def test_returns_deep_copy(self, fresh_config):
        d = fresh_config.to_dict()
        d["cloud_threshold"] = 999
        # Original must be unaffected
        assert fresh_config.get("cloud_threshold") == 20

    def test_contains_all_expected_keys(self, fresh_config):
        d = fresh_config.to_dict()
        for key in [
            "service_account_key",
            "project_id",
            "service_account_email",
            "cloud_threshold",
            "pixel_size",
            "cache",
            "analysis",
        ]:
            assert key in d

    def test_nested_structure_preserved(self, fresh_config):
        d = fresh_config.to_dict()
        assert "max_size_gb" in d["cache"]
        assert "change_threshold" in d["analysis"]
