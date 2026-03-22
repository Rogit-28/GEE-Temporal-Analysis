"""
Configuration management for SatChange.

This module handles loading, saving, and validating configuration settings
for the SatChange CLI tool.
"""

import copy
import os
import shutil
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Exception raised for configuration-related errors."""

    pass


class Config:
    """Configuration management for SatChange."""

    DEFAULT_CONFIG = {
        "service_account_key": None,
        "project_id": None,
        "service_account_email": None,
        "cloud_threshold": 20,
        "pixel_size": 100,
        "cache": {
            "max_size_gb": 5,
            "eviction_policy": "least-recently-used",
            "directory": "~/.satchange/cache",
        },
        "analysis": {
            "change_threshold": 0.2,
            "emboss_intensity": 1.0,
            "min_temporal_gap_days": 180,
        },
    }

    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager.

        Args:
            config_file: Path to configuration file. If None, uses default location.
        """
        self.config_file = config_file or self._get_default_config_path()
        self._config = copy.deepcopy(self.DEFAULT_CONFIG)

        # Load existing configuration if it exists
        if os.path.exists(self.config_file):
            self.load()

    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        home_dir = Path.home()
        config_dir = home_dir / ".satchange"
        config_dir.mkdir(exist_ok=True)
        return str(config_dir / "config.yaml")

    def _get_managed_key_path(self, source_key_path: str, project_id: str) -> str:
        """Get managed key path under ~/.satchange/keys for persistent auth."""
        home_dir = Path.home()
        keys_dir = home_dir / ".satchange" / "keys"
        keys_dir.mkdir(parents=True, exist_ok=True)
        source_name = Path(source_key_path).stem
        managed_name = f"{project_id}_{source_name}.json"
        return str(keys_dir / managed_name)

    def load(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config:
                    self._config.update(loaded_config)

            logger.debug(f"Configuration loaded from {self.config_file}")

        except FileNotFoundError:
            logger.debug(f"Configuration file not found: {self.config_file}")
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in config file: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to load configuration: {e}")

    def save(self) -> None:
        """Save configuration to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

            with open(self.config_file, "w", encoding="utf-8") as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)

            # Set secure permissions for config file
            import platform

            if platform.system() != "Windows":
                try:
                    os.chmod(self.config_file, 0o600)
                except OSError:
                    logger.warning("Could not set secure permissions on config file")
            else:
                # On Windows, use icacls to restrict permissions
                try:
                    import subprocess

                    username = os.environ.get("USERNAME", os.environ.get("USER", ""))
                    if username:
                        subprocess.run(
                            [
                                "icacls",
                                self.config_file,
                                "/inheritance:r",
                                "/grant:r",
                                f"{username}:(R,W)",
                            ],
                            capture_output=True,
                            check=False,
                        )
                except Exception:
                    logger.warning(
                        "Could not set secure permissions on config file (Windows)"
                    )

            logger.debug(f"Configuration saved to {self.config_file}")

        except Exception as e:
            raise ConfigError(f"Failed to save configuration: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default.

        Args:
            key: Configuration key (supports dot notation, e.g., 'cache.max_size_gb')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value: Any = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        config: Dict[str, Any] = self._config

        # Navigate to parent of the final key
        for k in keys[:-1]:
            nested = config.get(k)
            if not isinstance(nested, dict):
                nested = {}
                config[k] = nested
            config = nested

        # Set the final value
        config[keys[-1]] = value

    def initialize_auth(self, service_account_key: str, project_id: str) -> None:
        """Initialize authentication configuration.

        Args:
            service_account_key: Path to service account JSON key file
            project_id: Google Cloud project ID
        """
        # Validate service account key file exists
        if not os.path.exists(service_account_key):
            raise ConfigError(
                f"Service account key file not found: {service_account_key}"
            )

        # Try to load and validate the service account key
        try:
            import json

            with open(service_account_key, "r") as f:
                key_data = json.load(f)

            # Extract email from key file
            service_account_email = key_data.get("client_email")
            if not service_account_email:
                raise ConfigError("Invalid service account key: missing client_email")

            # Persist a managed copy outside repo to survive working tree cleanup.
            managed_key_path = self._get_managed_key_path(
                service_account_key, project_id
            )
            shutil.copy2(service_account_key, managed_key_path)

            # Restrict managed key permissions.
            import platform

            if platform.system() != "Windows":
                try:
                    os.chmod(managed_key_path, 0o600)
                except OSError:
                    logger.warning(
                        "Could not set secure permissions on managed key file"
                    )
            else:
                try:
                    import subprocess

                    username = os.environ.get("USERNAME", os.environ.get("USER", ""))
                    if username:
                        subprocess.run(
                            [
                                "icacls",
                                managed_key_path,
                                "/inheritance:r",
                                "/grant:r",
                                f"{username}:(R,W)",
                            ],
                            capture_output=True,
                            check=False,
                        )
                except Exception:
                    logger.warning(
                        "Could not set secure permissions on managed key file (Windows)"
                    )

            # Set authentication configuration
            self.set("service_account_key", managed_key_path)
            self.set("project_id", project_id)
            self.set("service_account_email", service_account_email)

            # Save configuration
            self.save()

        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid service account key file: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to initialize authentication: {e}")

    def is_authenticated(self) -> bool:
        """Check if authentication is configured.

        Returns:
            True if authentication is configured, False otherwise
        """
        return (
            self.get("service_account_key") is not None
            and self.get("project_id") is not None
            and self.get("service_account_email") is not None
        )

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            ConfigError: If configuration is invalid
        """
        # Check required fields
        required_fields = ["service_account_key", "project_id", "service_account_email"]
        for field in required_fields:
            if not self.get(field):
                raise ConfigError(f"Required field missing: {field}")

        # Validate service account key file exists
        key_file = self.get("service_account_key")
        if key_file and not os.path.exists(key_file):
            raise ConfigError(f"Service account key file not found: {key_file}")

        # Validate numeric ranges
        cloud_threshold = self.get("cloud_threshold", 20)
        if cloud_threshold is None or not isinstance(cloud_threshold, (int, float)):
            raise ConfigError("cloud_threshold must be between 0 and 100")
        if not 0 <= cloud_threshold <= 100:
            raise ConfigError("cloud_threshold must be between 0 and 100")

        pixel_size = self.get("pixel_size", 100)
        if pixel_size is None or not isinstance(pixel_size, (int, float)):
            raise ConfigError("pixel_size must be between 10 and 1000")
        if not 10 <= pixel_size <= 1000:
            raise ConfigError("pixel_size must be between 10 and 1000")

        change_threshold = self.get("analysis.change_threshold", 0.2)
        if change_threshold is None or not isinstance(change_threshold, (int, float)):
            raise ConfigError("change_threshold must be between 0.1 and 1.0")
        if not 0.1 <= change_threshold <= 1.0:
            raise ConfigError("change_threshold must be between 0.1 and 1.0")

        emboss_intensity = self.get("analysis.emboss_intensity", 1.0)
        if emboss_intensity is None or not isinstance(emboss_intensity, (int, float)):
            raise ConfigError("emboss_intensity must be between 0.0 and 2.0")
        if not 0.0 <= emboss_intensity <= 2.0:
            raise ConfigError("emboss_intensity must be between 0.0 and 2.0")

        min_temporal_gap_days = self.get("analysis.min_temporal_gap_days", 180)
        if min_temporal_gap_days is None or not isinstance(
            min_temporal_gap_days, (int, float)
        ):
            raise ConfigError("min_temporal_gap_days must be between 30 and 365")
        if not 30 <= min_temporal_gap_days <= 365:
            raise ConfigError("min_temporal_gap_days must be between 30 and 365")

    def get_cache_directory(self) -> str:
        """Get cache directory path.

        Returns:
            Absolute path to cache directory
        """
        cache_dir = self.get("cache.directory")
        return os.path.expanduser(cache_dir)

    def get_cache_max_size_bytes(self) -> int:
        """Get cache maximum size in bytes.

        Returns:
            Maximum cache size in bytes
        """
        max_size_gb = self.get("cache.max_size_gb", 5)
        return int(max_size_gb * 1024 * 1024 * 1024)

    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return copy.deepcopy(self._config)
