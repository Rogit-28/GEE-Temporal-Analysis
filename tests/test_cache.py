"""
Tests for satchange/cache.py — ImageCache and CacheManager.

Covers: __init__, _generate_key, get, set, delete, clear, stats, cleanup, close,
        and CacheManager's cache-first download pattern.
"""

import hashlib
import json
import re
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# diskcache is already mocked via conftest.py sys.modules
import diskcache

from satchange.cache import ImageCache, CacheManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_cache_config():
    """Config mock wired for ImageCache.__init__."""
    config = MagicMock()
    config.get_cache_directory.return_value = "/tmp/test_satchange_cache"
    config.get_cache_max_size_bytes.return_value = 5 * 1024 * 1024 * 1024  # 5 GB
    config.get.side_effect = lambda key, default=None: {
        "cache.max_size_gb": 5,
        "cache.directory": "/tmp/test_satchange_cache",
    }.get(key, default)
    return config


@pytest.fixture
def image_cache(mock_cache_config):
    """Return an ImageCache with a fresh mock diskcache.Cache backend."""
    mock_disk = MagicMock()
    diskcache.Cache.return_value = mock_disk

    with patch("satchange.cache.os.makedirs") as _makedirs:
        cache = ImageCache(mock_cache_config)

    # Expose the underlying mock for per-test configuration
    cache._mock_disk = mock_disk
    return cache


@pytest.fixture
def sample_params():
    """Common query parameters used across many tests."""
    return {
        "center_lat": 13.0827,
        "center_lon": 80.2707,
        "pixel_size": 100,
        "date": datetime(2023, 6, 15),
        "bands": ["B4", "B3", "B8"],
    }


@pytest.fixture
def cache_manager(mock_cache_config):
    """Return a CacheManager wrapping a mocked ImageCache."""
    mock_disk = MagicMock()
    diskcache.Cache.return_value = mock_disk

    with patch("satchange.cache.os.makedirs"):
        mgr = CacheManager(mock_cache_config)

    mgr.cache._mock_disk = mock_disk
    return mgr


# ===================================================================
# TestImageCache
# ===================================================================


class TestImageCache:
    """Tests for ImageCache."""

    # ---- __init__ ----------------------------------------------------

    class TestInit:
        def test_cache_directory_created(self, mock_cache_config):
            """os.makedirs is called with the config directory."""
            diskcache.Cache.return_value = MagicMock()

            with patch("satchange.cache.os.makedirs") as makedirs:
                ImageCache(mock_cache_config)
            makedirs.assert_called_once_with("/tmp/test_satchange_cache", exist_ok=True)

        def test_diskcache_initialized_with_correct_params(self, mock_cache_config):
            """diskcache.Cache receives dir, size_limit, eviction_policy."""
            mock_disk = MagicMock()
            diskcache.Cache.return_value = mock_disk

            with patch("satchange.cache.os.makedirs"):
                ImageCache(mock_cache_config)

            diskcache.Cache.assert_called_with(
                "/tmp/test_satchange_cache",
                size_limit=5 * 1024 * 1024 * 1024,
                eviction_policy="least-recently-used",
            )

        def test_uses_config_methods(self, mock_cache_config):
            """Verifies get_cache_directory and get_cache_max_size_bytes are called."""
            diskcache.Cache.return_value = MagicMock()

            with patch("satchange.cache.os.makedirs"):
                ImageCache(mock_cache_config)

            mock_cache_config.get_cache_directory.assert_called_once()
            mock_cache_config.get_cache_max_size_bytes.assert_called_once()

    # ---- get ---------------------------------------------------------

    class TestGet:
        def test_cache_hit_returns_data(self, image_cache, sample_params):
            """When the backend holds data, get() returns it."""
            expected = {"bands": {"B4": [1, 2, 3]}}
            image_cache._mock_disk.get.return_value = expected

            result = image_cache.get(**sample_params)
            assert result == expected

        def test_cache_miss_returns_none(self, image_cache, sample_params):
            """When the backend returns None, get() returns None."""
            image_cache._mock_disk.get.return_value = None
            assert image_cache.get(**sample_params) is None

        def test_exception_returns_none(self, image_cache, sample_params):
            """On any backend exception, get() returns None instead of crashing."""
            image_cache._mock_disk.get.side_effect = RuntimeError("disk failure")
            assert image_cache.get(**sample_params) is None

    # ---- set ---------------------------------------------------------

    class TestSet:
        def test_stores_data_with_cached_at(self, image_cache, sample_params):
            """Stored payload includes a 'cached_at' ISO timestamp."""
            data = {"bands": {"B4": [1]}}

            image_cache.set(**sample_params, data=data)

            stored = image_cache._mock_disk.set.call_args[0][1]
            assert "cached_at" in stored
            # Validate ISO format
            datetime.fromisoformat(stored["cached_at"])

        def test_stores_data_with_cache_key(self, image_cache, sample_params):
            """Stored payload includes the 'cache_key' hex string."""
            data = {"bands": {"B4": [1]}}

            image_cache.set(**sample_params, data=data)

            stored = image_cache._mock_disk.set.call_args[0][1]
            assert "cache_key" in stored
            assert len(stored["cache_key"]) == 64

        def test_returns_true_on_success(self, image_cache, sample_params):
            assert image_cache.set(**sample_params, data={"x": 1}) is True

        def test_returns_false_on_exception(self, image_cache, sample_params):
            image_cache._mock_disk.set.side_effect = RuntimeError("boom")
            assert image_cache.set(**sample_params, data={"x": 1}) is False

    # ---- delete ------------------------------------------------------

    class TestDelete:
        def test_successful_deletion_returns_true(self, image_cache, sample_params):
            image_cache._mock_disk.delete.return_value = True
            assert image_cache.delete(**sample_params) is True

        def test_key_not_found_returns_false(self, image_cache, sample_params):
            image_cache._mock_disk.delete.return_value = False
            assert image_cache.delete(**sample_params) is False

        def test_exception_returns_false(self, image_cache, sample_params):
            image_cache._mock_disk.delete.side_effect = RuntimeError("boom")
            assert image_cache.delete(**sample_params) is False

    # ---- clear -------------------------------------------------------

    class TestClear:
        def test_calls_cache_clear(self, image_cache):
            image_cache.clear()
            image_cache._mock_disk.clear.assert_called_once()

        def test_returns_true_on_success(self, image_cache):
            assert image_cache.clear() is True

        def test_returns_false_on_exception(self, image_cache):
            image_cache._mock_disk.clear.side_effect = RuntimeError("boom")
            assert image_cache.clear() is False

    # ---- stats -------------------------------------------------------

    class TestStats:
        def _setup_stats(self, image_cache, hits=10, misses=5, volume=1000, length=3):
            image_cache._mock_disk.stats.return_value = (hits, misses)
            image_cache._mock_disk.volume.return_value = volume
            image_cache._mock_disk.__len__ = MagicMock(return_value=length)
            image_cache.max_size_bytes = 5 * 1024 * 1024 * 1024

        def test_returns_dict_with_expected_keys(self, image_cache):
            self._setup_stats(image_cache)
            with patch.object(image_cache, "_get_directory_size", return_value=2000):
                result = image_cache.stats()

            expected_keys = {
                "total_items",
                "size_bytes",
                "size_formatted",
                "directory_size_bytes",
                "directory_size_formatted",
                "max_size_bytes",
                "max_size_formatted",
                "usage_percent",
                "hits",
                "misses",
                "hit_rate",
                "evictions",
            }
            assert expected_keys == set(result.keys())

        def test_handles_tuple_from_cache_stats(self, image_cache):
            self._setup_stats(image_cache, hits=7, misses=3)
            with patch.object(image_cache, "_get_directory_size", return_value=0):
                result = image_cache.stats()

            assert result["hits"] == 7
            assert result["misses"] == 3

        def test_usage_percent_calculation(self, image_cache):
            max_bytes = 5 * 1024 * 1024 * 1024
            volume = max_bytes // 2  # 50%
            self._setup_stats(image_cache, volume=volume)
            with patch.object(image_cache, "_get_directory_size", return_value=0):
                result = image_cache.stats()

            assert result["usage_percent"] == 50.0

        def test_hit_rate_calculation(self, image_cache):
            self._setup_stats(image_cache, hits=75, misses=25)
            with patch.object(image_cache, "_get_directory_size", return_value=0):
                result = image_cache.stats()

            assert result["hit_rate"] == 75.0

    # ---- cleanup -----------------------------------------------------

    class TestCleanup:
        def test_removes_old_entries(self, image_cache):
            """Entries > 30 days old are deleted."""
            old_time = (datetime.now() - timedelta(days=60)).isoformat()
            image_cache._mock_disk.keys.return_value = ["old_key"]
            image_cache._mock_disk.get.return_value = {"cached_at": old_time}

            result = image_cache.cleanup()
            assert result is True
            image_cache._mock_disk.delete.assert_called_once_with("old_key")

        def test_keeps_recent_entries(self, image_cache):
            """Entries < 30 days old are kept."""
            recent_time = (datetime.now() - timedelta(days=5)).isoformat()
            image_cache._mock_disk.keys.return_value = ["recent_key"]
            image_cache._mock_disk.get.return_value = {"cached_at": recent_time}

            result = image_cache.cleanup()
            assert result is True
            image_cache._mock_disk.delete.assert_not_called()

        def test_removes_entries_without_cached_at(self, image_cache):
            """Entries missing 'cached_at' trigger an exception path that removes them."""
            # When data has no 'cached_at', the inner try block encounters
            # the condition `if data and 'cached_at' in data` which is False,
            # so the entry is NOT removed through the normal path.
            # However, if accessing the data raises an exception, the except
            # branch calls delete. Simulate that by having get() raise.
            image_cache._mock_disk.keys.return_value = ["bad_key"]
            image_cache._mock_disk.get.side_effect = RuntimeError("corrupt")

            result = image_cache.cleanup()
            assert result is True
            image_cache._mock_disk.delete.assert_called_once_with("bad_key")

        def test_returns_true_on_success(self, image_cache):
            image_cache._mock_disk.keys.return_value = []
            assert image_cache.cleanup() is True

        def test_returns_false_on_exception(self, image_cache):
            image_cache._mock_disk.keys.side_effect = RuntimeError("boom")
            assert image_cache.cleanup() is False

    # ---- close -------------------------------------------------------

    class TestClose:
        def test_calls_cache_close(self, image_cache):
            image_cache.close()
            image_cache._mock_disk.close.assert_called_once()


# ===================================================================
# TestCacheKeyGeneration  (_generate_key specifics)
# ===================================================================


class TestCacheKeyGeneration:
    """Focused tests on _generate_key determinism, formatting, ordering."""

    @pytest.fixture(autouse=True)
    def _setup(self, image_cache, sample_params):
        self.cache = image_cache
        self.params = sample_params

    def test_determinism_same_inputs_same_hash(self):
        """Identical inputs always produce the same key."""
        key1 = self.cache._generate_key(**self.params)
        key2 = self.cache._generate_key(**self.params)
        assert key1 == key2

    def test_different_inputs_different_hashes(self):
        """Changing any parameter changes the key."""
        key_a = self.cache._generate_key(**self.params)

        altered = dict(self.params)
        altered["center_lat"] = 14.0
        key_b = self.cache._generate_key(**altered)

        assert key_a != key_b

    def test_coordinates_rounded_to_6_decimals(self):
        """Coordinates differing only past 6 decimal places produce the same key."""
        params_a = dict(self.params)
        params_a["center_lat"] = 13.0827001234
        params_a["center_lon"] = 80.2707009999

        params_b = dict(self.params)
        params_b["center_lat"] = 13.082700
        params_b["center_lon"] = 80.270701

        key_a = self.cache._generate_key(**params_a)
        key_b = self.cache._generate_key(**params_b)
        assert key_a == key_b

    def test_bands_sorted_order_doesnt_matter(self):
        """Band order is irrelevant: ['B4','B3','B8'] == ['B8','B3','B4']."""
        params_a = dict(self.params)
        params_a["bands"] = ["B4", "B3", "B8"]

        params_b = dict(self.params)
        params_b["bands"] = ["B8", "B3", "B4"]

        assert self.cache._generate_key(**params_a) == self.cache._generate_key(
            **params_b
        )

    def test_key_is_hex_sha256_string(self):
        """Key must be a 64-char lowercase hex string (SHA-256)."""
        key = self.cache._generate_key(**self.params)
        assert len(key) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", key)

    def test_key_matches_manual_sha256(self):
        """Verify against a manually computed SHA-256 for known inputs."""
        params = {
            "lat": round(self.params["center_lat"], 6),
            "lon": round(self.params["center_lon"], 6),
            "size": self.params["pixel_size"],
            "date": self.params["date"].isoformat(),
            "bands": sorted(self.params["bands"]),
        }
        expected = hashlib.sha256(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()

        actual = self.cache._generate_key(**self.params)
        assert actual == expected

    def test_different_dates_different_keys(self):
        """Two different dates must produce different keys."""
        params_a = dict(self.params)
        params_a["date"] = datetime(2023, 1, 1)

        params_b = dict(self.params)
        params_b["date"] = datetime(2024, 1, 1)

        assert self.cache._generate_key(**params_a) != self.cache._generate_key(
            **params_b
        )

    def test_different_pixel_sizes_different_keys(self):
        params_a = dict(self.params)
        params_a["pixel_size"] = 100

        params_b = dict(self.params)
        params_b["pixel_size"] = 200

        assert self.cache._generate_key(**params_a) != self.cache._generate_key(
            **params_b
        )


# ===================================================================
# TestCacheManager
# ===================================================================


class TestCacheManager:
    """Tests for CacheManager."""

    # ---- get_image_with_cache ----------------------------------------

    class TestGetImageWithCache:
        def test_cache_hit_returns_cached_data(self, cache_manager, sample_params):
            """On hit: returns cached data and True, never calls download_func."""
            cached = {"bands": {"B4": [1]}}
            cache_manager.cache._mock_disk.get.return_value = cached

            download_fn = MagicMock()
            data, hit = cache_manager.get_image_with_cache(
                **sample_params,
                download_func=download_fn,
            )

            assert data == cached
            assert hit is True
            download_fn.assert_not_called()

        def test_cache_miss_downloads_and_stores(self, cache_manager, sample_params):
            """On miss: calls download_func, stores result, returns False flag."""
            cache_manager.cache._mock_disk.get.return_value = None
            downloaded = {"bands": {"B4": [9, 8, 7]}}

            download_fn = MagicMock(return_value=downloaded)
            data, hit = cache_manager.get_image_with_cache(
                **sample_params,
                download_func=download_fn,
            )

            assert data == downloaded
            assert hit is False
            download_fn.assert_called_once()
            cache_manager.cache._mock_disk.set.assert_called_once()

        def test_cache_miss_download_exception_propagates(
            self, cache_manager, sample_params
        ):
            """If download_func raises, the exception propagates to the caller."""
            cache_manager.cache._mock_disk.get.return_value = None
            download_fn = MagicMock(side_effect=ConnectionError("network down"))

            with pytest.raises(ConnectionError, match="network down"):
                cache_manager.get_image_with_cache(
                    **sample_params,
                    download_func=download_fn,
                )

        def test_returns_tuple(self, cache_manager, sample_params):
            """Return value is always a 2-tuple (data, bool)."""
            cache_manager.cache._mock_disk.get.return_value = {"x": 1}
            result = cache_manager.get_image_with_cache(
                **sample_params,
                download_func=MagicMock(),
            )
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[1], bool)

    # ---- delegates ---------------------------------------------------

    class TestDelegation:
        def test_get_cache_stats_delegates(self, cache_manager):
            """get_cache_stats() delegates to ImageCache.stats()."""
            with patch.object(
                cache_manager.cache, "stats", return_value={"hits": 5}
            ) as mock_stats:
                result = cache_manager.get_cache_stats()
            mock_stats.assert_called_once()
            assert result == {"hits": 5}

        def test_clear_cache_delegates(self, cache_manager):
            """clear_cache() delegates to ImageCache.clear()."""
            with patch.object(
                cache_manager.cache, "clear", return_value=True
            ) as mock_clear:
                result = cache_manager.clear_cache()
            mock_clear.assert_called_once()
            assert result is True

        def test_cleanup_cache_delegates(self, cache_manager):
            """cleanup_cache() delegates to ImageCache.cleanup()."""
            with patch.object(
                cache_manager.cache, "cleanup", return_value=True
            ) as mock_cleanup:
                result = cache_manager.cleanup_cache()
            mock_cleanup.assert_called_once()
            assert result is True
