"""
Tests for the GEE client module.

Comprehensive test suite for satchange/gee_client.py covering the exception
hierarchy, authentication, imagery queries, metadata parsing, image-pair
selection, download with retry logic, local cloud analysis, alternative-date
search, temporal compositing, and the graduated cloud-fallback strategy.
"""

import sys
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ee is already mocked via conftest.py sys.modules
import ee

from satchange.gee_client import (
    GEEClient,
    GEEError,
    AuthenticationError,
    QuotaExceededError,
    RateLimitError,
    NoImageryError,
    DownloadError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    """Config mock that prevents auto-authenticate in __init__."""
    config = MagicMock()
    config.is_authenticated.return_value = False
    config.get.side_effect = lambda key, default=None: {
        "service_account_key": "/path/to/key.json",
        "project_id": "test-project",
        "service_account_email": "test@test.iam.gserviceaccount.com",
        "cloud_threshold": 20,
    }.get(key, default)
    return config


@pytest.fixture
def client(mock_config):
    """Create a GEEClient without triggering real authentication."""
    return GEEClient(mock_config)


def _make_feature(image_id, cloud_pct, epoch_ms, sensing_time=None):
    """Helper: build a GEE-style feature dict."""
    props = {
        "CLOUDY_PIXEL_PERCENTAGE": cloud_pct,
        "system:time_start": epoch_ms,
        "MGRS_TILE": "T43PGP",
        "PRODUCT_ID": "S2A_MSIL2A",
    }
    if sensing_time is not None:
        props["SENSING_TIME"] = sensing_time
    return {"id": image_id, "properties": props}


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    """Every custom exception inherits from GEEError."""

    def test_gee_error_is_base(self):
        assert issubclass(GEEError, Exception)

    def test_authentication_error_inherits(self):
        assert issubclass(AuthenticationError, GEEError)

    def test_quota_exceeded_error_inherits(self):
        assert issubclass(QuotaExceededError, GEEError)

    def test_rate_limit_error_inherits(self):
        assert issubclass(RateLimitError, GEEError)

    def test_no_imagery_error_inherits(self):
        assert issubclass(NoImageryError, GEEError)

    def test_download_error_inherits(self):
        assert issubclass(DownloadError, GEEError)

    def test_exception_message_propagation(self):
        msg = "something went wrong"
        err = AuthenticationError(msg)
        assert str(err) == msg

    def test_gee_error_catch_catches_subclass(self):
        with pytest.raises(GEEError):
            raise RateLimitError("rate limited")


# ---------------------------------------------------------------------------
# __init__ / _initialize
# ---------------------------------------------------------------------------


class TestGEEClientInit:
    def test_initialize_calls_authenticate_when_already_authed(self):
        config = MagicMock()
        config.is_authenticated.return_value = True
        # authenticate() will be called; mock the chain it uses
        ee.ServiceAccountCredentials.return_value = MagicMock()
        ee.ImageCollection.return_value.limit.return_value.size.return_value.getInfo.return_value = 1
        config.get.side_effect = lambda key, default=None: {
            "service_account_key": "/k.json",
            "project_id": "p",
            "service_account_email": "e@e.iam.gserviceaccount.com",
            "cloud_threshold": 20,
        }.get(key, default)
        c = GEEClient(config)
        assert c._authenticated is True

    def test_initialize_skips_authenticate_when_not_authed(self, mock_config):
        mock_config.is_authenticated.return_value = False
        c = GEEClient(mock_config)
        assert c._authenticated is False


# ---------------------------------------------------------------------------
# authenticate()
# ---------------------------------------------------------------------------


class TestAuthenticate:
    def test_successful_authentication(self, client):
        ee.ServiceAccountCredentials.return_value = MagicMock()
        ee.ImageCollection.return_value.limit.return_value.size.return_value.getInfo.return_value = 1
        result = client.authenticate()
        assert result is True
        assert client._authenticated is True
        ee.Initialize.assert_called()

    def test_authentication_failure_raises(self, client):
        ee.ServiceAccountCredentials.side_effect = Exception("bad creds")
        with pytest.raises(AuthenticationError, match="Authentication failed"):
            client.authenticate()
        assert client._authenticated is False
        # Reset
        ee.ServiceAccountCredentials.side_effect = None

    def test_missing_service_account_key_raises(self, mock_config):
        mock_config.get.side_effect = lambda key, default=None: {
            "service_account_key": None,
            "project_id": "p",
            "service_account_email": "e",
            "cloud_threshold": 20,
        }.get(key, default)
        c = GEEClient(mock_config)
        with pytest.raises(AuthenticationError, match="required"):
            c.authenticate()

    def test_missing_project_id_raises(self, mock_config):
        mock_config.get.side_effect = lambda key, default=None: {
            "service_account_key": "/k.json",
            "project_id": None,
            "service_account_email": "e",
            "cloud_threshold": 20,
        }.get(key, default)
        c = GEEClient(mock_config)
        with pytest.raises(AuthenticationError, match="required"):
            c.authenticate()


# ---------------------------------------------------------------------------
# create_bbox()
# ---------------------------------------------------------------------------


class TestCreateBbox:
    def _fresh_geopy_mock(self):
        """Inject a fresh geopy mock into sys.modules so `from geopy.distance import distance` works."""
        mock_distance_mod = MagicMock()
        mock_geopy = MagicMock()
        mock_geopy.distance = mock_distance_mod
        sys.modules["geopy"] = mock_geopy
        sys.modules["geopy.distance"] = mock_distance_mod
        return mock_distance_mod

    def test_create_bbox_calls_polygon(self, client):
        """create_bbox should invoke ee.Geometry.Polygon with 4-corner coords."""
        mock_distance_mod = self._fresh_geopy_mock()

        mock_dest = MagicMock()
        mock_dest.latitude = 13.1
        mock_dest.longitude = 80.3
        mock_distance_mod.distance.return_value.destination.return_value = mock_dest

        client.create_bbox(13.0, 80.0, 100, 10)
        ee.Geometry.Polygon.assert_called()

    def test_create_bbox_corner_calculations(self, client):
        """Verify geopy.distance is called with correct bearings."""
        mock_distance_mod = self._fresh_geopy_mock()

        mock_dest = MagicMock()
        mock_dest.latitude = 0.0
        mock_dest.longitude = 0.0

        bearings_used = []
        inst = MagicMock()

        def capture_destination(point, bearing):
            bearings_used.append(bearing)
            return mock_dest

        inst.destination.side_effect = capture_destination
        mock_distance_mod.distance.return_value = inst

        client.create_bbox(0.0, 0.0, 100, 10)

        # N(0), S(180), E(90), W(270)
        assert 0 in bearings_used
        assert 180 in bearings_used
        assert 90 in bearings_used
        assert 270 in bearings_used


# ---------------------------------------------------------------------------
# query_imagery()
# ---------------------------------------------------------------------------


class TestQueryImagery:
    def test_filters_collection(self, client):
        mock_bbox = MagicMock()
        start = datetime(2022, 1, 1)
        end = datetime(2022, 12, 31)

        # Build chained mock
        mock_coll = MagicMock()
        ee.ImageCollection.return_value = mock_coll
        mock_coll.filterBounds.return_value = mock_coll
        mock_coll.filterDate.return_value = mock_coll
        mock_coll.filter.return_value = mock_coll
        mock_coll.size.return_value.getInfo.return_value = 5

        result = client.query_imagery(mock_bbox, start, end, cloud_threshold=20)

        mock_coll.filterBounds.assert_called_once_with(mock_bbox)
        mock_coll.filterDate.assert_called_once_with(start, end)
        mock_coll.filter.assert_called_once()
        assert result is mock_coll


# ---------------------------------------------------------------------------
# get_scenes_metadata()
# ---------------------------------------------------------------------------


class TestGetScenesMetadata:
    def test_parsing_time_start(self, client):
        epoch_ms = 1640995200000  # 2022-01-01 00:00 UTC
        coll = MagicMock()
        coll.getInfo.return_value = {"features": [_make_feature("img1", 5.0, epoch_ms)]}
        scenes = client.get_scenes_metadata(coll)
        assert len(scenes) == 1
        assert scenes[0]["date"] == datetime.fromtimestamp(epoch_ms / 1000).strftime(
            "%Y-%m-%d"
        )

    def test_fallback_to_sensing_time(self, client):
        coll = MagicMock()
        coll.getInfo.return_value = {
            "features": [
                {
                    "id": "img2",
                    "properties": {
                        "CLOUDY_PIXEL_PERCENTAGE": 10,
                        "SENSING_TIME": "2023-06-15",
                        "MGRS_TILE": "T",
                        "PRODUCT_ID": "P",
                    },
                }
            ]
        }
        scenes = client.get_scenes_metadata(coll)
        assert scenes[0]["date"] == "2023-06-15"

    def test_empty_collection(self, client):
        coll = MagicMock()
        coll.getInfo.return_value = {"features": []}
        assert client.get_scenes_metadata(coll) == []

    def test_no_features_key(self, client):
        coll = MagicMock()
        coll.getInfo.return_value = {}
        assert client.get_scenes_metadata(coll) == []

    def test_invalid_overflow_timestamp(self, client):
        coll = MagicMock()
        coll.getInfo.return_value = {
            "features": [
                _make_feature("img3", 2.0, 99999999999999999)  # overflow
            ]
        }
        scenes = client.get_scenes_metadata(coll)
        assert scenes[0]["date"] == "Unknown"


# ---------------------------------------------------------------------------
# select_best_image_pair()
# ---------------------------------------------------------------------------


class TestSelectBestImagePair:
    def _build_collection(self, features):
        coll = MagicMock()
        coll.sort.return_value.getInfo.return_value = {"features": features}
        return coll

    def test_selects_clearest_pair(self, client):
        """Selects the lowest-cloud pair within respective windows."""
        start = datetime(2022, 1, 1)
        end = datetime(2023, 1, 1)

        # Date A candidate near start, Date B candidate near end with 6-month gap
        date_a_epoch = int(datetime(2022, 1, 10).timestamp() * 1000)
        date_b_epoch = int(datetime(2022, 12, 20).timestamp() * 1000)
        features = [
            _make_feature("imgA", 5.0, date_a_epoch),
            _make_feature("imgB", 8.0, date_b_epoch),
        ]
        coll = self._build_collection(features)
        pair = client.select_best_image_pair(coll, start, end, cloud_threshold=20)
        assert len(pair) == 2

    def test_raises_on_empty_scenes(self, client):
        coll = self._build_collection([])
        with pytest.raises(ValueError, match="No scenes found"):
            client.select_best_image_pair(
                coll, datetime(2022, 1, 1), datetime(2023, 1, 1)
            )

    def test_raises_when_all_exceed_threshold(self, client):
        start = datetime(2022, 1, 1)
        end = datetime(2023, 1, 1)
        date_a_epoch = int(datetime(2022, 1, 10).timestamp() * 1000)
        features = [_make_feature("imgX", 90.0, date_a_epoch)]
        coll = self._build_collection(features)
        with pytest.raises(ValueError, match="exceed"):
            client.select_best_image_pair(coll, start, end, cloud_threshold=5)

    def test_expanded_window_search(self, client):
        """If no scenes in 30-day window, expand to 90 days."""
        start = datetime(2022, 1, 1)
        end = datetime(2023, 1, 1)
        # Date A at day 60 (outside 30 but within 90)
        date_a_epoch = int(datetime(2022, 3, 1).timestamp() * 1000)
        date_b_epoch = int(datetime(2022, 12, 20).timestamp() * 1000)
        features = [
            _make_feature("imgA60", 5.0, date_a_epoch),
            _make_feature("imgB60", 5.0, date_b_epoch),
        ]
        coll = self._build_collection(features)
        pair = client.select_best_image_pair(coll, start, end, cloud_threshold=20)
        assert len(pair) == 2

    def test_minimum_6_month_gap(self, client):
        """Date B must be at least 180 days after Date A."""
        start = datetime(2022, 1, 1)
        end = datetime(2022, 12, 31)
        date_a_epoch = int(datetime(2022, 1, 10).timestamp() * 1000)
        # Date B only 2 months later — should be rejected, no valid B
        date_b_too_close = int(datetime(2022, 3, 10).timestamp() * 1000)
        features = [
            _make_feature("imgA", 5.0, date_a_epoch),
            _make_feature("imgTooClose", 5.0, date_b_too_close),
        ]
        coll = self._build_collection(features)
        with pytest.raises(ValueError, match="6-month gap"):
            client.select_best_image_pair(coll, start, end, cloud_threshold=20)


# ---------------------------------------------------------------------------
# download_image()
# ---------------------------------------------------------------------------


class TestDownloadImage:
    """Test download_image with retry logic, HTTP error handling.

    Note: download_image does ``import requests`` and ``import rasterio``
    locally inside the function body, so we patch on the real modules
    (``requests.get``) rather than on ``satchange.gee_client.requests``.
    ``rasterio`` is already mocked via conftest sys.modules.
    """

    def _setup_download_mocks(self):
        """Common setup: mock the image.select().getThumbURL() chain."""
        mock_image = MagicMock()
        mock_image.select.return_value.getThumbURL.return_value = (
            "https://example.com/img.tif"
        )
        mock_bbox = MagicMock()
        return mock_image, mock_bbox

    def _setup_rasterio_mock(self):
        """Configure the already-mocked rasterio.open as a context manager."""
        mock_rasterio = sys.modules["rasterio"]
        mock_dataset = MagicMock()
        mock_dataset.read.side_effect = lambda idx: MagicMock()
        mock_dataset.transform = MagicMock()
        mock_dataset.crs = "EPSG:32643"
        mock_dataset.bounds = MagicMock()
        mock_dataset.width = 100
        mock_dataset.height = 100
        mock_rasterio.open = MagicMock()
        mock_rasterio.open.return_value.__enter__ = MagicMock(return_value=mock_dataset)
        mock_rasterio.open.return_value.__exit__ = MagicMock(return_value=False)
        return mock_rasterio

    @patch("time.sleep")
    @patch("requests.get")
    def test_successful_download(self, mock_get, _sleep, client):
        mock_image, mock_bbox = self._setup_download_mocks()
        self._setup_rasterio_mock()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"GEOTIFFDATA"
        mock_get.return_value = mock_response

        band_arrays, metadata = client.download_image(mock_image, mock_bbox)
        assert "B4" in band_arrays
        assert "width" in metadata

    @patch("time.sleep")
    @patch("requests.get")
    def test_retry_on_timeout(self, mock_get, mock_sleep, client):
        mock_image, mock_bbox = self._setup_download_mocks()
        self._setup_rasterio_mock()

        import requests as req_mod

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"OK"

        mock_get.side_effect = [
            req_mod.exceptions.Timeout("t1"),
            req_mod.exceptions.Timeout("t2"),
            mock_response,
        ]

        band_arrays, metadata = client.download_image(mock_image, mock_bbox)
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("time.sleep")
    @patch("requests.get")
    def test_429_raises_rate_limit_error(self, mock_get, mock_sleep, client):
        mock_image, mock_bbox = self._setup_download_mocks()

        import requests as req_mod

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = req_mod.exceptions.HTTPError(
            response=mock_response
        )
        mock_get.return_value = mock_response

        with pytest.raises(RateLimitError, match="rate limit"):
            client.download_image(mock_image, mock_bbox)

    @patch("time.sleep")
    @patch("requests.get")
    def test_403_raises_quota_exceeded(self, mock_get, mock_sleep, client):
        mock_image, mock_bbox = self._setup_download_mocks()

        import requests as req_mod

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = req_mod.exceptions.HTTPError(
            response=mock_response
        )
        mock_get.return_value = mock_response

        with pytest.raises(QuotaExceededError, match="quota"):
            client.download_image(mock_image, mock_bbox)

    @patch("time.sleep")
    @patch("requests.get")
    def test_other_http_error_raises_download_error(self, mock_get, mock_sleep, client):
        mock_image, mock_bbox = self._setup_download_mocks()

        import requests as req_mod

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = req_mod.exceptions.HTTPError(
            response=mock_response
        )
        mock_get.return_value = mock_response

        with pytest.raises(DownloadError, match="HTTP 500"):
            client.download_image(mock_image, mock_bbox)


# ---------------------------------------------------------------------------
# check_local_cloud()
# ---------------------------------------------------------------------------


class TestCheckLocalCloud:
    def _setup_ee_chain(self, count, cloud_fraction=0.1, scene_cloud=12.0):
        """Wire up the ee mock chain for check_local_cloud."""
        # ee.Geometry.Point / buffer / bounds
        ee.Geometry.Point.return_value = MagicMock()
        ee.Geometry.Point.return_value.buffer.return_value.bounds.return_value = (
            MagicMock()
        )

        # Collection chain
        mock_coll = MagicMock()
        ee.ImageCollection.return_value = mock_coll
        mock_coll.filterBounds.return_value = mock_coll
        mock_coll.filterDate.return_value = mock_coll
        mock_coll.size.return_value.getInfo.return_value = count

        if count > 0:
            # ee.Image(collection.first())
            mock_image = MagicMock()
            mock_coll.first.return_value = mock_image
            ee.Image.return_value = mock_image

            mock_image.getInfo.return_value = {
                "id": "COPERNICUS/S2_SR_HARMONIZED/20220101",
                "properties": {"CLOUDY_PIXEL_PERCENTAGE": scene_cloud},
            }

            # SCL band → cloud_mask → reduceRegion chain
            mock_scl = MagicMock()
            mock_image.select.return_value = mock_scl

            # eq/Or chain returns a mock that has reduceRegion
            mock_cloud_mask = MagicMock()
            mock_scl.eq.return_value = mock_cloud_mask
            mock_cloud_mask.Or.return_value = mock_cloud_mask

            mock_stats = MagicMock()
            mock_cloud_mask.reduceRegion.return_value = mock_stats
            mock_scl_result = MagicMock()
            mock_stats.get.return_value = mock_scl_result
            mock_scl_result.getInfo.return_value = cloud_fraction

            ee.Reducer.mean.return_value = MagicMock()

        return mock_coll

    def test_returns_proper_dict_structure(self, client):
        self._setup_ee_chain(count=1, cloud_fraction=0.05, scene_cloud=8.0)
        result = client.check_local_cloud("2022-01-01", (13.0, 80.0))
        expected_keys = {
            "date",
            "image_id",
            "scene_cloud_pct",
            "local_cloud_pct",
            "is_good",
            "found",
        }
        assert set(result.keys()) == expected_keys
        assert result["found"] is True

    def test_no_image_found(self, client):
        self._setup_ee_chain(count=0)
        result = client.check_local_cloud("2022-01-01", (13.0, 80.0))
        assert result["found"] is False
        assert result["image_id"] is None

    def test_cloud_mask_includes_scl_values(self, client):
        """SCL values 3,8,9,10,11 should all be included in cloud mask."""
        self._setup_ee_chain(count=1, cloud_fraction=0.3)

        # Capture calls to scl.eq()
        mock_coll = ee.ImageCollection.return_value
        mock_image = ee.Image.return_value
        mock_scl = mock_image.select.return_value

        client.check_local_cloud("2022-01-01", (13.0, 80.0))

        # The source calls scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
        eq_calls = [call.args[0] for call in mock_scl.eq.call_args_list]
        for val in [3, 8, 9, 10, 11]:
            assert val in eq_calls, f"SCL value {val} not included in cloud mask"

    def test_local_cloud_pct_calculation(self, client):
        """local_cloud_pct should be fraction * 100."""
        self._setup_ee_chain(count=1, cloud_fraction=0.25, scene_cloud=15.0)
        result = client.check_local_cloud("2022-01-01", (13.0, 80.0))
        assert result["local_cloud_pct"] == 25.0
        assert result["is_good"] is False  # 25 > 15 threshold


# ---------------------------------------------------------------------------
# find_alternative_dates()
# ---------------------------------------------------------------------------


class TestFindAlternativeDates:
    def _setup_alternatives(self, features, search_count=5):
        """Wire ee mocks for find_alternative_dates."""
        ee.Geometry.Point.return_value = MagicMock()
        ee.Geometry.Point.return_value.buffer.return_value.bounds.return_value = (
            MagicMock()
        )

        mock_coll = MagicMock()
        ee.ImageCollection.return_value = mock_coll
        mock_coll.filterBounds.return_value = mock_coll
        mock_coll.filterDate.return_value = mock_coll
        mock_coll.filter.return_value = mock_coll
        mock_coll.sort.return_value = mock_coll
        mock_coll.limit.return_value = mock_coll
        mock_coll.size.return_value.getInfo.return_value = len(features)
        mock_coll.getInfo.return_value = {"features": features}

        # Per-image SCL compute chain
        mock_image = MagicMock()
        ee.Image.return_value = mock_image
        mock_scl = MagicMock()
        mock_image.select.return_value = mock_scl
        mock_cloud_mask = MagicMock()
        mock_scl.eq.return_value = mock_cloud_mask
        mock_cloud_mask.Or.return_value = mock_cloud_mask

        mock_stats = MagicMock()
        mock_cloud_mask.reduceRegion.return_value = mock_stats
        # Default: all alternatives have 5% local cloud
        mock_scl_result = MagicMock()
        mock_stats.get.return_value = mock_scl_result
        mock_scl_result.getInfo.return_value = 0.05

        ee.Reducer.mean.return_value = MagicMock()
        ee.Filter.lt.return_value = MagicMock()

        return mock_coll

    def test_incremental_search_windows(self, client):
        epoch = int(datetime(2022, 1, 15).timestamp() * 1000)
        features = [_make_feature("alt1", 10.0, epoch)]
        self._setup_alternatives(features)

        result = client.find_alternative_dates("2022-01-01", (13.0, 80.0))
        assert result["threshold_met"] is True
        assert result["search_window"] in [
            "±2 weeks",
            "±1 month",
            "±2 months",
            "±3 months",
        ]

    def test_alternatives_sorted_by_local_cloud(self, client):
        epoch1 = int(datetime(2022, 1, 10).timestamp() * 1000)
        epoch2 = int(datetime(2022, 1, 20).timestamp() * 1000)
        features = [
            _make_feature("alt1", 10.0, epoch1),
            _make_feature("alt2", 5.0, epoch2),
        ]
        self._setup_alternatives(features)

        result = client.find_alternative_dates("2022-01-01", (13.0, 80.0))
        alts = result["alternatives"]
        assert len(alts) >= 2
        # Already sorted by local_cloud_pct (all same from mock)
        for i in range(len(alts) - 1):
            assert alts[i]["local_cloud_pct"] <= alts[i + 1]["local_cloud_pct"]

    def test_marks_best_as_recommended(self, client):
        epoch = int(datetime(2022, 1, 10).timestamp() * 1000)
        features = [_make_feature("alt1", 10.0, epoch)]
        self._setup_alternatives(features)

        result = client.find_alternative_dates("2022-01-01", (13.0, 80.0))
        assert result["alternatives"][0]["is_recommended"] is True

    def test_threshold_met_true_when_good(self, client):
        epoch = int(datetime(2022, 1, 10).timestamp() * 1000)
        features = [_make_feature("alt1", 10.0, epoch)]
        self._setup_alternatives(features)

        result = client.find_alternative_dates("2022-01-01", (13.0, 80.0))
        assert result["threshold_met"] is True

    def test_threshold_met_false_when_no_good(self, client):
        """When all alternatives exceed threshold, threshold_met=False."""
        epoch = int(datetime(2022, 1, 10).timestamp() * 1000)
        features = [_make_feature("alt1", 40.0, epoch)]

        ee.Geometry.Point.return_value = MagicMock()
        ee.Geometry.Point.return_value.buffer.return_value.bounds.return_value = (
            MagicMock()
        )

        mock_coll = MagicMock()
        ee.ImageCollection.return_value = mock_coll
        mock_coll.filterBounds.return_value = mock_coll
        mock_coll.filterDate.return_value = mock_coll
        mock_coll.filter.return_value = mock_coll
        mock_coll.sort.return_value = mock_coll
        mock_coll.limit.return_value = mock_coll
        mock_coll.size.return_value.getInfo.return_value = len(features)
        mock_coll.getInfo.return_value = {"features": features}

        # SCL: return very high cloud (90%)
        mock_image = MagicMock()
        ee.Image.return_value = mock_image
        mock_scl = MagicMock()
        mock_image.select.return_value = mock_scl
        mock_cloud_mask = MagicMock()
        mock_scl.eq.return_value = mock_cloud_mask
        mock_cloud_mask.Or.return_value = mock_cloud_mask

        mock_stats = MagicMock()
        mock_cloud_mask.reduceRegion.return_value = mock_stats
        mock_scl_result = MagicMock()
        mock_stats.get.return_value = mock_scl_result
        mock_scl_result.getInfo.return_value = 0.90  # 90% cloud

        ee.Reducer.mean.return_value = MagicMock()
        ee.Filter.lt.return_value = MagicMock()

        result = client.find_alternative_dates(
            "2022-01-01", (13.0, 80.0), cloud_threshold=15.0
        )
        assert result["threshold_met"] is False

    def test_returns_top_10(self, client):
        """Even if 15 features are available, alternatives capped at 10."""
        features = []
        for i in range(15):
            epoch = int(datetime(2022, 1, 1 + i).timestamp() * 1000)
            features.append(_make_feature(f"alt{i}", 5.0, epoch))
        self._setup_alternatives(features)

        result = client.find_alternative_dates("2022-01-01", (13.0, 80.0))
        assert len(result["alternatives"]) <= 10


# ---------------------------------------------------------------------------
# create_temporal_composite()
# ---------------------------------------------------------------------------


class TestCreateTemporalComposite:
    def _setup_composite(self, count):
        mock_coll = MagicMock()
        ee.ImageCollection.return_value = mock_coll
        mock_coll.filterBounds.return_value = mock_coll
        mock_coll.filterDate.return_value = mock_coll
        mock_coll.sort.return_value = mock_coll
        mock_coll.limit.return_value = mock_coll
        mock_coll.size.return_value.getInfo.return_value = count
        mock_coll.median.return_value = MagicMock(name="median_composite")
        return mock_coll

    @patch.object(GEEClient, "create_bbox", return_value=MagicMock())
    def test_returns_median_when_enough_scenes(self, _mock_bbox, client):
        mock_coll = self._setup_composite(count=3)
        result = client.create_temporal_composite((13.0, 80.0), "2022-06-01")
        assert result is not None
        mock_coll.median.assert_called_once()

    @patch.object(GEEClient, "create_bbox", return_value=MagicMock())
    def test_returns_none_when_fewer_than_2(self, _mock_bbox, client):
        self._setup_composite(count=1)
        result = client.create_temporal_composite((13.0, 80.0), "2022-06-01")
        assert result is None

    @patch.object(GEEClient, "create_bbox", return_value=MagicMock())
    def test_uses_correct_window_and_limit(self, mock_bbox, client):
        mock_coll = self._setup_composite(count=4)
        client.create_temporal_composite(
            (13.0, 80.0), "2022-06-01", window_days=45, max_scenes=3
        )
        mock_coll.limit.assert_called_with(3)


# ---------------------------------------------------------------------------
# handle_cloudy_scenes()
# ---------------------------------------------------------------------------


class TestHandleCloudyScenes:
    def _make_cloud_result(self, found=True, local_cloud_pct=50.0, image_id="img123"):
        return {
            "date": "2022-01-01",
            "image_id": image_id,
            "scene_cloud_pct": 30.0,
            "local_cloud_pct": local_cloud_pct,
            "is_good": local_cloud_pct < 15.0,
            "found": found,
        }

    def _make_alt_result(self, threshold_met=True, alternatives=None):
        if alternatives is None:
            alternatives = [
                {
                    "date": "2022-01-05",
                    "image_id": "alt_img",
                    "scene_cloud_pct": 10.0,
                    "local_cloud_pct": 8.0,
                    "is_recommended": True,
                }
            ]
        return {
            "target_date": "2022-01-01",
            "search_window": "±2 weeks",
            "threshold_met": threshold_met,
            "alternatives": alternatives,
        }

    @patch.object(GEEClient, "check_local_cloud")
    def test_strategy1_graduated_threshold_succeeds(self, mock_check, client):
        """Strategy 1 succeeds when local cloud is under a relaxed threshold."""
        # First call with threshold 20 → too cloudy; second call threshold 40 → passes
        mock_check.side_effect = [
            self._make_cloud_result(found=True, local_cloud_pct=35.0),
            self._make_cloud_result(found=True, local_cloud_pct=35.0),
        ]
        result = client.handle_cloudy_scenes(
            "2022-01-01", (13.0, 80.0), max_cloud_threshold=20
        )
        assert result["found"] is True
        assert result["strategy_used"] == "increased_threshold"

    @patch.object(GEEClient, "create_temporal_composite", return_value=None)
    @patch.object(GEEClient, "find_alternative_dates")
    @patch.object(GEEClient, "check_local_cloud")
    def test_strategy2_expanded_window_succeeds(
        self, mock_check, mock_alt, mock_comp, client
    ):
        """Strategy 2 succeeds when Strategy 1 fails."""
        # Strategy 1: all checks fail (cloud > 60)
        mock_check.return_value = self._make_cloud_result(
            found=True, local_cloud_pct=75.0
        )
        # Strategy 2: find_alternative_dates succeeds
        mock_alt.return_value = self._make_alt_result(threshold_met=True)

        result = client.handle_cloudy_scenes("2022-01-01", (13.0, 80.0))
        assert result["found"] is True
        assert result["strategy_used"] == "expanded_window"

    @patch.object(GEEClient, "create_temporal_composite")
    @patch.object(GEEClient, "find_alternative_dates")
    @patch.object(GEEClient, "check_local_cloud")
    def test_strategy3_temporal_composite_succeeds(
        self, mock_check, mock_alt, mock_comp, client
    ):
        """Strategy 3 succeeds when Strategies 1 & 2 fail."""
        mock_check.return_value = self._make_cloud_result(
            found=True, local_cloud_pct=75.0
        )
        mock_alt.return_value = self._make_alt_result(
            threshold_met=False,
            alternatives=[
                {
                    "date": "2022-01-05",
                    "image_id": "x",
                    "scene_cloud_pct": 80.0,
                    "local_cloud_pct": 80.0,
                    "is_recommended": True,
                }
            ],
        )
        mock_comp.return_value = MagicMock(name="composite_image")

        result = client.handle_cloudy_scenes("2022-01-01", (13.0, 80.0))
        assert result["found"] is True
        assert result["strategy_used"] == "temporal_composite"
        assert result["date"] == "composite"

    @patch.object(GEEClient, "create_temporal_composite", return_value=None)
    @patch.object(GEEClient, "find_alternative_dates")
    @patch.object(GEEClient, "check_local_cloud")
    def test_all_strategies_exhausted(self, mock_check, mock_alt, mock_comp, client):
        """Returns found=False when no strategy works."""
        mock_check.return_value = self._make_cloud_result(
            found=True, local_cloud_pct=95.0
        )
        mock_alt.return_value = self._make_alt_result(
            threshold_met=False,
            alternatives=[
                {
                    "date": "2022-01-05",
                    "image_id": "bad",
                    "scene_cloud_pct": 90.0,
                    "local_cloud_pct": 90.0,
                    "is_recommended": True,
                }
            ],
        )

        result = client.handle_cloudy_scenes("2022-01-01", (13.0, 80.0))
        assert result["found"] is False
        assert result["strategy_used"] == "none"

    @patch.object(GEEClient, "create_temporal_composite", return_value=None)
    @patch.object(GEEClient, "find_alternative_dates")
    @patch.object(GEEClient, "check_local_cloud")
    def test_no_image_for_date_skips_to_strategy2(
        self, mock_check, mock_alt, mock_comp, client
    ):
        """When check_local_cloud returns found=False, Strategy 1 breaks early."""
        mock_check.return_value = self._make_cloud_result(
            found=False, local_cloud_pct=0
        )
        mock_alt.return_value = self._make_alt_result(threshold_met=True)

        result = client.handle_cloudy_scenes("2022-01-01", (13.0, 80.0))
        # Should have skipped remaining thresholds and gone to Strategy 2
        assert result["found"] is True
        assert result["strategy_used"] == "expanded_window"
        # check_local_cloud called only once (broke out after found=False)
        assert mock_check.call_count == 1

    @patch.object(GEEClient, "check_local_cloud")
    def test_return_dict_structure(self, mock_check, client):
        """Verify all required keys in the return dict."""
        mock_check.return_value = self._make_cloud_result(
            found=True, local_cloud_pct=10.0
        )
        result = client.handle_cloudy_scenes("2022-01-01", (13.0, 80.0))
        required_keys = {
            "found",
            "strategy_used",
            "image",
            "image_id",
            "date",
            "local_cloud_pct",
            "details",
        }
        assert required_keys.issubset(set(result.keys()))


# ---------------------------------------------------------------------------
# is_authenticated()
# ---------------------------------------------------------------------------


class TestIsAuthenticated:
    def test_initially_false(self, client):
        assert client.is_authenticated() is False

    def test_true_after_successful_auth(self, client):
        ee.ServiceAccountCredentials.return_value = MagicMock()
        ee.ImageCollection.return_value.limit.return_value.size.return_value.getInfo.return_value = 1
        client.authenticate()
        assert client.is_authenticated() is True
