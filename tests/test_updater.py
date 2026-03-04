"""Tests for the auto-updater version comparison and UpdateChecker logic."""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.core.updater import compare_versions, UpdateChecker, UpdateInfo


# -- compare_versions() tests ------------------------------------------------

class TestCompareVersions:
    """Test the compare_versions() utility function."""

    def test_equal_versions(self):
        assert compare_versions("1.0.0", "1.0.0") == 0

    def test_local_older(self):
        assert compare_versions("1.0.0", "2.0.0") == -1

    def test_local_newer(self):
        assert compare_versions("3.0.0", "2.0.0") == 1

    def test_minor_version_update(self):
        assert compare_versions("1.0.0", "1.1.0") == -1

    def test_patch_version_update(self):
        assert compare_versions("1.0.0", "1.0.1") == -1

    def test_local_newer_minor(self):
        assert compare_versions("1.2.0", "1.1.0") == 1

    def test_local_newer_patch(self):
        assert compare_versions("1.0.2", "1.0.1") == 1

    def test_strips_v_prefix(self):
        assert compare_versions("v1.0.0", "v1.0.1") == -1
        assert compare_versions("v2.0.0", "1.0.0") == 1

    def test_two_part_version(self):
        """Two-part versions should be padded to three parts."""
        assert compare_versions("1.0", "1.0.0") == 0
        assert compare_versions("1.0", "1.0.1") == -1

    def test_multi_digit_versions(self):
        assert compare_versions("2.10.0", "2.9.0") == 1
        assert compare_versions("2.2.0", "2.10.0") == -1

    def test_same_major_minor_different_patch(self):
        assert compare_versions("2.2.0", "2.2.1") == -1
        assert compare_versions("2.2.1", "2.2.0") == 1

    def test_real_cadence_version(self):
        """Test with current Cadence version."""
        assert compare_versions("2.2.0", "2.3.0") == -1
        assert compare_versions("2.2.0", "2.2.0") == 0
        assert compare_versions("2.2.0", "2.1.0") == 1


# -- UpdateChecker tests -----------------------------------------------------

class TestUpdateChecker:
    """Test the UpdateChecker class with mocked HTTP responses."""

    def _make_checker(self, current_version="2.2.0"):
        """Create an UpdateChecker with a specific current version."""
        checker = UpdateChecker.__new__(UpdateChecker)
        checker.logger = MagicMock()
        checker.current_version = current_version
        return checker

    def test_update_available(self):
        """Should return UpdateInfo when remote version is newer."""
        checker = self._make_checker("1.0.0")
        fake_response = json.dumps({
            "tag_name": "v2.0.0",
            "assets": [
                {"name": "Cadence-v2.0.0-windows.zip", "browser_download_url": "https://example.com/dl.zip"}
            ],
            "body": "## What's new\n- Feature A\n- Fix B",
        }).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = fake_response
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("src.core.updater.urlopen", return_value=mock_resp):
            result = checker.check_for_update()

        assert result is not None
        assert result.version_str == "2.0.0"
        assert result.tag_name == "v2.0.0"
        assert result.download_url == "https://example.com/dl.zip"
        assert "Feature A" in result.release_body

    def test_no_update_when_current(self):
        """Should return None when already on latest version."""
        checker = self._make_checker("2.0.0")
        fake_response = json.dumps({
            "tag_name": "v2.0.0",
            "assets": [
                {"name": "Cadence-v2.0.0-windows.zip", "browser_download_url": "https://example.com/dl.zip"}
            ],
        }).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = fake_response
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("src.core.updater.urlopen", return_value=mock_resp):
            result = checker.check_for_update()

        assert result is None

    def test_no_update_when_ahead(self):
        """Should return None when local is ahead of remote."""
        checker = self._make_checker("3.0.0")
        fake_response = json.dumps({
            "tag_name": "v2.0.0",
            "assets": [
                {"name": "Cadence-v2.0.0-windows.zip", "browser_download_url": "https://example.com/dl.zip"}
            ],
        }).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = fake_response
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("src.core.updater.urlopen", return_value=mock_resp):
            result = checker.check_for_update()

        assert result is None

    def test_no_zip_asset(self):
        """Should return None when release has no .zip asset."""
        checker = self._make_checker("1.0.0")
        fake_response = json.dumps({
            "tag_name": "v2.0.0",
            "assets": [
                {"name": "Cadence-v2.0.0.tar.gz", "browser_download_url": "https://example.com/dl.tar.gz"}
            ],
        }).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = fake_response
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("src.core.updater.urlopen", return_value=mock_resp):
            result = checker.check_for_update()

        assert result is None

    def test_empty_tag_name(self):
        """Should return None when tag_name is empty."""
        checker = self._make_checker("1.0.0")
        fake_response = json.dumps({
            "tag_name": "",
            "assets": [],
        }).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = fake_response
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("src.core.updater.urlopen", return_value=mock_resp):
            result = checker.check_for_update()

        assert result is None

    def test_network_error_returns_none(self):
        """Should return None gracefully on network failure."""
        checker = self._make_checker("1.0.0")
        from urllib.error import URLError
        with patch("src.core.updater.urlopen", side_effect=URLError("timeout")):
            result = checker.check_for_update()

        assert result is None

    def test_json_error_returns_none(self):
        """Should return None on malformed JSON."""
        checker = self._make_checker("1.0.0")

        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("src.core.updater.urlopen", return_value=mock_resp):
            result = checker.check_for_update()

        assert result is None

    def test_source_update_message(self):
        """get_source_update_message should include version and instructions."""
        info = UpdateInfo(version_str="3.0.0", tag_name="v3.0.0", download_url="")
        msg = UpdateChecker.get_source_update_message(info)
        assert "3.0.0" in msg
        assert "git pull" in msg
        assert "uv sync" in msg

    def test_release_body_preserved(self):
        """Release body (changelog) should be passed through."""
        checker = self._make_checker("1.0.0")
        changelog = "## Changes\n- Added auto-update\n- Fixed bugs"
        fake_response = json.dumps({
            "tag_name": "v2.0.0",
            "assets": [
                {"name": "Cadence.zip", "browser_download_url": "https://example.com/dl.zip"}
            ],
            "body": changelog,
        }).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = fake_response
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("src.core.updater.urlopen", return_value=mock_resp):
            result = checker.check_for_update()

        assert result is not None
        assert result.release_body == changelog

    def test_missing_body_defaults_to_empty(self):
        """Missing release body should default to empty string."""
        checker = self._make_checker("1.0.0")
        fake_response = json.dumps({
            "tag_name": "v2.0.0",
            "assets": [
                {"name": "Cadence.zip", "browser_download_url": "https://example.com/dl.zip"}
            ],
        }).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = fake_response
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("src.core.updater.urlopen", return_value=mock_resp):
            result = checker.check_for_update()

        assert result is not None
        assert result.release_body == ""
