"""
Auto-updater for Cadence.
Checks GitHub Releases for new versions and applies updates.
"""

import json
import os
import re
import sys
import zipfile
import tempfile
import subprocess
from dataclasses import dataclass
from urllib.request import urlopen, Request
from urllib.error import URLError

from PySide6.QtCore import QObject, QThread, Signal

from utils.resource_path import get_app_data_path, is_bundled
from utils.logger import get_logger

GITHUB_REPO = "whorne89/Cadence"
RELEASES_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"


@dataclass
class UpdateInfo:
    """Information about an available update."""
    version_str: str
    tag_name: str
    download_url: str
    release_body: str = ""


def compare_versions(local: str, remote: str) -> int:
    """
    Compare two semantic version strings.

    Returns:
        -1 if local < remote (update available)
         0 if local == remote
         1 if local > remote
    """
    def parse(v: str):
        """Parse version string into tuple of ints."""
        v = v.lstrip("v")
        parts = re.split(r"[.\-]", v)
        result = []
        for p in parts:
            try:
                result.append(int(p))
            except ValueError:
                break
        # Pad to at least 3 parts
        while len(result) < 3:
            result.append(0)
        return tuple(result)

    lp = parse(local)
    rp = parse(remote)

    if lp < rp:
        return -1
    elif lp > rp:
        return 1
    return 0


class UpdateChecker:
    """Checks for and applies Cadence updates from GitHub Releases."""

    def __init__(self):
        self.logger = get_logger()
        try:
            from version import __version__
            self.current_version = __version__
        except Exception:
            self.current_version = "0.0.0"

    def check_for_update(self):
        """
        Check GitHub Releases for a newer version.

        Returns:
            UpdateInfo if a newer version is available, None otherwise.
        """
        try:
            req = Request(RELEASES_URL, headers={"Accept": "application/vnd.github.v3+json"})
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            tag = data.get("tag_name", "")
            # Strip leading 'v' from tag (e.g., "v3.1.0" -> "3.1.0")
            version_str = tag.lstrip("v")

            if not version_str:
                return None

            if compare_versions(self.current_version, version_str) >= 0:
                self.logger.info(f"Up to date: local={self.current_version}, remote={version_str}")
                return None

            # Find .zip asset in release assets
            download_url = None
            for asset in data.get("assets", []):
                if asset.get("name", "").endswith(".zip"):
                    download_url = asset["browser_download_url"]
                    break

            if not download_url:
                self.logger.warning("Update found but no .zip asset in release")
                return None

            release_body = data.get("body", "") or ""

            self.logger.info(f"Update available: {version_str} (current: {self.current_version})")
            return UpdateInfo(
                version_str=version_str,
                tag_name=tag,
                download_url=download_url,
                release_body=release_body,
            )

        except (URLError, json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Update check failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during update check: {e}")
            return None

    def download_update(self, update_info, progress_callback=None):
        """
        Download the update asset to .cadence/updates/.

        Args:
            update_info: UpdateInfo with the download URL.
            progress_callback: Optional callable(downloaded_bytes, total_bytes).

        Returns:
            Path to the downloaded file, or None on failure.
        """
        try:
            updates_dir = get_app_data_path("updates")
            filename = f"Cadence-{update_info.version_str}.zip"
            dest_path = os.path.join(updates_dir, filename)

            # Remove previous download if it exists
            if os.path.exists(dest_path):
                os.remove(dest_path)

            req = Request(update_info.download_url)
            with urlopen(req, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 64 * 1024

                with open(dest_path, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(downloaded, total)

            self.logger.info(f"Update downloaded to {dest_path}")
            return dest_path

        except Exception as e:
            self.logger.error(f"Update download failed: {e}")
            return None

    def apply_update(self, downloaded_path):
        """
        Apply an update for bundled EXE installs.

        Extracts the ZIP to the system temp directory, writes a batch script
        that waits for this process to exit, copies new files over the
        existing install, relaunches, and cleans up.

        Args:
            downloaded_path: Path to the downloaded .zip file.

        Returns:
            True if the update script was launched, False on failure.
        """
        if not is_bundled():
            self.logger.warning("apply_update called on non-bundled install")
            return False

        try:
            app_dir = os.path.dirname(sys.executable)

            # Extract to system temp (not inside app dir) to avoid clutter
            temp_root = tempfile.mkdtemp(prefix="cadence_update_")

            with zipfile.ZipFile(downloaded_path, "r") as zf:
                zf.extractall(temp_root)

            # If the ZIP contains a single top-level folder, use its contents
            source_dir = temp_root
            entries = os.listdir(temp_root)
            if len(entries) == 1 and os.path.isdir(os.path.join(temp_root, entries[0])):
                source_dir = os.path.join(temp_root, entries[0])

            pid = os.getpid()
            exe_path = sys.executable

            # Write batch script to system temp
            bat_path = os.path.join(tempfile.gettempdir(), "_cadence_update.bat")
            log_path = os.path.join(tempfile.gettempdir(), "_cadence_update.log")

            bat_content = (
                "@echo off\n"
                f'echo [%date% %time%] Update script started > "{log_path}"\n'
                f'echo [%date% %time%] Waiting for PID {pid} to exit >> "{log_path}"\n'
                ":waitloop\n"
                f'tasklist /FI "PID eq {pid}" 2>NUL | find /I "{pid}" >NUL\n'
                "if not errorlevel 1 (\n"
                "    timeout /t 1 /nobreak >NUL\n"
                "    goto waitloop\n"
                ")\n"
                f'echo [%date% %time%] Process exited, cleaning old dist-info >> "{log_path}"\n'
                f'pushd "{app_dir}\\_internal"\n'
                f'for /D %%d in (cadence-*.dist-info) do (\n'
                f'    echo [%date% %time%] Removing %%d >> "{log_path}"\n'
                f'    rd /S /Q "%%d" 2>NUL\n'
                f')\n'
                f'popd\n'
                f'echo [%date% %time%] Copying new files >> "{log_path}"\n'
                f'xcopy /E /Y /Q "{source_dir}\\*" "{app_dir}\\" >> "{log_path}" 2>&1\n'
                f'echo [%date% %time%] xcopy exit code: %errorlevel% >> "{log_path}"\n'
                f'echo [%date% %time%] Relaunching >> "{log_path}"\n'
                f'start "" "{exe_path}"\n'
                f'rd /S /Q "{temp_root}" 2>NUL\n'
                f'del /F /Q "{downloaded_path}" 2>NUL\n'
                f'echo [%date% %time%] Cleanup done >> "{log_path}"\n'
                '(goto) 2>NUL & del /F /Q "%~f0"\n'
            )

            with open(bat_path, "w", encoding="utf-8") as f:
                f.write(bat_content)

            self.logger.info(f"Update script: {bat_path}")
            self.logger.info(f"Source: {source_dir} -> {app_dir}")

            # Launch the batch script as a new process group so it survives
            # our exit. CREATE_NEW_PROCESS_GROUP keeps it alive;
            # CREATE_NO_WINDOW suppresses the console.
            subprocess.Popen(
                ["cmd.exe", "/c", bat_path],
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    | subprocess.CREATE_NO_WINDOW
                ),
            )

            self.logger.info("Update script launched, application will restart")
            return True

        except Exception as e:
            self.logger.error(f"Failed to apply update: {e}")
            return False

    @staticmethod
    def get_source_update_message(update_info):
        """
        Get instructions for updating a source (non-bundled) install.

        Args:
            update_info: UpdateInfo for the available update.

        Returns:
            String with update instructions.
        """
        return f"Cadence {update_info.version_str} is available.\nRun: git pull && uv sync"


class UpdateWorker(QObject):
    """
    QObject worker that checks for updates in a background QThread.

    Signals:
        update_available(UpdateInfo): Emitted when a newer version is found.
        finished(): Emitted when the check completes (regardless of result).
    """

    update_available = Signal(object)  # UpdateInfo
    finished = Signal()

    def __init__(self):
        super().__init__()
        self._checker = UpdateChecker()

    def run(self):
        """Perform the update check. Call from a QThread."""
        try:
            info = self._checker.check_for_update()
            if info is not None:
                self.update_available.emit(info)
        finally:
            self.finished.emit()
