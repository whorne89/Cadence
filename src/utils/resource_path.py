"""
Resource path utilities for Cadence.
Handles path resolution for both development and PyInstaller bundled EXE.
"""

import os
import sys
from pathlib import Path


def _get_app_root():
    """
    Get the root directory of the application.
    When running as bundled EXE: directory containing the .exe.
    When running as script: project root (parent of src/).
    """
    if hasattr(sys, '_MEIPASS'):
        return Path(os.path.dirname(sys.executable))
    else:
        # src/utils/resource_path.py -> go up two levels to project root
        return Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def get_resource_path(relative_path=""):
    """
    Get absolute path to a bundled resource (icons, etc.).
    In dev: src/resources/
    In bundled EXE: sys._MEIPASS/resources/
    """
    if hasattr(sys, '_MEIPASS'):
        base_path = os.path.join(sys._MEIPASS, 'resources')
    else:
        base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources')

    if relative_path:
        return os.path.join(base_path, relative_path)
    return base_path


def get_app_data_path(subdir=""):
    """
    Get path to application data directory (.cadence/).
    Stored relative to the application directory for portability.
    """
    app_data = _get_app_root() / ".cadence"
    if subdir:
        path = app_data / subdir
    else:
        path = app_data
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def is_bundled():
    """Check if running as a PyInstaller bundle."""
    return hasattr(sys, '_MEIPASS')
