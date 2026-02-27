@echo off
cd /d "%~dp0.."
set UV_CACHE_DIR=%~dp0..\.uv-cache
uv sync --all-extras
uv run pyinstaller scripts/build_exe.spec --clean --noconfirm
