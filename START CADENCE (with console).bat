@echo off
title Cadence - Meeting Transcription
color 0A

echo.
echo ================================================
echo        CADENCE - Meeting Transcription
echo ================================================
echo.
echo Starting application...
echo.
echo Click START RECORDING in the window or
echo use the system tray icon to begin.
echo.
echo First recording: ~5-10 seconds to load model
echo After that: real-time transcription
echo.
echo ================================================
echo.

cd /d "%~dp0"

REM Use local cache to avoid OneDrive hardlink issues
set UV_CACHE_DIR=%~dp0.uv-cache

uv sync
uv run python src\main.py

echo.
echo ================================================
echo Application stopped.
echo ================================================
pause
exit /b 0

:error
echo.
echo ================================================
echo Failed to start Cadence
echo ================================================
pause
exit /b 1
