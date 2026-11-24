@echo off
rem Starts the Voronoi Body Generator (installs dependencies on first run).
cd /d "%~dp0"
set "PY="
where py >nul 2>nul && set "PY=py"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo Python was not found. Install Python 3.13 from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" in the installer.
  pause
  exit /b 1
)
%PY% scripts\start.py %*
if errorlevel 1 (
  echo.
  echo The app did not start or stopped with an error. Read the messages above.
  pause
)
