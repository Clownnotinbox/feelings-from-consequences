@echo off
rem The drone in a real place: a page in the browser with the map and the 3D view. Double-click to start.
rem Another place: run_demo.bat --place tokyo_res    The same layout again: run_demo.bat --seed NUMBER
chcp 65001 >nul
cd /d "%~dp0src"
set OSM2_DIR=%~dp0..\..\work\datasets\osm2
set PYTHONIOENCODING=utf-8
"%~dp0..\..\work\drone-venv\Scripts\python.exe" demo_server.py --place kazan_novosav %*
pause
