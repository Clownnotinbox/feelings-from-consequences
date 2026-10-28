@echo off
rem The drone app: a page in the browser (http://localhost:8765) with the map and the 3D view. One request: text + a click near the target.
rem A separate MuJoCo 3D window as well: run_app.bat --viewer
rem Another place: run_app.bat --place tokyo_res    The same layout again: run_app.bat --seed NUMBER
chcp 65001 >nul
cd /d "%~dp0src"
set PYTHONIOENCODING=utf-8
if exist "%~dp0osm2" (set OSM2_DIR=%~dp0osm2) else (set OSM2_DIR=%~dp0..\..\work\datasets\osm2)
if exist "%~dp0.venv\Scripts\python.exe" (set PY=%~dp0.venv\Scripts\python.exe) else (set PY=%~dp0..\..\work\drone-venv\Scripts\python.exe)
"%PY%" demo_server.py --place kazan_novosav %*
pause
