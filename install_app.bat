@echo off
rem One-time install of the drone app on a new Windows computer (NVIDIA GPU, Python 3.11 installed, internet).
rem Creates .venv next to this file, installs the packages and downloads the base language model (~3 GB) for the parser.
chcp 65001 >nul
cd /d "%~dp0"
py -3.11 -m venv .venv || python -m venv .venv
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if not exist "models\qwen1.5b" ".venv\Scripts\python.exe" -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-1.5B-Instruct', local_dir='models/qwen1.5b')"
echo.
echo Done. Start the app with run_app.bat
pause
