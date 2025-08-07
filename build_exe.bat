@echo off
echo Building executable for Call Transcription Analyzer...

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
pip install pyinstaller

REM Build executable
echo Building executable...
pyinstaller --onefile --name="CallAnalyzer" --add-data="config.json;." main.py

echo Build complete! Executable is in the 'dist' folder.
pause