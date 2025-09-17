@echo off
cd /d "C:\github\whisperthing"
call venv\Scripts\activate.bat
python main.py
if %ERRORLEVEL% neq 0 (
	echo Application crashed, restarting in 30s...
	timeout /t 30 /nobreak
	goto :start
)