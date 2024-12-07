@echo off
REM Build the application into a single .exe file
pyinstaller --onefile --windowed main.py
pause