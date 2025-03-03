REM @echo off
REM Build the application into a single .exe file
REM pyinstaller --onefile --windowed program/main.py
REM pause

@echo off
echo Cleaning previous builds...
rmdir /s /q build
rmdir /s /q dist
echo Building application...
pyinstaller main.spec
echo Build complete.
pause