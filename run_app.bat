@echo off
title ScratchNet Flower Classifier
echo ========================================================
echo   ScratchNet 7-Class Flower Classifier
echo   Starting web application...
echo ========================================================
echo.
py app.py
if errorlevel 1 (
    echo.
    echo [!] Encountered an issue running with 'py'. Trying 'python'...
    python app.py
)
pause
