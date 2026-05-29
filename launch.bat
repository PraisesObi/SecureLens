@echo off
REM SecureLens launcher.
REM Starts the FastAPI server then opens the browser at the dashboard URL.

setlocal
cd /d "%~dp0"

echo ===============================================
echo   SecureLens · Insider Threat Detection
echo ===============================================
echo.
echo Starting local server on http://127.0.0.1:8000 ...
echo.

REM Open the browser after a short delay so the server has time to come up.
start "" cmd /c "timeout /t 3 >nul && start http://127.0.0.1:8000"

REM Run the FastAPI app in this window so logs are visible.
python app.py

echo.
echo ===============================================
echo   Server stopped. Exit code: %ERRORLEVEL%
echo ===============================================
echo.
pause

endlocal
