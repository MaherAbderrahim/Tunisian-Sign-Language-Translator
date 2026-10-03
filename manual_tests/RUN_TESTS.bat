@echo off
title TSL Translator - Manual Test Runner
setlocal EnableDelayedExpansion

:: -----------------------------------------------
:: RUN_TESTS.bat — Start TSL Translator and open browser
:: Usage: Double-click or run from cmd
:: -----------------------------------------------

set "PROJECT_ROOT=%~dp0.."
set "BACKEND_PORT=8000"
set "FRONTEND_PORT=3000"

echo [TSL] Starting TSL Translator Manual Test...
echo.

:: --- Check Python ---
echo [TSL] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [TSL] [FAIL] Python not found. Install Python and add it to PATH.
    goto :fail
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do echo [TSL] [OK] Found: Python %%v

:: --- Check .env ---
if not exist "%PROJECT_ROOT%\.env" (
    echo [TSL] [FAIL] .env file not found at %PROJECT_ROOT%\.env
    goto :fail
)
echo [TSL] [OK] .env found

:: --- Check models ---
if not exist "%PROJECT_ROOT%\models" (
    echo [TSL] [FAIL] models\ directory not found at %PROJECT_ROOT%\models
    goto :fail
)
set "MODEL_COUNT=0"
for /r "%PROJECT_ROOT%\models" %%f in (*) do set /a MODEL_COUNT+=1
if "!MODEL_COUNT!"=="0" (
    echo [TSL] [FAIL] No model files found in models\
    goto :fail
)
echo [TSL] [OK] Found !MODEL_COUNT! file(s) in models\

:: --- Check start.py ---
if not exist "%PROJECT_ROOT%\start.py" (
    echo [TSL] [FAIL] start.py not found at %PROJECT_ROOT%\start.py
    goto :fail
)
echo [TSL] [OK] start.py found

:: --- Start the server ---
echo [TSL] Starting TSL Translator (python start.py)...
start "TSL-Server" /D "%PROJECT_ROOT%" python start.py
echo [TSL] Server process started in background.

:: --- Wait for port 8000 ---
echo [TSL] Waiting for port %BACKEND_PORT%...
set "ELAPSED=0"
:wait_loop
if !ELAPSED! geq 60 (
    echo [TSL] [FAIL] Port %BACKEND_PORT% did not open within 60 seconds
    goto :fail
)
powershell -Command "$c = New-Object System.Net.Sockets.TcpClient; try { $c.Connect('127.0.0.1', %BACKEND_PORT%); $c.Close(); exit 0 } catch { exit 1 }" >nul 2>&1
if %errorlevel% equ 0 (
    echo [TSL] [OK] Port %BACKEND_PORT% is open
    goto :port_ready
)
set /a ELAPSED+=1
if !ELAPSED! mod 10 equ 0 echo [TSL] Still waiting... !ELAPSED! s
timeout /t 1 /nobreak >nul
goto :wait_loop

:port_ready

:: --- Open browser ---
echo [TSL] Opening browser at http://localhost:%FRONTEND_PORT%
start "" "http://localhost:%FRONTEND_PORT%"

echo.
echo [TSL] TSL Translator is running. Close the "TSL-Server" window to stop.
echo [TSL] Press any key to stop the server and exit...
pause >nul

:: --- Stop the server ---
echo [TSL] Stopping server...
taskkill /FI "WindowTitle eq TSL-Server" /T /F >nul 2>&1
taskkill /IM python.exe /F >nul 2>&1
echo [TSL] Cleaned up. Goodbye.
goto :end

:fail
echo.
echo [TSL] Test run failed. Fix the issue above and try again.
pause
exit /b 1

:end
endlocal
