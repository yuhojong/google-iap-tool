@echo off
setlocal enabledelayedexpansion

rem =============================================
rem Google IAP Tool - Windows CMD one-click setup
rem =============================================

echo.
echo ===== Preparing environment =====

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR%"=="" set "SCRIPT_DIR=."
cd /d "%SCRIPT_DIR%"

call :ensure_python
if errorlevel 1 goto :error

echo.
echo ===== Creating virtual environment =====
set "VENV_DIR=%SCRIPT_DIR%\.venv"
if not exist "%VENV_DIR%" (
    "%PYTHON_EXE%" -m venv "%VENV_DIR%"
    if errorlevel 1 goto :error
)

set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo Failed to locate virtual environment python at "%VENV_PYTHON%"
    goto :error
)

echo.
echo ===== Installing Python dependencies =====
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto :error
"%VENV_PYTHON%" -m pip install -r "%SCRIPT_DIR%\requirements.txt"
if errorlevel 1 goto :error

call :load_env

set "APP_PORT=8000"
echo.
echo ===== Starting FastAPI server =====
"%VENV_PYTHON%" -m uvicorn main:app --host 0.0.0.0 --port %APP_PORT%
if errorlevel 1 goto :error

goto :eof

:ensure_python
where python >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%I in ('where python') do (
        set "PYTHON_EXE=%%~fI"
        goto :ensure_python_done
    )
)

echo Python not found. Installing Python 3.11 ...
set "PYTHON_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
set "PYTHON_INSTALLER=%TEMP%\python-installer.exe"

if exist "%PYTHON_INSTALLER%" del /f /q "%PYTHON_INSTALLER%" >nul 2>&1

where curl >nul 2>&1
if not errorlevel 1 (
    curl -L -o "%PYTHON_INSTALLER%" "%PYTHON_URL%"
) else (
    echo curl not found. Falling back to PowerShell download...
    powershell -Command "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_INSTALLER%'"
)
if errorlevel 1 (
    echo Failed to download Python installer.
    goto :error
)

echo Running Python installer (this may take a minute)...
start /wait "" "%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 SimpleInstall=1
set "INSTALL_EXIT=%errorlevel%"
del /f /q "%PYTHON_INSTALLER%" >nul 2>&1
if not "%INSTALL_EXIT%"=="0" (
    echo Python installer exited with code %INSTALL_EXIT%.
    goto :error
)

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Python installation succeeded but python.exe is not yet on PATH.
    echo Please close this window and re-run the script from a new terminal.
    goto :error
)

for /f "delims=" %%I in ('where python') do (
    set "PYTHON_EXE=%%~fI"
    goto :ensure_python_done
)

:ensure_python_done
echo Using Python at "%PYTHON_EXE%"
exit /b 0

:load_env
set "ENV_FILE=%SCRIPT_DIR%\.env"
if not exist "%ENV_FILE%" (
    echo Warning: .env file not found at %ENV_FILE%. Skipping environment registration.
    exit /b 0
)

echo.
echo ===== Registering environment variables =====
for /f "usebackq eol=# tokens=1* delims==" %%A in ("%ENV_FILE%") do (
    set "KEY=%%~A"
    set "VALUE=%%~B"

    for /f "tokens=*" %%K in ("!KEY!") do set "KEY=%%K"
    if "!KEY!"=="" (
        rem Skip lines without a key
    ) else (
        for /f "tokens=*" %%V in ("!VALUE!") do set "VALUE=%%V"

        if not "!VALUE!"=="" (
            if "!VALUE:~0,1!"=="\"" if "!VALUE:~-1!"=="\"" set "VALUE=!VALUE:~1,-1!"
            if "!VALUE:~0,1!"=="'" if "!VALUE:~-1!"=="'" set "VALUE=!VALUE:~1,-1!"

            if exist "%SCRIPT_DIR%!VALUE!" (
                for %%P in ("%SCRIPT_DIR%!VALUE!") do set "VALUE=%%~fP"
            )
        )

        set "!KEY!=!VALUE!"
        call set "CURRENT=!%KEY%!"
        setx !KEY! "!CURRENT!" >nul
        echo Set !KEY!=!CURRENT!
    )
)
exit /b 0

:error
echo.
echo ===== Script failed =====
exit /b 1
