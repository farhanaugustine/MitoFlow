@echo off
SETLOCAL
SET "ROOT=%~dp0"
pushd "%ROOT%" >nul

REM Prefer a local virtual environment if present, otherwise fall back to PATH
SET "PYTHON=%ROOT%\.venv\Scripts\python.exe"
IF EXIST "%PYTHON%" (
    "%PYTHON%" "%ROOT%\mito_gui_fixed.py"
) ELSE (
    python "%ROOT%\mito_gui_fixed.py"
)
SET "EXITCODE=%ERRORLEVEL%"

popd >nul
IF NOT "%EXITCODE%"=="0" (
    ECHO.
    ECHO [ERROR] Mito GUI exited with code %EXITCODE%.
    ECHO Review the on-screen log for details.
    PAUSE
)
ENDLOCAL & EXIT /B %EXITCODE%
