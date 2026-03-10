@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Refreshes version.json with current git commit and UTC timestamp.
rem Requires: git, PowerShell

where git >nul 2>nul
if errorlevel 1 (
  echo [ERROR] git not found in PATH.
  exit /b 1
)

for /f "usebackq delims=" %%i in (`git rev-parse --short HEAD 2^>nul`) do set "COMMIT=%%i"
if not defined COMMIT (
  echo [ERROR] Unable to determine git commit.
  exit /b 1
)

for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "[DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ss.fffZ')"`) do set "BUILT_AT=%%i"
if not defined BUILT_AT (
  echo [ERROR] Unable to determine UTC time.
  exit /b 1
)

set "OUTFILE=%~dp0version.json"
set "TMPFILE=%OUTFILE%.tmp"

(
  echo {
  echo   "commit": "!COMMIT!",
  echo   "builtAt": "!BUILT_AT!"
  echo }
) > "!TMPFILE!"

move /y "!TMPFILE!" "!OUTFILE!" >nul
echo Updated version.json: commit=!COMMIT! builtAt=!BUILT_AT!

endlocal
