@echo off
rem ============================================================
rem  VailSight - bring up the radar + camera with data collection
rem  Usage:  collect_data.bat [label] [seconds] [dca]
rem    collect_data.bat                     (asks for label/duration)
rem    collect_data.bat metal_can 30        (no questions asked)
rem    collect_data.bat metal_can 30 dca    (+ raw ADC via DCA1000 FPGA)
rem  Everything is recorded to logs\session_* and a full analysis
rem  report is printed when the run ends (q or timeout).
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"

rem ---- rig calibration (edit after calibrating - see calibrate_radar_camera.py)
set "CAM=1"
set "HFOV=60"
set "YAW_OFFSET=0"
set "RADAR_HEIGHT=1.0"

set "LABEL=%~1"
set "SECS=%~2"
set "DCA=%~3"
set "INTERACTIVE="
if "%LABEL%"=="" (
    set "INTERACTIVE=1"
    echo.
    echo  VailSight data collection
    echo  -------------------------
    echo  Label examples: metal_can, empty_chair, person_walking, fabric_sofa
    echo  ^(the label tags every recorded object signature for ML training^)
    echo.
    set /p "LABEL=Session label (Enter = unlabeled): "
    set /p "SECS=Auto-stop after seconds (Enter = run until 'q'): "
    set /p "DCA=Raw ADC capture via DCA1000? (y = yes, Enter = no): "
)

rem ---- pre-flight: don't waste a recording session on broken code
python -m pytest tests -q --no-header >nul 2>&1
if errorlevel 1 (
    echo.
    echo *** Test suite FAILED - fix the code before recording:
    python -m pytest tests -q --no-header
    if defined INTERACTIVE pause
    exit /b 1
)

set "ARGS=--cam %CAM% --hfov %HFOV% --yaw-offset %YAW_OFFSET% --radar-height %RADAR_HEIGHT%"
if not "%LABEL%"=="" set "ARGS=%ARGS% --label %LABEL%"
if not "%SECS%"=="" set "ARGS=%ARGS% --max-seconds %SECS%"
if /i "%DCA%"=="dca" set "ARGS=%ARGS% --dca"
if /i "%DCA%"=="y" set "ARGS=%ARGS% --dca"

echo.
echo Starting: python live_radar_camera.py %ARGS%
echo (q = stop and save, s = snapshot)
echo.
python live_radar_camera.py %ARGS%
if errorlevel 1 (
    echo.
    echo *** Failed to start. Common causes:
    echo ***  - 'Access is denied' on COM7: another instance still runs - close it.
    echo ***  - DCA capture: FPGA not reachable - check Ethernet 2 is 192.168.33.30
    echo ***    and the DCA1000 is powered (see dca1000.py).
    if defined INTERACTIVE pause
    exit /b 1
)

echo.
echo ================= SESSION REPORT =================
python analyze_session.py
echo ==================================================
echo Report + CSV files are inside the session folder above.
if defined INTERACTIVE pause
