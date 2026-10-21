@echo off
rem BTC daily paper-observation routine (PAPER ONLY - places no orders).
rem Invoked by Task Scheduler; logs to quant_data\daily_status\runner.log
setlocal
cd /d "%~dp0.."
rem --- kill-switch guard (added by killswitch): abort if a STOP flag is present ---
if not exist quant_data\daily_status mkdir quant_data\daily_status 2>nul
if exist "STOP.flag" ( echo ==== %date% %time% ==== SKIPPED by kill switch ^(STOP.flag^)>> quant_data\daily_status\runner.log & endlocal & exit /b 0 )
if exist "..\GLOBAL_TRADING_STOP.flag" ( echo ==== %date% %time% ==== SKIPPED by GLOBAL kill switch>> quant_data\daily_status\runner.log & endlocal & exit /b 0 )
rem --- end kill-switch guard ---
if not exist quant_data\daily_status mkdir quant_data\daily_status
echo ==== %date% %time% ==== >> quant_data\daily_status\runner.log
venv\Scripts\python.exe -m quant.pipeline btc-daily-routine >> quant_data\daily_status\runner.log 2>&1
echo exit code %errorlevel% >> quant_data\daily_status\runner.log
endlocal
