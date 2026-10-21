# Registers the BTC daily paper-observation routine in Windows Task Scheduler.
# NO ADMIN REQUIRED: the task is created under your own user account
# (schtasks without /RU or /RL HIGHEST registers a per-user task).
#
# Setup (one time), from an ordinary PowerShell prompt:
#     cd <project root>
#     powershell -ExecutionPolicy Bypass -File scripts\setup_btc_daily_task.ps1
#
# Why 20:35 local: the BTC daily bar closes at 00:00 UTC, which is 20:00 ET
# in summer / 19:00 ET in winter; 20:35 local covers both plus Yahoo's
# publication lag. Adjust -At below if your local timezone differs.
#
# The task runs scripts\run_btc_daily_routine.bat, which is PAPER ONLY -
# it invokes `python -m quant.pipeline btc-daily-routine`, a code path that
# is statically verified on every run to import no order-placement code.
#
# Manage it later:
#     schtasks /Query  /TN "BTC Paper Daily Routine"
#     schtasks /Run    /TN "BTC Paper Daily Routine"     (manual test run)
#     schtasks /Delete /TN "BTC Paper Daily Routine" /F  (remove)

$TaskName = "BTC Paper Daily Routine"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Runner = Join-Path $ProjectRoot "scripts\run_btc_daily_routine.bat"

# LAB GUARD: this copy is the research lab. Registering the scheduled task
# from here would repoint daily paper observation at the lab and corrupt the
# observation record. Refuse while LAB_MODE exists at the project root.
if (Test-Path (Join-Path $ProjectRoot "LAB_MODE")) {
    Write-Error ("LAB_MODE detected: this is the research-lab copy. The scheduled " +
                 "task must only be registered from the original V4.1algobot-master " +
                 "folder. Aborting - no task was created or changed.")
    exit 1
}

if (-not (Test-Path $Runner)) {
    Write-Error "runner not found: $Runner"
    exit 1
}

schtasks /Create /F /SC DAILY /ST 20:35 /TN $TaskName /TR "`"$Runner`""
if ($LASTEXITCODE -eq 0) {
    Write-Output ""
    Write-Output "Task '$TaskName' registered (daily 20:35, runs as $env:USERNAME, no admin)."
    Write-Output "Note: per-user tasks only run while you are logged in."
    Write-Output "Test it now with:  schtasks /Run /TN `"$TaskName`""
} else {
    Write-Error "schtasks failed with exit code $LASTEXITCODE"
}
