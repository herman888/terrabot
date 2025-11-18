# Terrabot System Launcher
# Starts both the mainframe and detection system

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Terrabot Complete System Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$MAINFRAME = "..\..\build\bin\Debug\terrabot_mainframe.exe"
$DETECTOR = "..\AI\Detection\snow_detector_realtime.py"
$PYTHON = "python"

# Check if mainframe exists
if (-not (Test-Path $MAINFRAME)) {
    Write-Host "ERROR: Mainframe executable not found!" -ForegroundColor Red
    Write-Host "Path checked: $MAINFRAME" -ForegroundColor Yellow
    Write-Host "Please build the mainframe first." -ForegroundColor Yellow
    exit 1
}

# Check if detector exists
if (-not (Test-Path $DETECTOR)) {
    Write-Host "ERROR: Snow detector not found!" -ForegroundColor Red
    Write-Host "Path checked: $DETECTOR" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n[1/2] Starting C++ Mainframe..." -ForegroundColor Green
Write-Host "Mainframe will listen on port 5555 for detector connection" -ForegroundColor Gray

# Start mainframe in background
$mainframeJob = Start-Job -ScriptBlock {
    param($exe)
    & $exe
} -ArgumentList (Resolve-Path $MAINFRAME)

Start-Sleep -Seconds 2

Write-Host "`n[2/2] Starting Python Snow Detector..." -ForegroundColor Green
Write-Host "Detector will connect to mainframe and stream detection data" -ForegroundColor Gray

# Wait a moment for mainframe to initialize
Start-Sleep -Seconds 1

Write-Host "`nLaunching detector..." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop both systems" -ForegroundColor Yellow
Write-Host ""

# Run detector in foreground
try {
    & $PYTHON $DETECTOR
} finally {
    Write-Host "`nStopping mainframe..." -ForegroundColor Yellow
    Stop-Job $mainframeJob
    Remove-Job $mainframeJob
    Write-Host "System shutdown complete." -ForegroundColor Green
}
