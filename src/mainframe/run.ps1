# Terrabot Mainframe Runner
# Convenient script to run the mainframe executable

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Terrabot Mainframe" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$EXECUTABLE = "..\..\build\bin\Debug\terrabot_mainframe.exe"

# Check if executable exists
if (-not (Test-Path $EXECUTABLE)) {
    Write-Host "ERROR: Executable not found at $EXECUTABLE" -ForegroundColor Red
    Write-Host "Please build the mainframe first using build.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "Starting Terrabot Mainframe..." -ForegroundColor Green
Write-Host "Executable: $EXECUTABLE" -ForegroundColor Gray
Write-Host ""

# Run the mainframe
& $EXECUTABLE

Write-Host "`nMainframe terminated." -ForegroundColor Yellow
