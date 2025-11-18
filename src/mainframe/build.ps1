# Terrabot Mainframe Build Script
# Compiles the C++ mainframe with necessary dependencies

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Terrabot Mainframe Build Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Configuration
$SOURCE_FILE = "main.cpp"
$OUTPUT_FILE = "terrabot_mainframe.exe"
$COMPILER = "g++"

# Compiler flags
$CFLAGS = @(
    "-std=c++17",           # C++17 standard
    "-O2",                  # Optimization level 2
    "-Wall",                # Enable all warnings
    "-Wextra",              # Extra warnings
    "-pthread"              # POSIX threads support
)

# Linker flags for Windows
$LDFLAGS = @(
    "-lws2_32"              # Winsock2 library for Windows sockets
)

# Check if compiler exists
Write-Host "`nChecking for C++ compiler..." -ForegroundColor Yellow
$compilerCheck = Get-Command $COMPILER -ErrorAction SilentlyContinue

if (-not $compilerCheck) {
    Write-Host "ERROR: $COMPILER not found!" -ForegroundColor Red
    Write-Host "Please install MinGW-w64 or MSYS2 with g++ compiler" -ForegroundColor Red
    Write-Host "Download from: https://www.msys2.org/" -ForegroundColor Yellow
    exit 1
}

Write-Host "Compiler found: $($compilerCheck.Path)" -ForegroundColor Green

# Check if source file exists
if (-not (Test-Path $SOURCE_FILE)) {
    Write-Host "ERROR: Source file '$SOURCE_FILE' not found!" -ForegroundColor Red
    exit 1
}

Write-Host "Source file found: $SOURCE_FILE" -ForegroundColor Green

# Build command
$buildCommand = "$COMPILER $($CFLAGS -join ' ') $SOURCE_FILE -o $OUTPUT_FILE $($LDFLAGS -join ' ')"

Write-Host "`nBuilding mainframe..." -ForegroundColor Yellow
Write-Host "Command: $buildCommand" -ForegroundColor Gray

# Execute build
try {
    $output = Invoke-Expression $buildCommand 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nBuild successful!" -ForegroundColor Green
        Write-Host "Output: $OUTPUT_FILE" -ForegroundColor Green
        
        # Show file info
        if (Test-Path $OUTPUT_FILE) {
            $fileInfo = Get-Item $OUTPUT_FILE
            Write-Host "Size: $([math]::Round($fileInfo.Length / 1KB, 2)) KB" -ForegroundColor Cyan
            Write-Host "Created: $($fileInfo.CreationTime)" -ForegroundColor Cyan
        }
        
        Write-Host "`nTo run the mainframe, execute:" -ForegroundColor Yellow
        Write-Host "  .\$OUTPUT_FILE" -ForegroundColor White
    } else {
        Write-Host "`nBuild failed!" -ForegroundColor Red
        Write-Host $output -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "`nBuild error: $_" -ForegroundColor Red
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Build Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
