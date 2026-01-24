# Set Python 3.12 as Default
# Run this script as Administrator: Right-click -> Run as Administrator
# Or run: powershell -ExecutionPolicy Bypass -File set_python312_default.ps1

Write-Host "Setting Python 3.12 as default..." -ForegroundColor Cyan

# Common Python 3.12 installation paths
$possiblePaths = @(
    "C:\Python312",
    "C:\Python312\Scripts",
    "C:\Program Files\Python312",
    "C:\Program Files\Python312\Scripts",
    "$env:LOCALAPPDATA\Programs\Python\Python312",
    "$env:LOCALAPPDATA\Programs\Python\Python312\Scripts",
    "$env:USERPROFILE\AppData\Local\Programs\Python\Python312",
    "$env:USERPROFILE\AppData\Local\Programs\Python\Python312\Scripts"
)

# Find which Python 3.12 paths exist
$existingPaths = @()
foreach ($path in $possiblePaths) {
    if (Test-Path $path) {
        $existingPaths += $path
        Write-Host "Found: $path" -ForegroundColor Green
    }
}

if ($existingPaths.Count -eq 0) {
    Write-Host "ERROR: Python 3.12 not found in common locations." -ForegroundColor Red
    Write-Host "Please install Python 3.12 or specify the path manually." -ForegroundColor Yellow
    
    # Try to find any Python installation
    Write-Host "`nSearching for Python installations..." -ForegroundColor Cyan
    $pythonExes = Get-Command python* -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
    if ($pythonExes) {
        Write-Host "Found Python installations:" -ForegroundColor Yellow
        $pythonExes | ForEach-Object { Write-Host "  $_" }
    }
    exit 1
}

# Get current PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
Write-Host "`nCurrent User PATH:" -ForegroundColor Cyan
Write-Host $currentPath

# Remove any existing Python paths (to avoid duplicates)
$pathArray = $currentPath -split ";" | Where-Object { 
    $_ -and ($_ -notmatch "Python3\d+" -or $_ -match "Python312")
}

# Add Python 3.12 paths at the beginning
$newPathArray = $existingPaths + $pathArray | Select-Object -Unique
$newPath = $newPathArray -join ";"

# Set the new PATH
[Environment]::SetEnvironmentVariable("Path", $newPath, "User")

Write-Host "`nNew User PATH set:" -ForegroundColor Green
Write-Host $newPath

# Also update current session
$env:Path = $newPath + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")

# Verify
Write-Host "`n--- Verification ---" -ForegroundColor Cyan
Write-Host "Python version:" -ForegroundColor Yellow
& python --version

Write-Host "`nPython location:" -ForegroundColor Yellow
& where.exe python

Write-Host "`nDone! Restart your terminal for changes to take full effect." -ForegroundColor Green
Write-Host "You can verify by running: python --version" -ForegroundColor Cyan
