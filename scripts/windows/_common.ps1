# Dipanggil lewat dot-source oleh skrip lain.
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$Cli = Join-Path $Root ".venv\Scripts\cascade-sam3.exe"
if (-not (Test-Path $Cli)) { throw "Belum ada .venv. Jalankan scripts\windows\setup_env.ps1 dulu." }

function Invoke-Cli {
    Write-Host ">> cascade-sam3 $args" -ForegroundColor Cyan
    & $Cli @args
    if ($LASTEXITCODE -ne 0) { throw "cascade-sam3 $args gagal (exit $LASTEXITCODE)" }
}

function Start-Log([string]$Name) {
    $wd = (& $Cli show-config --key work_dir | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Gagal membaca work_dir dari config" }
    $logs = Join-Path $wd "logs"
    New-Item -ItemType Directory -Force -Path $logs | Out-Null
    $script:WorkDir = $wd
    Start-Transcript -Path (Join-Path $logs ("{0}_{1}.ps1.log" -f $Name, (Get-Date -Format yyyyMMdd_HHmmss))) | Out-Null
}
