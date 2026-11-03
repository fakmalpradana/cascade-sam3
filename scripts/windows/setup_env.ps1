# Buat .venv Python 3.12, pasang torch CUDA 12.8 + paket repo.
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

function Run { & $args[0] @($args | Select-Object -Skip 1); if ($LASTEXITCODE -ne 0) { throw "Gagal: $args" } }

& py -3.12 --version
if ($LASTEXITCODE -ne 0) { throw "Python 3.12 tidak ditemukan. Pasang dari python.org (centang 'py launcher')." }
if (-not (Test-Path ".venv")) { Run py -3.12 -m venv .venv }
$Py = Join-Path $Root ".venv\Scripts\python.exe"
Run $Py -m pip install --upgrade pip
Run $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
Run $Py -m pip install -e ".[train,dev]"
if (-not (Test-Path "configs\local.yaml")) {
    Copy-Item "configs\local.example.yaml" "configs\local.yaml"
    Write-Host "configs\local.yaml dibuat dari contoh; periksa path data & work_dir." -ForegroundColor Yellow
}
$qgis = @(Get-ChildItem "C:\Program Files\QGIS*\bin\gdal_translate.exe", "C:\OSGeo4W*\bin\gdal_translate.exe" -ErrorAction SilentlyContinue)
if (-not $qgis) { Write-Host "QGIS LTR belum terdeteksi: pasang dari qgis.org (dibutuhkan untuk ECW)." -ForegroundColor Yellow }
Run $Py -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
Write-Host ""
Write-Host "Langkah berikut:" -ForegroundColor Green
Write-Host "  1. Pastikan akses https://huggingface.co/facebook/sam3 sudah disetujui."
Write-Host "  2. .venv\Scripts\hf.exe auth login"
Write-Host "  3. scripts\windows\00_check_env.ps1"
