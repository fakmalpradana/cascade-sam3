# inspect -> konfirmasi -> convert-ecw -> prepare (catalog + make-testsets + build-dataset)
param([string]$City)
. "$PSScriptRoot\_common.ps1"
Start-Log "01_prepare_data"
try {
    $c = @(); if ($City) { $c = @("--city", $City) }
    Invoke-Cli inspect @c
    Write-Host "Baca $WorkDir\reports\data_report.md, sesuaikan configs\local.yaml (label_filter, coverage_mode, target_gsd_m)." -ForegroundColor Yellow
    $ans = Read-Host "Lanjut convert-ecw + prepare? (y/N)"
    if ($ans -notmatch '^[yY]') { Write-Host "Berhenti."; return }
    Invoke-Cli convert-ecw @c
    Invoke-Cli prepare @c
    Write-Host "Cek quicklook di $WorkDir\testsets\*\quicklook.png dan overlay di $WorkDir\dataset\overlays" -ForegroundColor Green
} finally { Stop-Transcript | Out-Null }
