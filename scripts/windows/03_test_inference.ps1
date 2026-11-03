# Ablation: cascade penuh, cascade tanpa konsep, detector, concept (zero-shot SAM 3).
param([switch]$ZeroShotSam3)
. "$PSScriptRoot\_common.ps1"
Start-Log "03_test_inference"
try {
    $zs = @(); if ($ZeroShotSam3) { $zs = @("--zero-shot-sam3") }
    Invoke-Cli predict --mode detector --name detector
    Invoke-Cli predict --mode concept --name concept
    Invoke-Cli --set infer.use_concept=false predict --mode cascade --name cascade_noconcept @zs
    Invoke-Cli predict --mode cascade --name cascade @zs
} finally { Stop-Transcript | Out-Null }
