param([switch]$SkipDetector, [switch]$SkipSam3, [string]$Name, [switch]$Resume)
. "$PSScriptRoot\_common.ps1"
Start-Log "02_train"
try {
    $extra = @(); if ($Name) { $extra += @("--name", $Name) }; if ($Resume) { $extra += "--resume" }
    if (-not $SkipDetector) { Invoke-Cli train-detector @extra }
    if (-not $SkipSam3) { Invoke-Cli train-sam3 @extra }
} finally { Stop-Transcript | Out-Null }
