param([string]$BaselineDir)
. "$PSScriptRoot\_common.ps1"
Start-Log "04_validate"
try {
    if ($BaselineDir) { Invoke-Cli validate --pred-dir $BaselineDir --name sam_yolo }
    $pred = Join-Path $WorkDir "predictions"
    if (Test-Path $pred) {
        Get-ChildItem $pred -Directory | ForEach-Object { Invoke-Cli validate --name $_.Name }
    }
    $rows = Get-ChildItem (Join-Path $WorkDir "reports\validation_*.json") | ForEach-Object {
        $j = Get-Content $_.FullName -Raw | ConvertFrom-Json
        $o = $j.overall
        [pscustomobject]@{
            run         = $j.run
            obj_f1      = [math]::Round($o.'obj@0.5'.f1, 4)
            obj_P       = [math]::Round($o.'obj@0.5'.precision, 4)
            obj_R       = [math]::Round($o.'obj@0.5'.recall, 4)
            pixel_iou   = [math]::Round($o.pixel.iou, 4)
            boundary_f1 = [math]::Round($o.boundary_f1, 4)
            ap50        = [math]::Round($o.ap50, 4)
        }
    }
    $rows | Sort-Object obj_f1 -Descending | Format-Table -AutoSize
} finally { Stop-Transcript | Out-Null }
