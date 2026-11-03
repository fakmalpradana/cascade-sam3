. "$PSScriptRoot\_common.ps1"
Start-Log "00_check_env"
try { Invoke-Cli check-env --smoke } finally { Stop-Transcript | Out-Null }
