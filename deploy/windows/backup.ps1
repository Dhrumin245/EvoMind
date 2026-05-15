param(
    [string]$DeployPath = $env:DEPLOY_PATH,
    [string]$EnvFile = ".env.production",
    [string]$OutputDir = "",
    [int]$KeepLast = 14,
    [int]$MaxAgeDays = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\common.ps1"

$deployRoot = Resolve-EvoMindDeployPath -DeployPath $DeployPath
Set-Location -LiteralPath $deployRoot

Import-EvoMindDotEnv -Path (Join-Path $deployRoot $EnvFile)
Resolve-EvoMindPathEnvironment -DeployPath $deployRoot

$python = Get-EvoMindVenvPython -DeployPath $deployRoot
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = if ([string]::IsNullOrWhiteSpace($env:EVOMIND_BACKUP_DIR)) { "backups" } else { $env:EVOMIND_BACKUP_DIR }
}
$resolvedOutputDir = Resolve-EvoMindPathValue -DeployPath $deployRoot -Value $OutputDir

& $python scripts\backup_job.py `
    --output-dir $resolvedOutputDir `
    --keep-last $KeepLast `
    --max-age-days $MaxAgeDays
exit $LASTEXITCODE
