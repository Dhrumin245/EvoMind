param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("api", "worker")]
    [string]$Role,
    [string]$DeployPath = $env:DEPLOY_PATH,
    [string]$EnvFile = ".env.production"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\common.ps1"

$deployRoot = Resolve-EvoMindDeployPath -DeployPath $DeployPath
$envPath = Join-Path $deployRoot $EnvFile
$logRoot = Join-Path $deployRoot "output_logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$transcriptPath = Join-Path $logRoot "$Role-$timestamp.log"
Start-Transcript -Path $transcriptPath -Append | Out-Null

try {
    Set-Location -LiteralPath $deployRoot
    Import-EvoMindDotEnv -Path $envPath
    Resolve-EvoMindPathEnvironment -DeployPath $deployRoot

    $python = Get-EvoMindVenvPython -DeployPath $deployRoot

    if ($Role -eq "api") {
        $hostName = if ([string]::IsNullOrWhiteSpace($env:EVOMIND_API_HOST)) { "127.0.0.1" } else { $env:EVOMIND_API_HOST }
        $port = if ([string]::IsNullOrWhiteSpace($env:EVOMIND_API_PORT)) { "8000" } else { $env:EVOMIND_API_PORT }
        & $python -m uvicorn api.server:app --host $hostName --port $port
        exit $LASTEXITCODE
    }

    & $python -m api.worker
    exit $LASTEXITCODE
}
finally {
    Stop-Transcript | Out-Null
}
