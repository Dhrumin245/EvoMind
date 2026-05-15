param(
    [string]$SourcePath = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string]$DeployPath = $env:DEPLOY_PATH,
    [string]$Python = $env:PYTHON_EXE,
    [string]$EnvFile = ".env.production",
    [int]$HealthTimeoutSeconds = 180,
    [switch]$SkipTaskRegistration
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\common.ps1"

function Invoke-RobocopyMirror {
    param(
        [Parameter(Mandatory = $true)]
        [string]$From,
        [Parameter(Mandatory = $true)]
        [string]$To
    )

    $excludedDirs = @(
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "artifacts",
        "backups",
        "data",
        "models",
        "output_logs",
        "secrets",
        "tests\.tmp"
    )
    $excludedFiles = @(".env", ".env.production", "*.pyc", "*.pyo", "*.pyd", "*.db", "*.sqlite", "*.sqlite3", "*.log")
    $arguments = @($From, $To, "/MIR", "/R:2", "/W:2", "/NP", "/XD") + $excludedDirs + @("/XF") + $excludedFiles

    & robocopy @arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -gt 7) {
        throw "Robocopy failed with exit code $exitCode"
    }
}

function Resolve-BasePython {
    param([string]$RequestedPython)

    if (-not [string]::IsNullOrWhiteSpace($RequestedPython)) {
        return $RequestedPython
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $pythonCommand) {
        return $pythonCommand.Source
    }

    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $pyCommand) {
        return $pyCommand.Source
    }

    throw "Python was not found. Install Python 3.13 or set PYTHON_EXE."
}

function Register-EvoMindTask {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TaskName,
        [Parameter(Mandatory = $true)]
        [string]$Role,
        [Parameter(Mandatory = $true)]
        [string]$DeployRoot,
        [Parameter(Mandatory = $true)]
        [string]$EnvFileName
    )

    $existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -ne $existingTask) {
        Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    }

    $pwshCommand = Get-Command pwsh -ErrorAction SilentlyContinue
    $powerShellExe = ""
    if ($null -ne $pwshCommand) {
        $powerShellExe = $pwshCommand.Source
    }
    if ([string]::IsNullOrWhiteSpace($powerShellExe)) {
        $powerShellExe = (Get-Command powershell -ErrorAction Stop).Source
    }

    $runner = Join-Path $DeployRoot "deploy\windows\run-service.ps1"
    $argument = '-NoProfile -ExecutionPolicy Bypass -File "' + $runner + '" -Role ' + $Role + ' -DeployPath "' + $DeployRoot + '" -EnvFile "' + $EnvFileName + '"'
    $action = New-ScheduledTaskAction -Execute $powerShellExe -Argument $argument
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 1)

    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Principal $principal `
        -Settings $settings `
        -Description "EvoMind $Role process" `
        -Force | Out-Null
}

function Wait-EvoMindReadiness {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url,
        [int]$TimeoutSeconds
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 5
            if ($response.StatusCode -eq 200) {
                return
            }
        }
        catch {
            Start-Sleep -Seconds 5
        }
    } while ((Get-Date) -lt $deadline)

    throw "EvoMind API did not become ready before timeout: $Url"
}

$sourceRoot = (Resolve-Path -LiteralPath $SourcePath).Path
$deployRoot = Resolve-EvoMindDeployPath -DeployPath $DeployPath

Write-Host "Mirroring source from $sourceRoot to $deployRoot"
Invoke-RobocopyMirror -From $sourceRoot -To $deployRoot

$envPath = Join-Path $deployRoot $EnvFile
if (-not (Test-Path -LiteralPath $envPath)) {
    throw "Create $envPath from .env.production.example and fill in production values before deploying."
}

$basePython = Resolve-BasePython -RequestedPython $Python
$venvDir = Join-Path $deployRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating virtual environment at $venvDir"
    & $basePython -m venv $venvDir
}

Write-Host "Installing Python dependencies"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $deployRoot "requirements.txt")

if (-not $SkipTaskRegistration) {
    $apiTaskName = if ([string]::IsNullOrWhiteSpace($env:EVOMIND_API_TASK_NAME)) { "EvoMindApi" } else { $env:EVOMIND_API_TASK_NAME }
    $workerTaskName = if ([string]::IsNullOrWhiteSpace($env:EVOMIND_WORKER_TASK_NAME)) { "EvoMindWorker" } else { $env:EVOMIND_WORKER_TASK_NAME }

    Write-Host "Registering scheduled tasks"
    Register-EvoMindTask -TaskName $apiTaskName -Role "api" -DeployRoot $deployRoot -EnvFileName $EnvFile
    Register-EvoMindTask -TaskName $workerTaskName -Role "worker" -DeployRoot $deployRoot -EnvFileName $EnvFile

    Start-ScheduledTask -TaskName $workerTaskName
    Start-ScheduledTask -TaskName $apiTaskName

    $readinessUrl = if ([string]::IsNullOrWhiteSpace($env:EVOMIND_READINESS_URL)) {
        "http://127.0.0.1:8000/health/readiness"
    } else {
        $env:EVOMIND_READINESS_URL
    }
    Wait-EvoMindReadiness -Url $readinessUrl -TimeoutSeconds $HealthTimeoutSeconds
}

Write-Host "Windows deployment complete: $deployRoot"
