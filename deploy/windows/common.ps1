Set-StrictMode -Version Latest

function Import-EvoMindDotEnv {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Environment file not found: $Path"
    }

    foreach ($rawLine in Get-Content -LiteralPath $Path) {
        $line = $rawLine.Trim()
        if ($line.Length -eq 0 -or $line.StartsWith("#")) {
            continue
        }
        if ($line.StartsWith("export ")) {
            $line = $line.Substring(7).Trim()
        }

        $separator = $line.IndexOf("=")
        if ($separator -lt 1) {
            continue
        }

        $name = $line.Substring(0, $separator).Trim()
        $value = $line.Substring($separator + 1).Trim()
        if (
            ($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

function Resolve-EvoMindDeployPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DeployPath
    )

    if ([string]::IsNullOrWhiteSpace($DeployPath)) {
        throw "DeployPath is required. Set DEPLOY_PATH or pass -DeployPath."
    }

    $fullPath = [System.IO.Path]::GetFullPath($DeployPath)
    New-Item -ItemType Directory -Force -Path $fullPath | Out-Null
    return $fullPath
}

function Get-EvoMindVenvPython {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DeployPath
    )

    $pythonPath = Join-Path $DeployPath ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        throw "Virtual environment Python not found: $pythonPath"
    }
    return $pythonPath
}

function Resolve-EvoMindPathValue {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DeployPath,
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    if ([System.IO.Path]::IsPathRooted($Value)) {
        return $Value
    }
    return [System.IO.Path]::GetFullPath((Join-Path $DeployPath $Value))
}

function Resolve-EvoMindPathEnvironment {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DeployPath
    )

    $pathNames = @(
        "EVOMIND_DATA_DIR",
        "EVOMIND_BACKUP_DIR",
        "EVOMIND_TENANT_ROOT_DIR",
        "EVOMIND_API_AUTH_DB",
        "EVOMIND_API_EVENTS_DB",
        "EVOMIND_API_JOBS_DB",
        "EVOMIND_CONTROL_PLANE_DB_URL_FILE",
        "EVOMIND_CONTROL_PLANE_DB_PASSWORD_FILE",
        "EVOMIND_WEBHOOK_SECRET_KEY_FILE"
    )

    foreach ($name in $pathNames) {
        $value = [Environment]::GetEnvironmentVariable($name, "Process")
        if (-not [string]::IsNullOrWhiteSpace($value)) {
            [Environment]::SetEnvironmentVariable(
                $name,
                (Resolve-EvoMindPathValue -DeployPath $DeployPath -Value $value),
                "Process"
            )
        }
    }
}
