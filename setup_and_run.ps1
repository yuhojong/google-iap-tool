#requires -version 5.1
$ErrorActionPreference = 'Stop'

function Write-Section($message) {
    Write-Host "`n=== $message ===" -ForegroundColor Cyan
}

function Get-PythonPath {
    try {
        $cmd = Get-Command python -ErrorAction Stop
        return $cmd.Path
    } catch {
        return $null
    }
}

function Ensure-Python {
    $pythonPath = Get-PythonPath
    if ($pythonPath) {
        Write-Host "Python already installed at $pythonPath" -ForegroundColor Green
        return $pythonPath
    }

    Write-Section "Installing Python 3.11"
    $pythonInstallerUrl = 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe'
    $installerPath = Join-Path $env:TEMP 'python-installer.exe'

    if (-not ([Net.ServicePointManager]::SecurityProtocol.HasFlag([Net.SecurityProtocolType]::Tls12))) {
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    }

    Write-Host "Downloading Python installer..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $pythonInstallerUrl -OutFile $installerPath

    Write-Host "Running Python installer (this may take a minute)..." -ForegroundColor Yellow
    $arguments = @(
        '/quiet',
        'InstallAllUsers=0',
        'PrependPath=1',
        'Include_test=0',
        'SimpleInstall=1'
    )
    $process = Start-Process -FilePath $installerPath -ArgumentList $arguments -Wait -PassThru
    Remove-Item $installerPath -Force

    if ($process.ExitCode -ne 0) {
        throw "Python installer failed with exit code $($process.ExitCode)."
    }

    $pythonPath = Get-PythonPath
    if (-not $pythonPath) {
        throw 'Python installation succeeded but python.exe was not found on PATH. Please restart the terminal and run the script again.'
    }

    Write-Host "Python installed at $pythonPath" -ForegroundColor Green
    return $pythonPath
}

function Ensure-Venv($pythonPath, $venvPath) {
    if (-not (Test-Path $venvPath)) {
        Write-Section "Creating virtual environment"
        & $pythonPath -m venv $venvPath
    }

    $venvPython = Join-Path $venvPath 'Scripts\\python.exe'
    if (-not (Test-Path $venvPython)) {
        throw "Virtual environment not found at $venvPython"
    }
    return $venvPython
}

function Install-Requirements($venvPython, $requirementsPath) {
    Write-Section "Installing Python dependencies"
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r $requirementsPath
}

function Load-EnvFile($envPath) {
    if (-not (Test-Path $envPath)) {
        Write-Warning "No .env file found at $envPath. Skipping environment variable registration."
        return @{}
    }

    $envVars = @{}
    $pattern = '^(?<key>[^#=]+?)\s*=\s*(?<value>.*)$'

    foreach ($line in Get-Content $envPath) {
        if ([string]::IsNullOrWhiteSpace($line) -or $line.TrimStart().StartsWith('#')) {
            continue
        }
        $match = [regex]::Match($line, $pattern)
        if (-not $match.Success) {
            Write-Warning "Skipping malformed line in .env: $line"
            continue
        }
        $key = $match.Groups['key'].Value.Trim()
        $value = $match.Groups['value'].Value.Trim()

        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        $envVars[$key] = $value
    }

    return $envVars
}

function Register-EnvironmentVariables($envVars, $projectRoot) {
    if ($envVars.Count -eq 0) {
        return
    }

    Write-Section "Registering environment variables"
    foreach ($entry in $envVars.GetEnumerator()) {
        $value = $entry.Value
        $expandedPath = Join-Path $projectRoot $value
        if (-not [string]::IsNullOrWhiteSpace($value) -and (Test-Path $expandedPath)) {
            $value = (Resolve-Path $expandedPath).Path
        }

        [System.Environment]::SetEnvironmentVariable($entry.Key, $value, 'Process')
        [System.Environment]::SetEnvironmentVariable($entry.Key, $value, 'User')
        try {
            setx $entry.Key $value | Out-Null
        } catch {
            Write-Warning "Failed to persist $($entry.Key) using setx: $_"
        }
        Write-Host "Set $($entry.Key)=$value" -ForegroundColor Green
    }
}

function Start-Server($venvPython, $projectRoot) {
    Write-Section "Starting FastAPI server"
    Push-Location $projectRoot
    try {
        & $venvPython -m uvicorn main:app --host 0.0.0.0 --port 8000
    } finally {
        Pop-Location
    }
}

Write-Section "Preparing environment"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$pythonPath = Ensure-Python
$venvPath = Join-Path $projectRoot '.venv'
$venvPython = Ensure-Venv -pythonPath $pythonPath -venvPath $venvPath
$requirementsPath = Join-Path $projectRoot 'requirements.txt'
Install-Requirements -venvPython $venvPython -requirementsPath $requirementsPath

$envPath = Join-Path $projectRoot '.env'
$envVars = Load-EnvFile -envPath $envPath
Register-EnvironmentVariables -envVars $envVars -projectRoot $projectRoot

Start-Server -venvPython $venvPython -projectRoot $projectRoot
