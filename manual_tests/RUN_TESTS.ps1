# RUN_TESTS.ps1 — Start TSL Translator and open browser
# Usage: Right-click → Run with PowerShell, or: powershell -ExecutionPolicy Bypass -File RUN_TESTS.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path
$BackendPort = 8000
$FrontendPort = 3000

function Write-Status($msg) { Write-Host "[TSL] $msg" -ForegroundColor Cyan }
function Write-OK($msg)     { Write-Host "[TSL] $msg" -ForegroundColor Green }
function Write-Fail($msg)   { Write-Host "[TSL] $msg" -ForegroundColor Red }

# Graceful Ctrl+C handling
$script:processes = @()
Register-EngineEvent PowerShell.Exiting -Action {
    foreach ($p in $script:processes) {
        if (!$p.HasExited) { $p.Kill() }
    }
}

try {
    # --- Check Python ---
    Write-Status "Checking Python..."
    try {
        $pyVersion = & python --version 2>&1
        Write-OK "Found: $pyVersion"
    } catch {
        Write-Fail "Python not found. Install Python and add it to PATH."
        exit 1
    }

    # --- Check .env ---
    $envFile = Join-Path $ProjectRoot ".env"
    if (!(Test-Path $envFile)) {
        Write-Fail ".env file not found at $envFile"
        exit 1
    }
    Write-OK ".env found"

    # --- Check models ---
    $modelsDir = Join-Path $ProjectRoot "models"
    if (!(Test-Path $modelsDir)) {
        Write-Fail "models/ directory not found at $modelsDir"
        exit 1
    }
    $modelFiles = Get-ChildItem -Path $modelsDir -Recurse -File -ErrorAction SilentlyContinue
    if ($modelFiles.Count -eq 0) {
        Write-Fail "No model files found in $modelsDir"
        exit 1
    }
    Write-OK "Found $($modelFiles.Count) file(s) in models/"

    # --- Check start.py ---
    $startPy = Join-Path $ProjectRoot "start.py"
    if (!(Test-Path $startPy)) {
        Write-Fail "start.py not found at $startPy"
        exit 1
    }
    Write-OK "start.py found"

    # --- Start the server ---
    Write-Status "Starting TSL Translator (python start.py)..."
    $startProcess = Start-Process -FilePath "python" -ArgumentList "start.py" `
        -WorkingDirectory $ProjectRoot -PassThru -NoNewWindow
    $script:processes += $startProcess

    # --- Wait for port 8000 ---
    Write-Status "Waiting for port $BackendPort..."
    $timeout = 60
    $elapsed = 0
    while ($elapsed -lt $timeout) {
        try {
            $tcp = New-Object System.Net.Sockets.TcpClient
            $tcp.Connect("127.0.0.1", $BackendPort)
            $tcp.Close()
            Write-OK "Port $BackendPort is open"
            break
        } catch {
            Start-Sleep -Seconds 1
            $elapsed++
            if ($elapsed % 10 -eq 0) {
                Write-Status "Still waiting... ($elapsed s)"
            }
        }
    }
    if ($elapsed -ge $timeout) {
        Write-Fail "Port $BackendPort did not open within $timeout seconds"
        exit 1
    }

    # --- Open browser ---
    Write-Status "Opening browser at http://localhost:$FrontendPort"
    Start-Process "http://localhost:$FrontendPort"
    Write-OK "TSL Translator is running. Press Ctrl+C to stop."

    # --- Keep alive until Ctrl+C ---
    while (!$startProcess.HasExited) {
        Start-Sleep -Seconds 2
    }
    Write-Fail "start.py exited unexpectedly"
} finally {
    # Cleanup: kill all spawned processes
    foreach ($p in $script:processes) {
        if (!$p.HasExited) {
            $p.Kill()
            Write-Status "Killed process $($p.Id)"
        }
    }
    Write-Status "Cleaned up. Goodbye."
}
