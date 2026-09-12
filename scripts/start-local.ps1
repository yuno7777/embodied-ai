param(
    [ValidateRange(1024, 65535)][int]$RustPort = 8080,
    [ValidateRange(1024, 65535)][int]$AgentPort = 8090,
    [ValidateSet(3000)][int]$ObserverPort = 3000,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (@($RustPort, $AgentPort, $ObserverPort | Select-Object -Unique).Count -ne 3) {
    throw 'Each service requires a distinct port.'
}
foreach ($servicePort in @($RustPort, $AgentPort, $ObserverPort)) {
    if (Get-NetTCPConnection -State Listen -LocalPort $servicePort -ErrorAction SilentlyContinue) {
        throw "Port $servicePort is already in use. Stop the existing service before starting the local stack."
    }
}
$logDirectory = Join-Path $workspace 'data\local-processes'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

. (Join-Path $PSScriptRoot 'use-rust-env.ps1')
Push-Location (Join-Path $workspace 'rust')
try {
    cargo build -p sim-server
    if ($LASTEXITCODE -ne 0) { throw 'Rust server build failed.' }
} finally { Pop-Location }
$env:SIM_SERVER_PORT = $RustPort
$env:SIM_SERVER_URL = "http://127.0.0.1:$RustPort"
$env:AGENT_SERVICE_PORT = $AgentPort
$env:NEXT_PUBLIC_SIM_SERVER_URL = $env:SIM_SERVER_URL
$env:NEXT_PUBLIC_SIM_WS_URL = "ws://127.0.0.1:$RustPort"
$env:NEXT_PUBLIC_AGENT_SERVICE_URL = "http://127.0.0.1:$AgentPort"
$env:PORT = $ObserverPort

function Start-LocalProcess([string]$Name, [string]$FilePath, [string[]]$Arguments, [string]$WorkingDirectory) {
    $stdout = Join-Path $logDirectory "$Name.out.log"
    $stderr = Join-Path $logDirectory "$Name.err.log"
    $launch = @{ FilePath = $FilePath; WorkingDirectory = $WorkingDirectory; WindowStyle = 'Hidden'; RedirectStandardOutput = $stdout; RedirectStandardError = $stderr; PassThru = $true }
    if ($Arguments.Count -gt 0) { $launch.ArgumentList = $Arguments }
    $process = Start-Process @launch
    [pscustomobject]@{ name = $Name; pid = $process.Id; stdout = $stdout; stderr = $stderr }
}

$processes = @(
    Start-LocalProcess 'rust' (Join-Path $workspace 'rust\target\debug\sim-server.exe') @() (Join-Path $workspace 'rust')
    Start-LocalProcess 'agent' 'python' @('-m', 'embodied_ai.agent_service', '--server-url', $env:SIM_SERVER_URL, '--port', "$AgentPort") $workspace
    Start-LocalProcess 'observer' 'node' @('node_modules/next/dist/bin/next', 'dev', '--port', "$ObserverPort") (Join-Path $workspace 'frontend\observer')
)

$manifest = Join-Path $logDirectory 'processes.json'
$processes | ConvertTo-Json | Set-Content -Encoding utf8 $manifest

$deadline = (Get-Date).AddSeconds(45)
$services = @(
    @{ name = 'Rust'; url = "$($env:SIM_SERVER_URL)/health" },
    @{ name = 'Agent'; url = "http://127.0.0.1:$AgentPort/health" },
    @{ name = 'Observer'; url = "http://127.0.0.1:$ObserverPort" }
)
do {
    $ready = @($services | Where-Object {
        try { (Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 $_.url).StatusCode -eq 200 } catch { $false }
    }).Count
    if ($ready -eq $services.Count) { break }
    Start-Sleep -Milliseconds 500
} while ((Get-Date) -lt $deadline)

if ($ready -ne $services.Count) {
    Write-Error "Only $ready/$($services.Count) services became healthy. Inspect $logDirectory"
}

Write-Host "All local processes are healthy. PIDs: $manifest"
Write-Host "Observer: http://127.0.0.1:$ObserverPort"
if (-not $NoBrowser) { Start-Process "http://127.0.0.1:$ObserverPort" }
