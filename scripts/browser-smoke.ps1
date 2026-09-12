param(
    [string]$ObserverUrl = 'http://localhost:3000',
    [string]$RustUrl = 'http://127.0.0.1:8080',
    [string]$BrowserPath = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $BrowserPath)) { throw 'Pass -BrowserPath pointing to an installed Chromium browser.' }
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$artifacts = Join-Path $workspace 'data/browser-smoke'
New-Item -ItemType Directory -Force -Path $artifacts | Out-Null
$session = 'embodied-smoke-' + [guid]::NewGuid().ToString('N')

function Invoke-Browser([string[]]$Arguments) {
    Write-Host "Browser: $($Arguments -join ' ')"
    $output = & npm exec --yes --package=agent-browser@0.37.1 -- agent-browser --session $session --json @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Browser command failed: $($Arguments -join ' ')`n$output" }
    $result = ($output -join "`n") | ConvertFrom-Json
    if (-not $result.success) { throw "Browser command failed: $($result.error)" }
    return $result.data
}
function Wait-Page([string]$Expression) {
    $null = Invoke-Browser @('wait', '--fn', $Expression)
}

try {
    $before = @((Invoke-RestMethod "$RustUrl/api/runs" -TimeoutSec 10) | ForEach-Object { $_.run_id })
    & npm exec --yes --package=agent-browser@0.37.1 -- agent-browser --session $session --executable-path $BrowserPath open $ObserverUrl
    if ($LASTEXITCODE -ne 0) { throw 'Browser failed to open.' }
    Wait-Page 'document.body.innerText.includes("RESEARCHER VIEW") && document.body.innerText.includes("online")'
    $null = Invoke-Browser @('snapshot', '-i')
    $null = Invoke-Browser @('find', 'label', 'Max steps', 'fill', '2')
    $null = Invoke-Browser @('find', 'role', 'button', 'click', '--name', 'Start manual', '--exact')
    Wait-Page 'document.body.innerText.includes("Manual run started.") && document.body.innerText.includes("LIVE LINK: connected")'
    $created = @((Invoke-RestMethod "$RustUrl/api/runs" -TimeoutSec 10) | Where-Object { $_.run_id -notin $before })
    if ($created.Count -ne 1) { throw 'Could not identify exactly one new smoke-test run.' }
    $runId = $created[0].run_id
    $null = Invoke-Browser @('find', 'role', 'button', 'click', '--name', 'Wait', '--exact')
    Wait-Page 'document.body.innerText.includes("STEP 1")'
    $null = Invoke-Browser @('find', 'role', 'button', 'click', '--name', 'Wait', '--exact')
    Wait-Page 'document.body.innerText.includes("STEP 2")'
    $null = Invoke-Browser @('select', 'select[aria-label="Event type"]', 'ActionCompleted')
    Wait-Page 'document.body.innerText.includes("You wait briefly.")'
    $null = Invoke-Browser @('find', 'label', 'Filter events', 'fill', 'impossible-smoke-text')
    Wait-Page 'document.body.innerText.includes("No matching events.")'
    $null = Invoke-Browser @('screenshot', (Join-Path $artifacts 'manual.png'))
    $null = Invoke-Browser @('open', "$ObserverUrl/replay/$runId")
    Wait-Page 'document.body.innerText.includes("Loaded persisted replay.")'
    Wait-Page 'Array.from(document.querySelectorAll("button")).filter(b => ["Abort", "Wait"].includes(b.textContent)).every(b => b.disabled)'
    $null = Invoke-Browser @('find', 'role', 'button', 'click', '--name', '|◀', '--exact')
    Wait-Page 'document.body.innerText.includes("STEP 0")'
    $null = Invoke-Browser @('find', 'role', 'button', 'click', '--name', 'Light mode', '--exact')
    Wait-Page 'document.querySelector("main.light") !== null'
    $null = Invoke-Browser @('set', 'viewport', '390', '844')
    Wait-Page 'document.documentElement.scrollWidth <= window.innerWidth'
    $null = Invoke-Browser @('screenshot', (Join-Path $artifacts 'mobile-replay.png'))
    Wait-Page '!document.querySelector("[data-nextjs-dialog], .vite-error-overlay")'
    $errors = Invoke-Browser @('errors')
    if ($errors.errors.Count -gt 0) { throw "Browser errors: $($errors | ConvertTo-Json -Depth 5)" }
    [pscustomobject]@{ passed = $true; run_id = $runId; checks = 'manual steps, live updates, event filters, replay navigation, read-only controls, theme, mobile overflow, page errors'; artifacts = $artifacts } | ConvertTo-Json -Compress
} finally {
    $null = Invoke-Browser @('close')
}
