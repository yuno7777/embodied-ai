param(
    [Parameter(Mandatory = $false)]
    [string]$Path = "scenarios\survival_room\scenario.rust.json"
)

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$candidate = if ([IO.Path]::IsPathRooted($Path)) { $Path } else { Join-Path $workspace $Path }
$scenarioPath = (Resolve-Path -LiteralPath $candidate).Path
. (Join-Path $PSScriptRoot 'use-rust-env.ps1')
Push-Location (Join-Path $workspace 'rust')
try {
    cargo run -p sim-server -- --validate-scenario $scenarioPath
    if ($LASTEXITCODE -ne 0) { throw "Scenario validation failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
