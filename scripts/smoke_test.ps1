param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 8095,
    [string]$OutputDirectory = "data/smoke"
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$smokeTarget = Join-Path $root 'rust\target\smoke'

if (Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use. Choose another -Port value."
}

. "$PSScriptRoot\use-rust-env.ps1"
Push-Location (Join-Path $root 'rust')
try {
    cargo build -p sim-server --target-dir $smokeTarget
    if ($LASTEXITCODE -ne 0) { throw "Rust server build failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}

$server = $null
$previousPort = $env:SIM_SERVER_PORT
Push-Location $root
try {
    $env:SIM_SERVER_PORT = "$Port"
    $server = Start-Process -FilePath (Join-Path $smokeTarget 'debug\sim-server.exe') -WorkingDirectory (Join-Path $root 'rust') -WindowStyle Hidden -PassThru
    $healthUrl = "http://127.0.0.1:$Port/health"
    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        try {
            if ((Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2).status -eq 'ok') { $ready = $true; break }
        } catch { Start-Sleep -Milliseconds 250 }
    }
    if (-not $ready) { throw "Rust server did not become healthy at $healthUrl" }

    $rewardProfilePath = Join-Path $root 'configs\rewards\exploration-v1.json'
    $rewardProfile = Get-Content -Raw $rewardProfilePath | ConvertFrom-Json
    $runOutput = & python -m embodied_ai.cli run --scenario survival_room --seed 42 --provider scripted --reward-config $rewardProfilePath --server-url "http://127.0.0.1:$Port" --output $OutputDirectory
    if ($LASTEXITCODE -ne 0) { throw "Scripted policy run failed with exit code $LASTEXITCODE" }
    $run = ($runOutput | Select-Object -Last 1 | ConvertFrom-Json)
    if ($run.outcome -ne 'escaped') { throw "Expected the scripted run to escape, got $($run.outcome)" }

    $replay = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/runs/$($run.run_id)/replay" -TimeoutSec 5
    $events = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/runs/$($run.run_id)/events" -TimeoutSec 5
    if ($replay.snapshot.terminal_reason -ne 'escaped') { throw 'Persisted replay is not terminally escaped.' }
    if ($replay.replay_version -ne 1) { throw 'Persisted replay did not declare replay version 1.' }
    if ($replay.timeline.Count -ne $replay.observations.Count) { throw 'Replay timeline and filtered observations are not frame-aligned.' }
    if ($replay.decisions.Count -ne $run.steps) { throw 'Replay decisions do not match the number of executed provider steps.' }
    if ($events.Count -lt 1) { throw 'The authoritative server did not persist any events.' }
    if ($replay.control.provider_steps -ne $run.steps) { throw 'Provider control steps do not match the trajectory.' }
    if ($replay.control.manual_steps -ne 0) { throw 'Provider-only run incorrectly recorded manual steps.' }
    if ($replay.control.simulation_latency_us.Count -ne $run.steps) { throw 'Missing per-step simulation latency.' }
    if ($replay.control.provider_control_ms -le 0) { throw 'Provider wall-clock control time was not recorded.' }
    foreach ($field in $rewardProfile.PSObject.Properties) {
        if ($replay.reward_config.($field.Name) -ne $field.Value) {
            throw "Persisted replay reward profile does not match $($field.Name)."
        }
    }
    $trajectoryRecord = Get-Content $run.jsonl | Select-Object -Last 1 | ConvertFrom-Json
    if ($trajectoryRecord.reward_breakdown -eq $null -or $trajectoryRecord.simulation_time -eq $null) {
        throw 'Trajectory did not retain authoritative reward components and simulation time.'
    }
    foreach ($field in $rewardProfile.PSObject.Properties) {
        if ($trajectoryRecord.reward_config.($field.Name) -ne $field.Value) {
            throw "Trajectory reward profile does not match $($field.Name)."
        }
    }
    $manifest = Get-Content -Raw $run.experiment_manifest | ConvertFrom-Json
    foreach ($field in $rewardProfile.PSObject.Properties) {
        if ($manifest.reward_config.($field.Name) -ne $field.Value) {
            throw "Experiment manifest reward profile does not match $($field.Name)."
        }
    }

    $frozenControl = $replay.control | ConvertTo-Json -Compress
    $null = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/api/runs/$($run.run_id)/abort" -TimeoutSec 5
    $afterAbort = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/runs/$($run.run_id)/replay" -TimeoutSec 5
    if (($afterAbort.control | ConvertTo-Json -Compress) -ne $frozenControl) { throw 'Terminal control clocks did not freeze.' }
    if ($afterAbort.timeline.Count -ne $replay.timeline.Count -or $afterAbort.snapshot.terminal_reason -ne 'escaped') { throw 'Repeated terminal control changed the replay.' }

    [pscustomobject]@{
        run_id = $run.run_id
        outcome = $run.outcome
        steps = $run.steps
        events = $events.Count
        replay_frames = $replay.timeline.Count
        decisions = $replay.decisions.Count
        output = $run.jsonl
    } | ConvertTo-Json -Compress
} finally {
    if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id }
    $env:SIM_SERVER_PORT = $previousPort
    Pop-Location
}
