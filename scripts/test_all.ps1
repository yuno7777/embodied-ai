$ErrorActionPreference = 'Stop'
Set-Location "$PSScriptRoot\.."

function Invoke-Checked([scriptblock]$Command, [string]$Name) {
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

Invoke-Checked { python -m pytest -q } 'Python tests'
Invoke-Checked { npm run lint --prefix frontend\observer } 'Frontend lint'
Invoke-Checked { npm exec --prefix frontend\observer -- tsc --noEmit -p frontend/observer/tsconfig.json } 'TypeScript check'
Invoke-Checked { npm run test --prefix frontend\observer } 'Frontend tests'
Invoke-Checked { npm run build --prefix frontend\observer } 'Frontend build'
. "$PSScriptRoot\use-rust-env.ps1"
Push-Location rust
try {
    Invoke-Checked { cargo fmt --check } 'Rust format check'
    Invoke-Checked { cargo test --workspace } 'Rust tests'
    Invoke-Checked { cargo clippy --all-targets --all-features -- -D warnings } 'Rust lint'
} finally {
    Pop-Location
}
