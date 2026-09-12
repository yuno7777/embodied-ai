. "$PSScriptRoot\use-rust-env.ps1"
Set-Location "$PSScriptRoot\..\rust"
cargo run -p sim-server
