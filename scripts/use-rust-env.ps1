$vsRoot = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC'
$sdkRoot = 'C:\Program Files (x86)\Windows Kits\10\Lib'
$msvc = Get-ChildItem $vsRoot -Directory | Sort-Object Name -Descending | Select-Object -First 1
$sdk = Get-ChildItem $sdkRoot -Directory | Sort-Object Name -Descending | Select-Object -First 1
if (-not $msvc -or -not $sdk) { throw 'MSVC Build Tools and Windows SDK are required for Rust builds.' }
$env:PATH = "$($msvc.FullName)\bin\Hostx64\x64;$env:PATH"
$env:PATH = "C:\Users\Abhi\.cargo\bin;$env:PATH"
$env:LIB = "$($sdk.FullName)\um\x64;$($sdk.FullName)\ucrt\x64;$($msvc.FullName)\lib\x64"
