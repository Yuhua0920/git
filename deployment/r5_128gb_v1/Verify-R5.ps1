param()
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$python = Join-Path $taskRoot '.venv-lab\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Existing .venv-lab is missing. STOP; do not reinstall or upgrade automatically.' }
& $python -B -X utf8 (Join-Path $PSScriptRoot 'driver.py') verify
if ($LASTEXITCODE -ne 0) { throw 'R5/base integrity verification failed: STOP' }
