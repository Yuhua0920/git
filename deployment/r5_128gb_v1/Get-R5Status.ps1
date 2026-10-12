param()
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$python = Join-Path $taskRoot '.venv-lab\Scripts\python.exe'
& $python -B -X utf8 (Join-Path $PSScriptRoot 'driver.py') status
if ($LASTEXITCODE -ne 0) { throw 'Status query failed.' }
