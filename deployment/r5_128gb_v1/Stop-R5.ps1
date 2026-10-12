param()
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
& (Join-Path $taskRoot '.venv-lab\Scripts\python.exe') -B -X utf8 (Join-Path $PSScriptRoot 'driver.py') stop
if ($LASTEXITCODE -ne 0) { throw 'STOP request failed. Inspect logs and stop the laboratory task through Windows if necessary.' }
