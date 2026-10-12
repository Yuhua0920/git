param()
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
& (Join-Path $taskRoot '.venv-lab\Scripts\python.exe') -B -X utf8 (Join-Path $PSScriptRoot 'driver.py') export
if ($LASTEXITCODE -ne 0) { throw 'Result export failed. Preserve all original logs.' }
