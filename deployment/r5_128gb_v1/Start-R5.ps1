param([Parameter(Mandatory=$true)][ValidateSet('preflight','cpu','smoke12000','smoke15000')][string]$Stage)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$python = Join-Path $taskRoot '.venv-lab\Scripts\python.exe'
& (Join-Path $PSScriptRoot 'Verify-R5.ps1')
$runRoot = Join-Path $taskRoot 'lab_runs\r5_128gb'
$runName = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + $Stage + '_' + ([guid]::NewGuid().ToString('N').Substring(0,8))
$job = Join-Path $runRoot $runName
New-Item -ItemType Directory -Path $job -Force | Out-Null
$utf8 = New-Object Text.UTF8Encoding($false)
$initial = @{state='RUNNING';PASS=$false;stage=$Stage;note='launcher started; awaiting driver gates';formal_training_authorized=$false} | ConvertTo-Json
[IO.File]::WriteAllText((Join-Path $job 'status.json'),$initial,$utf8)
$relativeJob = 'lab_runs/r5_128gb/' + $runName
[IO.File]::WriteAllText((Join-Path $runRoot 'latest_launch.json'),(@{job=$relativeJob;stage=$Stage;state='RUNNING'} | ConvertTo-Json),$utf8)
$arguments = '-B -X utf8 "' + (Join-Path $PSScriptRoot 'driver.py') + '" ' + $Stage + ' --job "' + $job + '"'
try {
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $job 'driver.stdout.log') -RedirectStandardError (Join-Path $job 'driver.stderr.log')
} catch {
    $failed = @{state='FAILED';PASS=$false;exception=$_.Exception.Message} | ConvertTo-Json
    [IO.File]::WriteAllText((Join-Path $job 'status.json'),$failed,$utf8)
    throw
}
Write-Output "RUNNING request: $Stage; driver launcher PID=$($process.Id)"
Write-Output "Logs: $job"
Write-Output 'Check Get-R5Status.ps1. Continue only if COMPLETE and PASS=true. Stages never chain automatically.'
