param(
    [ValidateSet('host','fidelity','self-test','smoke')][string]$Mode = 'host',
    [ValidateRange(1,10)][int]$Threads = 1
)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$python = Join-Path $taskRoot '.venv-lab\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run Install-Environment.ps1 first.' }
& (Join-Path $PSScriptRoot 'Verify-Package.ps1')
$runName = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + $Mode + '_' + ([guid]::NewGuid().ToString('N').Substring(0,8))
$job = Join-Path (Join-Path $taskRoot 'lab_runs') $runName
New-Item -ItemType Directory -Path $job | Out-Null
$initial = @{state='STARTING';action=$Mode;threads=$Threads;PASS=$false;formal_training_authorized=$false} | ConvertTo-Json
$latest = @{job=('lab_runs/' + $runName);action=$Mode;time=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json
$encoding = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText((Join-Path $job 'status.json'),$initial,$encoding)
[IO.File]::WriteAllText((Join-Path (Join-Path $taskRoot 'lab_runs') 'latest.json'),$latest,$encoding)
$launcher = Join-Path $PSScriptRoot 'lab.py'
$arguments = '-B -X utf8 "' + $launcher + '" ' + $Mode + ' --threads ' + $Threads + ' --job "' + $job + '"'
try {
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $job 'supervisor.stdout.log') -RedirectStandardError (Join-Path $job 'supervisor.stderr.log')
} catch {
    $failed = @{state='STOPPED';PASS=$false;stop_reason='PROCESS_LAUNCH_FAILED';exception=$_.Exception.Message;formal_training_authorized=$false} | ConvertTo-Json
    [IO.File]::WriteAllText((Join-Path $job 'status.json'),$failed,$encoding)
    throw
}
Write-Output "Started $Mode deployment job. Launcher PID: $($process.Id)"
Write-Output "Logs: $job"
Write-Output 'Use Get-LabStatus.ps1 to check completion. No formal candidate training is supported.'
