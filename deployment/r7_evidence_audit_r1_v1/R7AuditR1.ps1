param([Parameter(Mandatory=$true)][ValidateSet('Verify','ToyTests','MonitorToy','Audit','Status','StopAudit','Export')][string]$Action,
      [switch]$Background,[string]$PythonPath,[string]$JobPath,[switch]$IncludeAllModels,[switch]$CheckFailedAttempt)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Utility\Microsoft.PowerShell.Utility.psd1') -ErrorAction Stop
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Management\Microsoft.PowerShell.Management.psd1') -ErrorAction Stop
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if (-not $PythonPath) {$PythonPath=Join-Path $root '.venv-lab\Scripts\python.exe'}
if (-not (Test-Path -LiteralPath $PythonPath)) {throw 'Existing pinned laboratory Python missing; no installation permitted.'}
$map=@{Verify='verify';ToyTests='toy-tests';MonitorToy='monitor-toy';Audit='audit';Status='status';StopAudit='stop-audit';Export='export'}
$argsList=@('-B','-X','utf8',(Join-Path $PSScriptRoot 'audit_cli.py'),$map[$Action])
if ($CheckFailedAttempt) {
    if ($Action -ne 'Verify') {throw 'CheckFailedAttempt is for Verify only.'}
    $argsList+='--check-failed-attempt'
}
if ($JobPath) {$argsList+=@('--job',[IO.Path]::GetFullPath($JobPath))}
if ($IncludeAllModels) {
    if ($Action -ne 'Export') {throw 'IncludeAllModels is an export option only.'}
    $argsList+='--include-all-models'
}
$env:PYTHONDONTWRITEBYTECODE='1';$env:OPENBLAS_NUM_THREADS='6';$env:MKL_NUM_THREADS='6';$env:NUMEXPR_NUM_THREADS='6';$env:OMP_NUM_THREADS='1'
if ($Background) {
    if ($Action -notin @('Audit','MonitorToy')) {throw 'Background is allowed only for audit or synthetic monitor.'}
    & $PythonPath -B -X utf8 (Join-Path $PSScriptRoot 'audit_cli.py') verify
    if ($LASTEXITCODE -ne 0) {throw 'Tool verification failed: STOP.'}
    $logs=Join-Path $root ('lab_runs\r7_evidence_audit_r1\launchers\'+[guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $logs | Out-Null
    $quoted=$argsList | ForEach-Object {'"'+($_ -replace '"','\"')+'"'}
    $child=Start-Process -FilePath $PythonPath -ArgumentList $quoted -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs 'stdout.log') -RedirectStandardError (Join-Path $logs 'stderr.log')
    @{launcher_PID=$child.Id;action=$Action;logs=$logs;no_training=$true;personal_computer_required_online=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logs 'launcher.json') -Encoding UTF8
    Write-Output "Read-only audit launcher PID $($child.Id). Query Status; preserve logs on FAILED/STOP. No automatic retry."
} else {
    & $PythonPath @argsList
    if ($LASTEXITCODE -ne 0) {throw "R7Audit $Action FAILED/STOP/refused; preserve logs, no automatic retry."}
}
