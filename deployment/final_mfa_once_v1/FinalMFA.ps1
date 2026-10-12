param(
 [Parameter(Mandatory=$true)][ValidateSet('Verify','Preflight','ToyTests','MonitorToy','StartFinal','Status','Export','Stop')][string]$Action,
 [switch]$ConfirmFinalTraining,[string]$OperatorName,[switch]$Background,[switch]$IncludeModel,[string]$PythonPath
)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Utility\Microsoft.PowerShell.Utility.psd1') -ErrorAction Stop
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Management\Microsoft.PowerShell.Management.psd1') -ErrorAction Stop
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if (-not $PythonPath) {$PythonPath=Join-Path $root '.venv-lab\Scripts\python.exe'}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {throw 'Existing pinned laboratory Python required; do not install or upgrade.'}
if ($ConfirmFinalTraining -and $Action -ne 'StartFinal') {throw 'Final confirmation applies only to StartFinal.'}
if ($IncludeModel -and $Action -ne 'Export') {throw 'IncludeModel applies only to Export.'}
if ($Action -eq 'StartFinal' -and (-not $ConfirmFinalTraining -or -not $OperatorName.Trim())) {throw 'Actual operator must explicitly supply -ConfirmFinalTraining and -OperatorName. No old R7/R8 approval can authorize this action.'}
$map=@{Verify='verify';Preflight='preflight';ToyTests='toy-tests';MonitorToy='monitor-toy';StartFinal='start-final';Status='status';Export='export';Stop='stop'}
$argv=@('-B','-X','utf8',(Join-Path $PSScriptRoot 'final_once.py'),$map[$Action])
if ($ConfirmFinalTraining) {$argv+=@('--confirm-final-training','--operator',$OperatorName)}
if ($IncludeModel) {$argv+='--include-model'}
$env:PYTHONDONTWRITEBYTECODE='1';$env:OPENBLAS_NUM_THREADS='6';$env:MKL_NUM_THREADS='6';$env:NUMEXPR_NUM_THREADS='6';$env:OMP_NUM_THREADS='1'
if ($Background) {
 if ($Action -notin @('StartFinal','MonitorToy')) {throw 'Background is only supported for one final job or synthetic handshake.'}
 & $PythonPath -B -X utf8 (Join-Path $PSScriptRoot 'final_once.py') verify
 if ($LASTEXITCODE -ne 0) {throw 'Tool verification failed; nothing launched.'}
 $logs=Join-Path $root ('lab_runs\final_mfa_once\launchers\'+[guid]::NewGuid().ToString('N'))
 New-Item -ItemType Directory -Path $logs | Out-Null
 $quoted=$argv | ForEach-Object {'"'+($_ -replace '"','\"')+'"'}
 $child=Start-Process -FilePath $PythonPath -ArgumentList $quoted -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs 'stdout.log') -RedirectStandardError (Join-Path $logs 'stderr.log')
 @{launcher_PID=$child.Id;action=$Action;logs=$logs;automatic_retry=$false;personal_computer_required_online=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logs 'launcher.json') -Encoding UTF8
 Write-Output "Started laboratory-local launcher PID $($child.Id). Query Status and inspect launcher logs. FAILED/STOP: preserve all records, no retry."
} else {
 & $PythonPath @argv
 if ($LASTEXITCODE -ne 0) {throw "Final-only $Action refused or failed; preserve logs, no automatic retry."}
}
