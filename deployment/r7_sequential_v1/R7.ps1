param([Parameter(Mandatory=$true)][ValidateSet('Verify','ToyTests','AuthorizationBindings','ImportApproval','Preflight','Start','Status','Stop','Export','ToyQueue')][string]$Action,
      [switch]$Background,[string]$PythonPath,[string]$ApprovalPath,[string]$InstructionPath,[string]$ToyDirectory)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Utility\Microsoft.PowerShell.Utility.psd1') -ErrorAction Stop
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Management\Microsoft.PowerShell.Management.psd1') -ErrorAction Stop
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if (-not $PythonPath) {$PythonPath=Join-Path $root '.venv-lab\Scripts\python.exe'}
if (-not (Test-Path -LiteralPath $PythonPath)) {throw 'Frozen laboratory Python missing.'}
$map=@{Verify='verify';ToyTests='technical-tests';AuthorizationBindings='authorization-bindings';ImportApproval='import-approval';Preflight='preflight';Start='start';Status='status';Stop='stop';Export='export'}
$argsList=@('-B','-X','utf8',(Join-Path $PSScriptRoot 'r7.py'))
if ($Action -eq 'ToyQueue') {
    if (-not $ToyDirectory) {throw 'ToyDirectory is required.'}
    $argsList=@('-B','-X','utf8',(Join-Path $PSScriptRoot 'toy_queue.py'),'--folder',$ToyDirectory)
} else {$argsList+=@($map[$Action])}
if ($Action -eq 'ImportApproval') {
    if (-not $ApprovalPath -or -not $InstructionPath) {throw 'Independently provided mentor record and instruction required.'}
    $argsList+=@('--approval',[IO.Path]::GetFullPath($ApprovalPath),'--instruction',[IO.Path]::GetFullPath($InstructionPath))
}
$env:PYTHONDONTWRITEBYTECODE='1';$env:OPENBLAS_NUM_THREADS='6';$env:MKL_NUM_THREADS='6';$env:NUMEXPR_NUM_THREADS='6';$env:OMP_NUM_THREADS='1'
$runRoot=Join-Path $root 'lab_runs\r7_sequential'
if ($Background) {
    if ($Action -notin @('Start','ToyQueue')) {throw 'Background is allowed only for Start or synthetic ToyQueue.'}
    if ($Action -eq 'Start') {
        & $PythonPath -B -X utf8 (Join-Path $PSScriptRoot 'r7.py') preflight
        if ($LASTEXITCODE -ne 0) {throw 'Formal queue preflight/authorization refused; nothing started.'}
    }
    $logs=Join-Path $runRoot ('launchers\'+[guid]::NewGuid().ToString('N'));New-Item -ItemType Directory -Path $logs | Out-Null
    $quoted=$argsList | ForEach-Object {'"'+($_ -replace '"','\"')+'"'}
    $child=Start-Process -FilePath $PythonPath -ArgumentList $quoted -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs 'stdout.log') -RedirectStandardError (Join-Path $logs 'stderr.log')
    @{launcher_PID=$child.Id;action=$Action;logs=$logs;lab_local_independent=$true;personal_computer_required_online=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logs 'launcher.json') -Encoding UTF8
    Write-Output "R7 laboratory-local launcher PID $($child.Id); logs $logs. Query Status for actual controller/worker PID."
} else {
    & $PythonPath @argsList
    if ($LASTEXITCODE -ne 0) {throw "R7 $Action FAILED/STOP/refused. Preserve logs; no automatic retry."}
}
