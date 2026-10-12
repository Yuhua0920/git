param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('Verify','Preflight','StartCandidate','ValidateCandidate','CompleteSelection','StartFinal','Status','Stop','Export','ToyTests','MonitorToy')]
    [string]$Action,
    [ValidateSet('K1_r4','K1_r8','K1_r16','K2_r4','K2_r8','K2_r16','K4_r4','K4_r8','K4_r16')]
    [string]$Candidate,
    [switch]$Background,
    [switch]$IncludeFinalModel,
    [string]$PythonPath,
    [switch]$PersonalMetadataOnly
)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Utility\Microsoft.PowerShell.Utility.psd1') -ErrorAction Stop
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Management\Microsoft.PowerShell.Management.psd1') -ErrorAction Stop
$projectRoot=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not $PythonPath) { $PythonPath=Join-Path $projectRoot '.venv-lab\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw 'Pinned .venv-lab Python missing; do not install/upgrade packages.' }
$runRoot=Join-Path $projectRoot 'lab_runs\r6_formal'
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
$env:PYTHONDONTWRITEBYTECODE='1'
$env:TEMP=Join-Path $runRoot 'temporary'
$env:TMP=$env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
if ($Action -eq 'ToyTests') {
    & $PythonPath -B -X utf8 (Join-Path $PSScriptRoot 'toy_check.py')
    if ($LASTEXITCODE -ne 0) { throw "ToyTests failed with exit $LASTEXITCODE" }; return
}
$map=@{Verify='verify';Preflight='preflight';StartCandidate='train';ValidateCandidate='validate';CompleteSelection='select';StartFinal='final';Status='status';Stop='stop';Export='export'}
$argsList=@('-B','-X','utf8',(Join-Path $PSScriptRoot 'runner.py'),$map[$Action])
if ($Action -eq 'MonitorToy') { $argsList=@('-B','-X','utf8',(Join-Path $PSScriptRoot 'monitor_toy.py')) }
if ($Action -in @('StartCandidate','ValidateCandidate')) {
    if (-not $Candidate) { throw 'Specify -Candidate from frozen nine-candidate registry.' }
    $argsList+=@('--candidate',$Candidate)
}
if ($IncludeFinalModel) { if ($Action -ne 'Export') { throw 'IncludeFinalModel only applies to Export.' }; $argsList+='--include-final-model' }
if ($PersonalMetadataOnly) { if ($Action -notin @('Verify','Preflight')) { throw 'PersonalMetadataOnly is metadata-only.' }; $argsList+='--personal-metadata-only' }
if ($Background) {
    if ($Action -notin @('StartCandidate','ValidateCandidate','StartFinal','MonitorToy')) { throw 'Background is supported only for monitored jobs.' }
    # Refuse before background dispatch: no hidden formal attempt while pending.
    if ($Action -ne 'MonitorToy') {
        & $PythonPath -B -X utf8 -c "import sys; sys.path.insert(0,sys.argv[1]); import common; common.scope(); common.approval()" $PSScriptRoot
        if ($LASTEXITCODE -ne 0) { throw 'Mentor approval gate refused; nothing launched.' }
    }
    $logRoot=Join-Path $runRoot ('launchers\'+(Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
    New-Item -ItemType Directory -Path $logRoot | Out-Null
    # Explicit Windows quoting; neither project paths nor arguments are evaluated as commands.
    $quotedArgs=$argsList | ForEach-Object { '"'+($_ -replace '"','\"')+'"' }
    $child=Start-Process -FilePath $PythonPath -ArgumentList $quotedArgs -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logRoot 'stdout.log') -RedirectStandardError (Join-Path $logRoot 'stderr.log')
    @{supervisor_launcher_PID=$child.Id;action=$Action;candidate=$Candidate;logs=$logRoot;user_computer_required_online=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logRoot 'launcher.json') -Encoding UTF8
    Write-Output "Started local laboratory supervisor launcher PID $($child.Id); logs: $logRoot"
} else {
    & $PythonPath @argsList
    if ($LASTEXITCODE -ne 0) { throw "R6 $Action failed/refused with exit $LASTEXITCODE; inspect saved logs. No retry." }
}
