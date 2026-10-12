param(
 [Parameter(Mandatory=$true)][ValidateSet('Verify','Preflight','ToyTests','MonitorToy','Run','Status','Stop','Export')][string]$Action,
 [string]$Python='', [switch]$Background, [switch]$ConfirmFormalEvaluation
)
$ErrorActionPreference='Stop'
$env:PSModulePath=(Join-Path $PSHOME 'Modules')+';'+$env:PSModulePath
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if (-not $Python) { $Python=Join-Path $root '.venv-lab\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python)) { throw 'Pinned .venv-lab Python missing: STOP' }
$script=Join-Path $PSScriptRoot 'control.py'
if ($Action -eq 'Run' -and -not $ConfirmFormalEvaluation) {throw 'Explicit -ConfirmFormalEvaluation required: STOP'}
if ($Background) {
 if ($Action -notin @('Run','MonitorToy')) {throw 'Background only supports Run/MonitorToy'}
 $logRoot=Join-Path $root 'lab_runs\b5c_paired_r1\controller_logs'
 New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
 $token=[guid]::NewGuid().ToString('N')
 $argsList=@('-B','-X','utf8',('"'+$script+'"'),$Action)
 if($Action -eq 'Run'){$argsList+='--formal'}
 $process=Start-Process -FilePath $Python -ArgumentList $argsList -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logRoot ($token+'.stdout.log')) -RedirectStandardError (Join-Path $logRoot ($token+'.stderr.log')) -PassThru
 [pscustomobject]@{ControllerPID=$process.Id;Logs=$logRoot;AutomaticRetry=$false} | Format-List
} else {
 if($Action -eq 'Run'){& $Python -B -X utf8 $script $Action --formal}
 else {& $Python -B -X utf8 $script $Action}
 if($LASTEXITCODE -ne 0){throw ('B5C '+$Action+' failed: STOP; preserve logs; no retry')}
}
