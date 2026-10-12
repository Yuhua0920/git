param(
 [Parameter(Mandatory=$true)][ValidateSet('Verify','ConcurrencyAudit','LabCheck','Rehearsal','Preflight','ToyTests','MonitorToy','Run','Status','Stop','Export')][string]$Action,
 [switch]$PersonalSelfTest, [string]$Python='', [string]$R1AuditJob='', [switch]$Background, [switch]$ConfirmFormalEvaluation
)
$ErrorActionPreference='Stop'
$env:PSModulePath=(Join-Path $PSHOME 'Modules')+';'+$env:PSModulePath
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if([string]::IsNullOrWhiteSpace($Python)){$Python=Join-Path $root '.venv-lab\Scripts\python.exe'}
if(-not(Test-Path -LiteralPath $Python)){throw 'Existing pinned .venv-lab Python required: STOP'}
$script=Join-Path $PSScriptRoot 'control.py'
if($PersonalSelfTest -and $Action -ne 'Rehearsal'){throw 'PersonalSelfTest supports synthetic Rehearsal only'}
if($Action -eq 'Run' -and -not $ConfirmFormalEvaluation){throw 'R1 must PASS and mentor must review the R2 boundary before explicit operator launch: STOP'}
$argsList=@('-B','-X','utf8',('"'+$script+'"'),$Action)
if($Action -eq 'Run'){$argsList+='--formal'}
if($PersonalSelfTest){$argsList+='--personal-selftest'}
if($R1AuditJob){
 if($R1AuditJob.Contains('"')){throw 'Invalid R1 job path'}
 $argsList+=@('--r1-audit-job',('"'+$R1AuditJob+'"'))
}
if($Background){
 if($Action -notin @('Run','MonitorToy')){throw 'Background supports Run/MonitorToy only'}
 $logs=Join-Path $root 'lab_runs\b5c_paired_r2\controller_logs'
 New-Item -ItemType Directory -Path $logs -Force|Out-Null
 $token=[guid]::NewGuid().ToString('N')
 $p=Start-Process -FilePath $Python -ArgumentList $argsList -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logs ($token+'.stdout.log')) -RedirectStandardError (Join-Path $logs ($token+'.stderr.log')) -PassThru
 [pscustomobject]@{ControllerPID=$p.Id;Logs=$logs;AutomaticRetry=$false}|Format-List
}else{
 $direct=@('-B','-X','utf8',$script,$Action)
 if($Action -eq 'Run'){$direct+='--formal'}
 if($PersonalSelfTest){$direct+='--personal-selftest'}
 if($R1AuditJob){$direct+=@('--r1-audit-job',$R1AuditJob)}
 & $Python @direct
 if($LASTEXITCODE -ne 0){throw ('R2 '+$Action+' failed: preserve all records; no automatic retry')}
}
