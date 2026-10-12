param([Parameter(Mandatory=$true)][ValidateSet('Verify','Recover','Status','Stop','Export','RenderFigures')][string]$Action,[string]$Python='', [switch]$Background)
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
function Plain([string]$path){
 $p=[IO.Path]::GetFullPath($path)
 while($p){
  if((Test-Path -LiteralPath $p) -and ((Get-Item -LiteralPath $p -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)){throw ('Reparse path rejected: '+$p)}
  $parent=[IO.Directory]::GetParent($p);if($null -eq $parent){break};$p=$parent.FullName
 }
}
Plain $root
if([string]::IsNullOrWhiteSpace($Python)){$Python=Join-Path $root '.venv-lab\Scripts\python.exe'}
if(-not(Test-Path -LiteralPath $Python -PathType Leaf)){throw 'Existing pinned LAB Python required: STOP'}
$env:PYTHONDONTWRITEBYTECODE='1'
foreach($name in @('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')){[Environment]::SetEnvironmentVariable($name,'6','Process')}
$env:OMP_NUM_THREADS='1'
$script=Join-Path $PSScriptRoot 'control.py'
if($Background){
 if($Action -ne 'Recover'){throw 'Background supports Recover only'}
 $logs=Join-Path $root 'lab_runs\b5c_paired_r2_recovery\controller_logs'
 Plain $logs
 New-Item -ItemType Directory -Path $logs -Force|Out-Null
 $token=[guid]::NewGuid().ToString('N')
 $p=Start-Process -FilePath $Python -ArgumentList @('-B','-X','utf8',('"'+$script+'"'),$Action) -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logs ($token+'.stdout.log')) -RedirectStandardError (Join-Path $logs ($token+'.stderr.log')) -PassThru
 [pscustomobject]@{PID=$p.Id;Logs=$logs;AutomaticRetry=$false;NewMFAInference=$false}|Format-List
}else{
 & $Python -B -X utf8 $script $Action
 if($LASTEXITCODE -ne 0){throw 'Recovery STOP: preserve logs; no automatic retry'}
}
