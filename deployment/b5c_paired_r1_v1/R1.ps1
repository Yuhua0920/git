param([Parameter(Mandatory=$true)][ValidateSet('Verify','ToyTests','IdentityAudit','Status','Export')][string]$Action,[string]$Python='')
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if(-not $Python){$Python=Join-Path $root '.venv-lab\Scripts\python.exe'}
if(-not(Test-Path -LiteralPath $Python)){throw 'Pinned Python missing: STOP'}
& $Python -B -X utf8 (Join-Path $PSScriptRoot 'r1_control.py') $Action
if($LASTEXITCODE -ne 0){throw ('R1 '+$Action+' failed: preserve attempt; no automatic retry')}
