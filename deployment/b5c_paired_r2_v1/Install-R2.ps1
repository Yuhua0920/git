param([Parameter(Mandatory=$true)][string]$ProjectRoot,[Parameter(Mandatory=$true)][string]$ZipPath,[Parameter(Mandatory=$true)][string]$ExpectedSHA256,[string]$Python='')
$ErrorActionPreference='Stop'
$env:PSModulePath=(Join-Path $PSHOME 'Modules')+';'+$env:PSModulePath
$root=[IO.Path]::GetFullPath($ProjectRoot)
if(-not(Test-Path -LiteralPath $root -PathType Container)){throw 'Existing LAB project required'}
if([string]::IsNullOrWhiteSpace($Python)){$Python=Join-Path $root '.venv-lab\Scripts\python.exe'}
if(-not(Test-Path -LiteralPath $Python -PathType Leaf)){throw 'Existing LAB Python interpreter missing: STOP'}
if((Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedSHA256.ToLowerInvariant()){throw 'R2 ZIP SHA256 mismatch: STOP'}
$dest=Join-Path $root 'deployment\b5c_paired_r2_v1'
if(Test-Path -LiteralPath $dest){throw 'R2 namespace already exists: STOP; no overwrite'}
$r1=Join-Path $root 'deployment\b5c_paired_r1_v1\package_manifest.json'
if((Get-FileHash -LiteralPath $r1 -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'a2d3074aea59fc2f4292aacaecc53acfe5bebe8e2125ebcc748d85914ff1aeba'){throw 'Certified R1 manifest mismatch: STOP'}
foreach($prop in (Get-Content -LiteralPath $r1 -Raw -Encoding UTF8|ConvertFrom-Json).files.PSObject.Properties){
 $q=Join-Path $root $prop.Name
 if(-not(Test-Path -LiteralPath $q -PathType Leaf)){throw ('Frozen existing dependency missing: '+$prop.Name)}
 if((Get-FileHash -LiteralPath $q -Algorithm SHA256).Hash.ToLowerInvariant() -ne $prop.Value.sha256){throw ('Frozen dependency changed: '+$prop.Name)}
}
foreach($name in @('b5c_paired','b5c_paired_r1','b5c_paired_r2')){
 if(Test-Path -LiteralPath (Join-Path $root ('lab_runs\'+$name+'\execution.lock'))){throw 'Existing task still running: STOP'}
}
function HistoricalHashes {
 $r=@{}
 foreach($name in @('deployment\b5c_paired_v1','deployment\b5c_paired_r1_v1','lab_runs\b5c_paired','lab_runs\b5c_paired_r1')){
  $folder=Join-Path $root $name
  if(Test-Path -LiteralPath $folder){Get-ChildItem -LiteralPath $folder -Recurse -File|ForEach-Object {$r[$_.FullName]=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash}}
 }
 return $r
}
$before=HistoricalHashes
Add-Type -AssemblyName System.IO.Compression.FileSystem
$z=[IO.Compression.ZipFile]::OpenRead([IO.Path]::GetFullPath($ZipPath))
try{
 $seen=@{}
 foreach($e in $z.Entries){
  $n=$e.FullName
  if(-not $n.StartsWith('deployment/b5c_paired_r2_v1/') -or $n.Contains('..') -or $n.Contains(':') -or $n.Contains('\') -or $seen.ContainsKey($n)){throw 'Unsafe R2 ZIP entry'}
  $seen[$n]=$true
 }
 $entry=$z.GetEntry('deployment/b5c_paired_r2_v1/Install-R2.ps1')
 if($null -eq $entry){throw 'Packed installer missing'}
 $s=$entry.Open();$h=[Security.Cryptography.SHA256]::Create()
 try{$packed=([BitConverter]::ToString($h.ComputeHash($s))).Replace('-','').ToLowerInvariant()}finally{$s.Dispose();$h.Dispose()}
 if((Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $packed){throw 'External installer mismatch: STOP'}
 foreach($e in $z.Entries){
  $q=[IO.Path]::GetFullPath((Join-Path $root $e.FullName))
  if(-not $q.StartsWith(($dest.TrimEnd('\')+'\'),[StringComparison]::OrdinalIgnoreCase)){throw 'Target escaped R2 namespace'}
  [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($q))|Out-Null
  $src=$e.Open();$out=[IO.File]::Open($q,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
  try{$src.CopyTo($out)}finally{$src.Dispose();$out.Dispose()}
 }
}finally{$z.Dispose()}
$after=HistoricalHashes
if($before.Count -ne $after.Count){throw 'Historical inventory changed: STOP'}
foreach($k in $before.Keys){if($before[$k] -ne $after[$k]){throw ('Historical file changed: '+$k)}}
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $dest 'R2.ps1') -Action Verify -Python $Python
if($LASTEXITCODE -ne 0){throw 'R2 installation Verify failed: STOP'}
[pscustomobject]@{PASS=$true;HistoricalFilesUnchanged=$before.Count;NoModelOrDataCopied=$true;FormalStarted=$false}|Format-List
