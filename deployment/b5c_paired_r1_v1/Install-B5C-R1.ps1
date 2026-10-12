param([Parameter(Mandatory=$true)][string]$ProjectRoot,[Parameter(Mandatory=$true)][string]$ZipPath,[Parameter(Mandatory=$true)][string]$ExpectedSHA256,[string]$Python='')
$ErrorActionPreference='Stop'
$env:PSModulePath=(Join-Path $PSHOME 'Modules')+';'+$env:PSModulePath
$root=[IO.Path]::GetFullPath($ProjectRoot)
if(-not(Test-Path -LiteralPath $root -PathType Container)){throw 'Existing project root required'}
if((Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedSHA256.ToLowerInvariant()){throw 'Fix ZIP SHA256 mismatch: STOP'}
$dest=Join-Path $root 'deployment\b5c_paired_r1_v1'
if(Test-Path -LiteralPath $dest){throw 'R1 namespace already exists: STOP; never overwrite'}
$oldOut=Join-Path $root 'lab_runs\b5c_paired'
if(Test-Path -LiteralPath (Join-Path $oldOut 'execution.lock')){throw 'Original attempt still running: STOP'}
$oldManifest=Join-Path $root 'deployment\b5c_paired_v1\package_manifest.json'
if((Get-FileHash -LiteralPath $oldManifest -Algorithm SHA256).Hash.ToLowerInvariant() -ne '556e96ddc788454d9060ad850ae982cba706879a318acbbbbc98a345670519b5'){throw 'Original B5C frozen manifest mismatch: STOP'}
$records=(Get-Content -LiteralPath $oldManifest -Raw -Encoding UTF8 | ConvertFrom-Json).files
foreach($prop in $records.PSObject.Properties){
 $q=Join-Path $root $prop.Name
 if(-not(Test-Path -LiteralPath $q -PathType Leaf)){throw ('Original dependency missing: '+$prop.Name)}
 if((Get-FileHash -LiteralPath $q -Algorithm SHA256).Hash.ToLowerInvariant() -ne $prop.Value.sha256){throw ('Original frozen dependency changed: '+$prop.Name)}
}
function SnapshotOld {
 $result=@{}
 foreach($folder in @((Join-Path $root 'deployment\b5c_paired_v1'),$oldOut)){
  if(Test-Path -LiteralPath $folder){Get-ChildItem -LiteralPath $folder -Recurse -File | ForEach-Object {$result[$_.FullName]=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash}}
 }
 return $result
}
$before=SnapshotOld
Add-Type -AssemblyName System.IO.Compression.FileSystem
$z=[IO.Compression.ZipFile]::OpenRead([IO.Path]::GetFullPath($ZipPath))
try{
 $seen=@{}
 foreach($e in $z.Entries){
  $name=$e.FullName
  if(-not $name.StartsWith('deployment/b5c_paired_r1_v1/') -or $name.Contains('..') -or $name.Contains(':') -or $name.Contains('\') -or $seen.ContainsKey($name)){throw 'Invalid fix ZIP member: STOP'}
  $seen[$name]=$true
 }
 # Authenticate the outside installer against the ZIP copy before any write.
 $entry=$z.GetEntry('deployment/b5c_paired_r1_v1/Install-B5C-R1.ps1')
 if($null -eq $entry){throw 'Packed installer missing'}
 $stream=$entry.Open();$hasher=[Security.Cryptography.SHA256]::Create()
 try{$packed=([BitConverter]::ToString($hasher.ComputeHash($stream))).Replace('-','').ToLowerInvariant()}finally{$stream.Dispose();$hasher.Dispose()}
 if((Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $packed){throw 'External installer differs from frozen ZIP installer: STOP'}
 foreach($e in $z.Entries){
  $target=[IO.Path]::GetFullPath((Join-Path $root $e.FullName))
  if(-not $target.StartsWith(($dest.TrimEnd('\')+'\'),[StringComparison]::OrdinalIgnoreCase)){throw 'Fix target escaped namespace'}
  [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target))|Out-Null
  $input=$e.Open();$output=[IO.File]::Open($target,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
  try{$input.CopyTo($output)}finally{$input.Dispose();$output.Dispose()}
 }
}finally{$z.Dispose()}
$after=SnapshotOld
if($before.Count -ne $after.Count){throw 'Original failure/source file inventory changed: STOP'}
foreach($key in $before.Keys){if($before[$key] -ne $after[$key]){throw ('Original file changed: '+$key)}}
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $dest 'R1.ps1') -Action Verify -Python $Python
if($LASTEXITCODE -ne 0){throw 'R1 installed verification failed: STOP'}
[pscustomobject]@{PASS=$true;OriginalFilesPreserved=$before.Count;FormalRestartEnabled=$false;Installed=$dest}|Format-List
