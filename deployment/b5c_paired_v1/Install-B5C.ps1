param([Parameter(Mandatory=$true)][string]$ProjectRoot,[Parameter(Mandatory=$true)][string]$CodeZip,[Parameter(Mandatory=$true)][string]$ExpectedCodeSHA256,[Parameter(Mandatory=$true)][string]$DataZip,[Parameter(Mandatory=$true)][string]$ExpectedDataSHA256)
$ErrorActionPreference='Stop'
$env:PSModulePath=(Join-Path $PSHOME 'Modules')+';'+$env:PSModulePath
$root=[IO.Path]::GetFullPath($ProjectRoot)
if (-not(Test-Path -LiteralPath $root -PathType Container)){throw 'Existing laboratory project root required'}
foreach($pair in @(@($CodeZip,$ExpectedCodeSHA256),@($DataZip,$ExpectedDataSHA256))){
 $actual=(Get-FileHash -LiteralPath $pair[0] -Algorithm SHA256).Hash.ToLowerInvariant()
 if($actual -ne $pair[1].ToLowerInvariant()){throw 'ZIP SHA256 mismatch: STOP'}
}
$destination=Join-Path $root 'lab_inputs\b5c_b2_v1'
$namespace=Join-Path $root 'deployment\b5c_paired_v1'
if ((Test-Path -LiteralPath $destination) -or (Test-Path -LiteralPath $namespace)){throw 'B5C namespace/input directory already exists: STOP; never overwrite'}
Add-Type -AssemblyName System.IO.Compression.FileSystem
function ReadZip($zipPath){return [IO.Compression.ZipFile]::OpenRead([IO.Path]::GetFullPath($zipPath))}
function SafeTarget($base,$name){
 if($name.Contains('..') -or $name.Contains(':') -or $name.StartsWith('/') -or $name.StartsWith('\')){throw 'Unsafe ZIP path'}
 $target=[IO.Path]::GetFullPath((Join-Path $base $name))
 if(-not $target.StartsWith(($base.TrimEnd('\')+'\'),[StringComparison]::OrdinalIgnoreCase)){throw 'ZIP path escaped root'}
 return $target
}
$archive=ReadZip $CodeZip
try {
 # Check every existing dependency before writing any deployment file.
 foreach($entry in $archive.Entries){
  $target=SafeTarget $root $entry.FullName
  if(Test-Path -LiteralPath $target){
   $stream=$entry.Open();$hasher=[Security.Cryptography.SHA256]::Create()
   try{$packed=([BitConverter]::ToString($hasher.ComputeHash($stream))).Replace('-','').ToLowerInvariant()}finally{$stream.Dispose();$hasher.Dispose()}
   $existing=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
   if($packed -ne $existing){throw ('Frozen dependency conflict: STOP '+$entry.FullName)}
  }
 }
 foreach($entry in $archive.Entries){
  $target=SafeTarget $root $entry.FullName
  if(-not(Test-Path -LiteralPath $target)){
   [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target))|Out-Null
   $inputStream=$entry.Open();$outputStream=[IO.File]::Open($target,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
   try{$inputStream.CopyTo($outputStream)}finally{$inputStream.Dispose();$outputStream.Dispose()}
  }
 }
}finally{$archive.Dispose()}
$archive=ReadZip $DataZip
try {
 [IO.Directory]::CreateDirectory($destination)|Out-Null
 foreach($entry in $archive.Entries){
  $target=SafeTarget $destination $entry.FullName
  [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target))|Out-Null
  $inputStream=$entry.Open();$outputStream=[IO.File]::Open($target,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
  try{$inputStream.CopyTo($outputStream)}finally{$inputStream.Dispose();$outputStream.Dispose()}
 }
}finally{$archive.Dispose()}
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $namespace 'B5C.ps1') -Action Verify
if($LASTEXITCODE -ne 0){throw 'Installed package verify failed: STOP'}
Write-Output 'INSTALL_COMPLETE. Run Preflight separately. No formal evaluation was started.'
