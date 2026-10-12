param([Parameter(Mandatory=$true)][string]$ProjectRoot,[Parameter(Mandatory=$true)][string]$ZipPath,[Parameter(Mandatory=$true)][string]$ExpectedSha256)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Utility\Microsoft.PowerShell.Utility.psd1') -ErrorAction Stop
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Management\Microsoft.PowerShell.Management.psd1') -ErrorAction Stop
Add-Type -AssemblyName System.IO.Compression.FileSystem
$root=[IO.Path]::GetFullPath($ProjectRoot);$zip=[IO.Path]::GetFullPath($ZipPath)
if (-not(Test-Path -LiteralPath $root -PathType Container)) {throw 'Existing laboratory project required.'}
if ($ExpectedSha256 -notmatch '^[a-fA-F0-9]{64}$' -or (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedSha256.ToLowerInvariant()) {throw 'ZIP SHA256 mismatch: STOP.'}
$target=Join-Path $root 'deployment\final_mfa_once_v1'
if (Test-Path -LiteralPath $target) {throw 'Final-only namespace exists: no overwrite.'}
$pins=@{'r6_formal_v1'='3fd2174087381d5420ceee4db4b7ea9412a6922d831e6b7d57038ab49d3856d2';'r8_selection_v1'='ea1c37cff3c2b41e3a652b3bc475b29d1672f14814de1f0e026728bfaf0b2578'}
foreach ($k in $pins.Keys) {
 $p=Join-Path $root "deployment\$k\package_manifest.json"
 if (-not(Test-Path -LiteralPath $p) -or (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant() -ne $pins[$k]) {throw "Frozen $k deployment missing or changed."}
}
$z=[IO.Compression.ZipFile]::OpenRead($zip)
try {
 $names=@{}
 foreach ($e in $z.Entries) {
  $n=$e.FullName
  if ($n -notmatch '^deployment/final_mfa_once_v1/[^/]+$' -or $n.Contains('..') -or $n.Contains('\') -or $n.Contains(':') -or $names.ContainsKey($n)) {throw 'Unsafe or duplicate ZIP entry.'}
  $names[$n]=$true
 }
} finally {$z.Dispose()}
$staging=Join-Path ([IO.Path]::GetDirectoryName($zip)) ('FinalMFA_Staging_'+[guid]::NewGuid().ToString('N'))
Expand-Archive -LiteralPath $zip -DestinationPath $staging
$m=Get-Content -LiteralPath (Join-Path $staging 'deployment\final_mfa_once_v1\package_manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($names.Count -ne @($m.files.PSObject.Properties).Count+1) {throw 'ZIP membership differs from manifest.'}
foreach ($x in $m.files.PSObject.Properties) {
 if (-not $names.ContainsKey($x.Name)) {throw 'Manifest member missing.'}
 $p=Join-Path $staging $x.Name
 if ((Get-Item -LiteralPath $p).Length -ne $x.Value.bytes -or (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant() -ne $x.Value.sha256) {throw 'Extracted member hash/size mismatch.'}
}
if ((Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath (Join-Path $staging 'deployment\final_mfa_once_v1\Install-FinalMFA.ps1') -Algorithm SHA256).Hash) {throw 'Outer installer differs from packed bytes.'}
Copy-Item -LiteralPath (Join-Path $staging 'deployment\final_mfa_once_v1') -Destination $target -Recurse
foreach ($x in $m.files.PSObject.Properties) {
 if ((Get-FileHash -LiteralPath (Join-Path $root $x.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $x.Value.sha256) {throw 'Installed member hash mismatch; preserve partial installation and STOP.'}
}
Write-Output 'Final-only increment installed. No old code, authorization, data or result was copied or overwritten. No final fit has started.'
