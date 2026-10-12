param([Parameter(Mandatory=$true)][string]$ZipPath,[Parameter(Mandatory=$true)][string]$ExpectedSha256,[string]$ProjectRoot='D:\Research\Project_MFA_Lab')
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Utility\Microsoft.PowerShell.Utility.psd1') -ErrorAction Stop
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Management\Microsoft.PowerShell.Management.psd1') -ErrorAction Stop
$root=[IO.Path]::GetFullPath($ProjectRoot)
if ($ExpectedSha256 -notmatch '^[0-9a-fA-F]{64}$') {throw 'Fixed expected SHA256 required.'}
if ((Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedSha256.ToLowerInvariant()) {throw 'R7 ZIP SHA256 mismatch: STOP.'}
if (-not (Test-Path -LiteralPath (Join-Path $root 'deployment\r6_formal_v1\package_manifest.json'))) {throw 'Existing frozen R6 installation required.'}
if (Test-Path -LiteralPath (Join-Path $root 'deployment\r7_sequential_v1')) {throw 'R7 namespace already exists: no overwrite, no reinstall.'}
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive=[IO.Compression.ZipFile]::OpenRead([IO.Path]::GetFullPath($ZipPath));$seen=@{}
try {
    foreach ($e in $archive.Entries) {
        $n=$e.FullName.Replace('\','/');$dest=[IO.Path]::GetFullPath((Join-Path $root $n));$prefix=(Join-Path $root 'deployment\r7_sequential_v1')+'\'
        if ($n.EndsWith('/') -or $n -notmatch '^deployment/r7_sequential_v1/' -or $n -match '(^|/)\.\.(/|$)' -or $n.Contains(':') -or -not $dest.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) {throw 'Unsafe archive path: STOP.'}
        if ($seen.ContainsKey($dest.ToLowerInvariant()) -or (Test-Path -LiteralPath $dest)) {throw 'Duplicate/overwrite conflict: STOP.'};$seen[$dest.ToLowerInvariant()]=$true
    }
    foreach ($e in $archive.Entries) {$dest=Join-Path $root $e.FullName;[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($dest))|Out-Null;[IO.Compression.ZipFileExtensions]::ExtractToFile($e,$dest,$false)}
} finally {$archive.Dispose()}
$mf=Get-Content -LiteralPath (Join-Path $root 'deployment\r7_sequential_v1\package_manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($p in $mf.files.PSObject.Properties) {if ((Get-FileHash -LiteralPath (Join-Path $root $p.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $p.Value.sha256) {throw 'Installed R7 file hash mismatch: STOP.'}}
$cfg=Get-Content -LiteralPath (Join-Path $root 'deployment\r7_sequential_v1\queue_contract.json') -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($p in $cfg.R6_dependencies.PSObject.Properties) {if ((Get-FileHash -LiteralPath (Join-Path $root $p.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $p.Value) {throw 'Existing frozen R6 drift: STOP.'}}
Write-Output 'R7 installation PASS. Formal authorization remains pending; no training started.'
