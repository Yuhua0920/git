param([Parameter(Mandatory=$true)][string]$ProjectRoot,[Parameter(Mandatory=$true)][string]$ZipPath,[Parameter(Mandatory=$true)][string]$ExpectedSha256)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Utility\Microsoft.PowerShell.Utility.psd1') -ErrorAction Stop
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Management\Microsoft.PowerShell.Management.psd1') -ErrorAction Stop
Add-Type -AssemblyName System.IO.Compression.FileSystem
$root=[IO.Path]::GetFullPath($ProjectRoot)
$zip=[IO.Path]::GetFullPath($ZipPath)
if (-not (Test-Path -LiteralPath $root -PathType Container)) {throw 'Existing laboratory project required; installer never creates replacement project.'}
if ($ExpectedSha256 -notmatch '^[a-fA-F0-9]{64}$') {throw 'Fixed expected SHA256 required.'}
$actual=(Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $ExpectedSha256.ToLowerInvariant()) {throw 'Incremental ZIP SHA256 mismatch: STOP.'}
$target=Join-Path $root 'deployment\r7_evidence_audit_r1_v1'
if (Test-Path -LiteralPath $target) {throw 'Audit namespace already exists; no overwrite or retry.'}
foreach ($stage in @('r5_128gb_v1','r6_formal_v1','r7_sequential_v1')) {
    if (-not (Test-Path -LiteralPath (Join-Path $root "deployment\$stage\package_manifest.json"))) {throw "Missing frozen $stage package."}
}
$oldManifest=Join-Path $root 'deployment\r7_evidence_audit_v1\package_manifest.json'
if (-not (Test-Path -LiteralPath $oldManifest) -or (Get-FileHash -LiteralPath $oldManifest -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'fcaa443044e84298e3f4fa59b562eafecdcddeb175baf4b35a7ae7fd10b61812') {throw 'Original v1 audit deployment missing or changed; STOP.'}
$archive=[IO.Compression.ZipFile]::OpenRead($zip)
try {
    $names=@{}
    foreach ($entry in $archive.Entries) {
        $n=$entry.FullName
        if ($n -notmatch '^deployment/r7_evidence_audit_r1_v1/[^/]+$' -or $n.Contains('..') -or $n.Contains('\') -or $n.Contains(':') -or $names.ContainsKey($n)) {throw 'Unsafe or duplicate ZIP member.'}
        $names[$n]=$true
    }
} finally {$archive.Dispose()}
$staging=Join-Path ([IO.Path]::GetDirectoryName($zip)) ('R7Audit_Staging_'+[guid]::NewGuid().ToString('N'))
Expand-Archive -LiteralPath $zip -DestinationPath $staging
$manifestPath=Join-Path $staging 'deployment\r7_evidence_audit_r1_v1\package_manifest.json'
$manifest=Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($names.Count -ne @($manifest.files.PSObject.Properties).Count+1) {throw 'Package membership differs from manifest.'}
foreach ($item in $manifest.files.PSObject.Properties) {
    if (-not $names.ContainsKey($item.Name)) {throw 'Manifest member missing.'}
    $p=Join-Path $staging $item.Name
    if ((Get-Item -LiteralPath $p).Length -ne $item.Value.bytes -or (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant() -ne $item.Value.sha256) {throw 'Extracted member SHA/size mismatch.'}
}
$outer=$PSCommandPath
$packed=Join-Path $staging 'deployment\r7_evidence_audit_r1_v1\Install-R7AuditR1.ps1'
if ((Get-FileHash -LiteralPath $outer -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath $packed -Algorithm SHA256).Hash) {throw 'External installer differs from original packed bytes.'}
Copy-Item -LiteralPath (Join-Path $staging 'deployment\r7_evidence_audit_r1_v1') -Destination $target -Recurse
foreach ($item in $manifest.files.PSObject.Properties) {
    $p=Join-Path $root $item.Name
    if ((Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant() -ne $item.Value.sha256) {throw 'Installed tool hash mismatch; preserve partial installation and STOP.'}
}
Write-Output 'R7 read-only audit installed in a new namespace. Scientific directories were not copied or overwritten. Laboratory physical audit has NOT run.'
