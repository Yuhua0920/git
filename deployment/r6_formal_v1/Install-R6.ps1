param([Parameter(Mandatory=$true)][string]$ZipPath,
      [Parameter(Mandatory=$true)][string]$Sha256Path,
      [string]$ProjectRoot='D:\Research\Project_MFA_Lab')
$ErrorActionPreference='Stop'
# PowerShell 5.1 can inherit a PowerShell 7 PSModulePath. Resolve its own
# standard modules explicitly; no dependency installation or environment edit.
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Utility\Microsoft.PowerShell.Utility.psd1') -ErrorAction Stop
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Management\Microsoft.PowerShell.Management.psd1') -ErrorAction Stop
$targetRoot=[IO.Path]::GetFullPath($ProjectRoot)
if (-not (Test-Path -LiteralPath (Join-Path $targetRoot 'migration_manifest.json'))) { throw 'Existing original lab migration_manifest.json required.' }
$expected=((Get-Content -LiteralPath $Sha256Path -Raw).Trim() -split '\s+')[0].ToLower()
if ((Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash.ToLower() -ne $expected) { throw 'ZIP SHA256 mismatch; STOP.' }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip=[IO.Compression.ZipFile]::OpenRead([IO.Path]::GetFullPath($ZipPath))
$seen=@{}
try {
    # Check all paths/collisions before writing the first file.
    foreach ($entry in $zip.Entries) {
        $name=$entry.FullName.Replace('\','/')
        if ($name.EndsWith('/')) { throw 'Unexpected directory entry.' }
        if ($name -notmatch '^deployment/r6_formal_v1/' -or $name -match '(^|/)\.\.(/|$)' -or $name.Contains(':')) { throw "Unsafe/nonincremental entry $name" }
        $dest=[IO.Path]::GetFullPath((Join-Path $targetRoot $name))
        $prefix=(Join-Path $targetRoot 'deployment\r6_formal_v1')+'\'
        if (-not $dest.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) { throw 'Resolved ZIP path escaped new namespace.' }
        if ($seen.ContainsKey($dest.ToLower())) { throw 'Duplicate ZIP path.' }; $seen[$dest.ToLower()]=$true
        if (Test-Path -LiteralPath $dest) { throw "Overwrite conflict: $dest. No files installed." }
    }
    foreach ($entry in $zip.Entries) {
        $dest=Join-Path $targetRoot $entry.FullName
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($dest)) | Out-Null
        [IO.Compression.ZipFileExtensions]::ExtractToFile($entry,$dest,$false)
    }
} finally { $zip.Dispose() }
$manifestPath=Join-Path $targetRoot 'deployment\r6_formal_v1\package_manifest.json'
$manifest=Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($property in $manifest.files.PSObject.Properties) {
    $file=Join-Path $targetRoot $property.Name
    if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLower() -ne $property.Value.sha256) { throw "Installed leaf mismatch: $file" }
}
Write-Output "R6 installed and leaf hashes verified under $targetRoot. Resource amendment remains MENTOR_APPROVAL_PENDING."
