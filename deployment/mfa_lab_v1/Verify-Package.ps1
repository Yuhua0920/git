param()
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$manifest = Get-Content -LiteralPath (Join-Path $taskRoot 'migration_manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$verified = 0
foreach ($item in $manifest.files.PSObject.Properties) {
    $target = [IO.Path]::GetFullPath((Join-Path $taskRoot $item.Name))
    if (-not $target.StartsWith($taskRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Manifest path escapes package root' }
    if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { throw "Missing payload: $($item.Name)" }
    $file = Get-Item -LiteralPath $target
    if ($file.Length -ne $item.Value.bytes) { throw "Size mismatch: $($item.Name)" }
    # .NET stream hashing also works in Windows PowerShell sessions whose
    # optional script-based Get-FileHash command is not available.
    $stream = [IO.File]::OpenRead($target)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try { $hash = ([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-','').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
    if ($hash -ne $item.Value.sha256) { throw "SHA256 mismatch: $($item.Name)" }
    $verified++
}
Write-Output "PACKAGE_INTEGRITY_PASS: $verified payload files; formal training is not authorized."
