param([string]$JobName)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if ($JobName) {
    $job = [IO.Path]::GetFullPath((Join-Path (Join-Path $taskRoot 'lab_runs') $JobName))
    $allowed = [IO.Path]::GetFullPath((Join-Path $taskRoot 'lab_runs')) + [IO.Path]::DirectorySeparatorChar
    if (-not $job.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid job name' }
} else {
    $latestPath = Join-Path $taskRoot 'lab_runs\latest.json'
    if (-not (Test-Path -LiteralPath $latestPath)) { Write-Output 'NO_JOB_STARTED'; exit 0 }
    $latest = Get-Content -LiteralPath $latestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $job = Join-Path $taskRoot $latest.job
}
$status = Join-Path $job 'status.json'
if (Test-Path -LiteralPath $status) { Get-Content -LiteralPath $status -Raw -Encoding UTF8 } else { Write-Output "STARTING_OR_PRECHECK_FAILED: inspect $job" }
$errorLog = Join-Path $job 'supervisor.stderr.log'
if (Test-Path -LiteralPath $errorLog) { Get-Content -LiteralPath $errorLog -Tail 8 }
