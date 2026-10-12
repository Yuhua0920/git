param()
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
& (Join-Path $PSScriptRoot 'Verify-Package.ps1')
$runtimeDir = Join-Path $taskRoot '.python-runtime'
$basePython = Join-Path $runtimeDir 'python.exe'
if (-not (Test-Path -LiteralPath $basePython)) {
    $installer = Join-Path $PSScriptRoot 'installers\python-3.13.15-amd64.exe'
    $installerArgs = '/quiet InstallAllUsers=0 Include_launcher=0 Include_test=0 Include_pip=1 PrependPath=0 AssociateFiles=0 Shortcuts=0 TargetDir="' + $runtimeDir + '"'
    $process = Start-Process -FilePath $installer -ArgumentList $installerArgs -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -notin @(0,3010)) { throw "Python installer failed: $($process.ExitCode)" }
}
if (-not (Test-Path -LiteralPath $basePython)) { throw 'Python was not installed at the requested portable project location.' }
& $basePython -c "import sys; assert sys.version_info[:3] == (3,13,15); assert sys.maxsize > 2**32; print(sys.version)"
if ($LASTEXITCODE -ne 0) { throw 'Python version/architecture gate failed.' }
$venv = Join-Path $taskRoot '.venv-lab'
if (-not (Test-Path -LiteralPath (Join-Path $venv 'Scripts\python.exe'))) {
    & $basePython -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed.' }
}
$python = Join-Path $venv 'Scripts\python.exe'
& $python -m pip --isolated install --no-index --find-links (Join-Path $PSScriptRoot 'wheelhouse') --require-hashes -r (Join-Path $PSScriptRoot 'requirements-win-cpu.lock')
if ($LASTEXITCODE -ne 0) { throw 'Offline pinned dependency installation failed.' }
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
Write-Output 'ENVIRONMENT_INSTALLED. Run host and fidelity gates next. No MFA training was started.'
