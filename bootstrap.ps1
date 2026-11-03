param([string]$Python = 'python',
      [string]$VerifiedPackages = '',
      [switch]$Install)
$ErrorActionPreference = 'Stop'
if (!$Install -and [string]::IsNullOrWhiteSpace($VerifiedPackages)) {
    throw 'Fill -VerifiedPackages with your local verified dependency directory, or explicitly choose -Install.'
}
if (!$Install -and !(Test-Path -LiteralPath (Join-Path $VerifiedPackages 'torch\__init__.py'))) {
    throw 'Verified package directory missing; fill -VerifiedPackages or explicitly choose -Install.'
}
& $Python -m venv "$PSScriptRoot\.venv"
if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
$TaskPython = "$PSScriptRoot\.venv\Scripts\python.exe"
if ($Install) {
    $BridgePath = "$PSScriptRoot\.venv\Lib\site-packages\softcas-local-dependencies.pth"
    if (Test-Path -LiteralPath $BridgePath) { Remove-Item -LiteralPath $BridgePath }
    & $TaskPython -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed' }
    & $TaskPython -m pip install ultralytics==8.4.0 numpy==2.3.5 scipy==1.18.1
    if ($LASTEXITCODE -ne 0) { throw 'Backend dependency installation failed' }
} else {
    if (!(Test-Path -LiteralPath "$VerifiedPackages\torch\__init__.py")) { throw 'Verified package directory missing; use -Install' }
    # Read-only dependency bridge: no mutation of the previous project's environment.
    Set-Content -LiteralPath "$PSScriptRoot\.venv\Lib\site-packages\softcas-local-dependencies.pth" -Value $VerifiedPackages -Encoding utf8
}
& $TaskPython -c 'import torch,ultralytics,numpy,scipy; assert torch.__version__.split("+")[0]=="2.5.1"; assert ultralytics.__version__=="8.4.0"; print(torch.__version__,ultralytics.__version__,numpy.__version__,scipy.__version__)'
if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed' }
