param([string]$Python = 'python', [string]$Environment = '.venv-5080')
$ErrorActionPreference = 'Stop'
$TaskEnvironment = Join-Path $PSScriptRoot $Environment
& $Python -m venv $TaskEnvironment
if ($LASTEXITCODE -ne 0) { throw 'Install/select Python 3.10+ and retry with -Python <path>' }
$TaskPython = Join-Path $TaskEnvironment 'Scripts\python.exe'
& $TaskPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip setup failed' }
# Official PyTorch cu128 builds include Blackwell support. This environment is
# separate from the previous CPU reference environment and from Jetson wheels.
& $TaskPython -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'Blackwell CUDA wheel installation failed' }
& $TaskPython -m pip install 'ultralytics==8.4.0' 'numpy>=1.26,<3' 'scipy>=1.11' 'Pillow>=10' 'psutil>=5.9' PyYAML 'onnx>=1.16,<2' 'onnxscript>=0.1,<1'
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $TaskPython -m pip install --no-deps -e $PSScriptRoot
if ($LASTEXITCODE -ne 0) { throw 'SoftCAS installation failed' }
Write-Output "Prepared $TaskEnvironment. Hardware checks occur only when you explicitly invoke softcas.production doctor."
