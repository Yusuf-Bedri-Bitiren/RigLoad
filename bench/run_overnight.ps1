# Runs every pending benchmark unattended (about 1.5-2 h). From the repo root:
#   powershell -ExecutionPolicy Bypass -File bench\run_overnight.ps1
#
# Before starting: laptop plugged in, browser and other heavy apps closed.
# Keeps Windows from sleeping while it runs (SetThreadExecutionState; ends with the script,
# no setting is changed). Output: bench\results\*.json and bench\results\overnight.log.
# A failing step is logged and the next one still runs.

$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$env:Path = "$env:USERPROFILE\.cargo\bin;$PWD\.venv\Scripts;$env:Path"
$env:VIRTUAL_ENV = "$PWD\.venv"
$env:PYO3_PYTHON = (Resolve-Path $py).Path
New-Item -ItemType Directory -Force bench\results | Out-Null
$log = "bench\results\overnight.log"
"=== run started $(Get-Date -Format s) ===" | Tee-Object -FilePath $log

Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
# Decimal on purpose: PowerShell 5.1 parses 0x80000001 as a negative Int32.
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

function Invoke-Step([string]$Name, [scriptblock]$Command) {
    "`n=== $Name  $(Get-Date -Format s) ===" | Tee-Object -FilePath $log -Append
    & $Command 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE) { "!!! $Name failed (exit $LASTEXITCODE)" | Tee-Object -FilePath $log -Append }
}

try {
    Invoke-Step "build rigload (release)" { maturin develop --release --extras test }
    Invoke-Step "criterion decode bench" { cargo bench -p rigload-core --bench decode }
    Invoke-Step "decode comparison" { & $py bench\decode_compare.py }
    Invoke-Step "baseline torchvision (aug)" { & $py bench\bench.py --loader torchvision --repeats 7 --batches 200 }
    Invoke-Step "baseline albumentations (aug)" { & $py bench\bench.py --loader albumentations --repeats 7 --batches 200 }
    Invoke-Step "baseline torchvision (no aug)" { & $py bench\bench.py --loader torchvision --no-augment --repeats 7 --batches 200 }
}
finally {
    [Win32.Power]::SetThreadExecutionState([uint32]2147483648) | Out-Null  # ES_CONTINUOUS: back to normal
    "`n=== run finished $(Get-Date -Format s) ===" | Tee-Object -FilePath $log -Append
}
