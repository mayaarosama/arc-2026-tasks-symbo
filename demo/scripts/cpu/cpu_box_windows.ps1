<#
CPU box, Windows: set up, start, stop and check the three workers (t2i :8101, i23d :8102, music :8103).

  powershell -ExecutionPolicy Bypass -File scripts\cpu\cpu_box_windows.ps1 setup     # needs network
  powershell -ExecutionPolicy Bypass -File scripts\cpu\cpu_box_windows.ps1 start
  powershell -ExecutionPolicy Bypass -File scripts\cpu\cpu_box_windows.ps1 status
  powershell -ExecutionPolicy Bypass -File scripts\cpu\cpu_box_windows.ps1 stop
  powershell -ExecutionPolicy Bypass -File scripts\cpu\cpu_box_windows.ps1 firewall  # as Administrator, once

Secrets come from a file (default: ..\config.txt next to demo\), never from arguments:
  ARC_DEMO_TOKEN=<shared secret, same on every machine>
  HF_TOKEN=hf_...            (a bare line starting with hf_ is also accepted)
Values are never printed. Every command is echoed with "+"; setup is also saved to results\setup-<time>.log.
Requires: Python 3.10/3.11 (py launcher), git, and Visual Studio Build Tools with
"Desktop development with C++" (SF3D compiles texture_baker / uv_unwrapper).
#>
param(
    [Parameter(Position = 0)][ValidateSet("setup", "start", "stop", "status", "firewall")][string]$Action = "status",
    [string]$Secrets = "",
    [string]$WorkerConfig = "config/worker-cpu.yaml",
    [string]$PythonVersion = "3.11",
    [string]$TorchVersion = ""          # match the DGX lock, e.g. 2.6.0; empty = latest
)
$ErrorActionPreference = "Stop"
$Demo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Secrets) { $Secrets = Join-Path $Demo "..\config.txt" }
$Sf3dDir = Join-Path $Demo "third_party\stable-fast-3d"
$Sa3Dir = Join-Path $Demo "third_party\stable-audio-3"
$Sa3Torch = "2.7.1"                 # stable-audio-3 pins torch + torchaudio to this version
$TorchIndex = "https://download.pytorch.org/whl/cpu"
$Ports = [ordered]@{ t2i = 8101; i23d = 8102; music = 8103 }
Set-Location $Demo
New-Item -ItemType Directory -Force results | Out-Null

function Invoke-Step([string]$Exe, [string[]]$Arguments) {
    # Echo, run, and stop on a non-zero exit code (native tools do not throw on their own).
    Write-Host "+ $Exe $($Arguments -join ' ')"
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "failed ($LASTEXITCODE): $Exe $($Arguments -join ' ')" }
}

function Get-VenvPython([string]$Venv) { Join-Path $Demo "$Venv\Scripts\python.exe" }

function Import-DemoSecrets {
    if (-not (Test-Path -LiteralPath $Secrets)) { throw "secrets file not found: $Secrets (use -Secrets <path>)" }
    foreach ($raw in Get-Content -LiteralPath $Secrets) {
        $line = $raw.Trim()
        if ($line -eq "" -or $line.StartsWith("#")) { continue }
        if ($line.StartsWith("hf_")) { $env:HF_TOKEN = $line; continue }
        $i = $line.IndexOf("=")
        if ($i -lt 1) { Write-Warning "secrets file: ignoring a line that is neither KEY=VALUE nor an hf_ token"; continue }
        $key = $line.Substring(0, $i).Trim()
        if ($key -eq "HF_TOKEN" -or $key -eq "ARC_DEMO_TOKEN") {
            Set-Item -Path "env:$key" -Value $line.Substring($i + 1).Trim()   # empty value = unset
        } else {
            Write-Warning "secrets file: ignoring unknown key '$key'"
        }
    }
    $hf = if ($env:HF_TOKEN) { "set" } else { "MISSING" }
    $arc = if ($env:ARC_DEMO_TOKEN) { "set" } else { "empty (workers accept anyone)" }
    Write-Host "secrets: HF_TOKEN $hf, ARC_DEMO_TOKEN $arc"
}

function New-DemoVenv([string]$Venv, [string[]]$Torch = @()) {
    # CPU torch first, so nothing installed later pulls a CUDA wheel.
    if (-not (Test-Path $Venv)) { Invoke-Step "py" @("-$PythonVersion", "-m", "venv", $Venv) }
    $py = Get-VenvPython $Venv
    Invoke-Step $py @("-m", "pip", "install", "-q", "--upgrade", "pip")
    if (-not $Torch) { $Torch = @($(if ($TorchVersion) { "torch==$TorchVersion" } else { "torch" })) }
    Invoke-Step $py (@("-m", "pip", "install", "-q") + $Torch + @("--index-url", $TorchIndex))
    Write-Host "torch in ${Venv}: $(& $py -c 'import torch; print(torch.__version__)')"
}

function Invoke-Setup {
    Start-Transcript -Path ("results\setup-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss")) | Out-Null
    try {
        if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw "the py launcher is missing: install Python $PythonVersion from python.org" }
        if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "git is required to fetch Stable Fast 3D" }
        $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
        if (-not (Test-Path $vswhere)) { Write-Warning "Visual Studio Build Tools not found: SF3D's C++ extensions will fail to build" }
        Import-DemoSecrets
        if (-not $env:HF_TOKEN) { throw "no Hugging Face token in $Secrets (gated repos need it)" }

        Write-Host "== text-to-image env (.venv-t2i)"
        New-DemoVenv ".venv-t2i"
        Invoke-Step (Get-VenvPython ".venv-t2i") @("-m", "pip", "install", "-q", "-r", "requirements/t2i.txt")

        Write-Host "== image-to-3D env (.venv-i23d): Stable Fast 3D"
        New-DemoVenv ".venv-i23d"
        if (-not (Test-Path $Sf3dDir)) { Invoke-Step "git" @("clone", "https://github.com/Stability-AI/stable-fast-3d", $Sf3dDir) }
        Write-Host "stable-fast-3d commit: $(git -C $Sf3dDir rev-parse HEAD)"
        $py3d = Get-VenvPython ".venv-i23d"
        Invoke-Step $py3d @("-m", "pip", "install", "-q", "-U", "setuptools==69.5.1", "wheel")
        Push-Location $Sf3dDir   # its requirements.txt installs ./texture_baker and ./uv_unwrapper by relative path
        try { Invoke-Step $py3d @("-m", "pip", "install", "-q", "-r", "requirements.txt") } finally { Pop-Location }
        Invoke-Step $py3d @("-m", "pip", "install", "-q", "-r", "requirements/worker-base.txt")
        # worker-base would pull huggingface_hub 1.x+, but SF3D pins transformers 4.42.3 (needs <1.0)
        Invoke-Step $py3d @("-m", "pip", "install", "-q", "huggingface_hub>=0.25,<1.0")

        Write-Host "== music env (.venv-music): Stable Audio 3.0 Small"
        New-DemoVenv ".venv-music" @("torch==$Sa3Torch", "torchaudio==$Sa3Torch")
        if (-not (Test-Path $Sa3Dir)) { Invoke-Step "git" @("clone", "https://github.com/Stability-AI/stable-audio-3", $Sa3Dir) }
        Write-Host "stable-audio-3 commit: $(git -C $Sa3Dir rev-parse HEAD)"
        $pyMusic = Get-VenvPython ".venv-music"
        Invoke-Step $pyMusic @("-m", "pip", "install", "-q", "-r", "requirements/music.txt")
        Invoke-Step $pyMusic @("-m", "pip", "install", "-q", "--no-deps", "-e", $Sa3Dir)

        Write-Host "== weights (HF_TOKEN from the secrets file)"
        Invoke-Step (Get-VenvPython ".venv-t2i") @("scripts/download_weights.py", "--machine", "cpu", "--task", "t2i")
        Invoke-Step $py3d @("scripts/download_weights.py", "--machine", "cpu", "--task", "i23d")
        Invoke-Step $pyMusic @("scripts/download_weights.py", "--machine", "cpu", "--task", "music")

        Write-Host "== this machine"
        Invoke-Step (Get-VenvPython ".venv-t2i") @("scripts/cpu/check_cpu.py")
        Write-Host "== offline check (network use is now forbidden; a FAIL here would fail on stage)"
        Remove-Item env:HF_TOKEN
        Invoke-Step (Get-VenvPython ".venv-t2i") @("scripts/verify_offline.py", "--config", $WorkerConfig, "--task", "t2i")
        $env:PYTHONPATH = $Sf3dDir
        try { Invoke-Step $py3d @("scripts/verify_offline.py", "--config", $WorkerConfig, "--task", "i23d") }
        finally { Remove-Item env:PYTHONPATH -ErrorAction SilentlyContinue }
        Invoke-Step $pyMusic @("scripts/verify_offline.py", "--config", $WorkerConfig, "--task", "music")
        Write-Host "setup done. Next, as Administrator: ... cpu_box_windows.ps1 firewall"
    } finally {
        Stop-Transcript | Out-Null
    }
}

function Get-Health([int]$Port) {
    $headers = @{}
    if ($env:ARC_DEMO_TOKEN) { $headers["X-Demo-Token"] = $env:ARC_DEMO_TOKEN }
    try { return (Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -Headers $headers -TimeoutSec 2 | ConvertTo-Json -Compress) }
    catch { return $null }
}

function Start-Workers {
    Import-DemoSecrets
    Remove-Item env:HF_TOKEN -ErrorAction SilentlyContinue   # not needed offline; keep it out of the workers
    $env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"; $env:SF3D_USE_CPU = "1"
    foreach ($task in $Ports.Keys) {
        $pidFile = "results\worker-$task.pid"
        if ((Test-Path $pidFile) -and (Get-Process -Id (Get-Content $pidFile) -ErrorAction SilentlyContinue)) {
            Write-Host "$task already running (pid $(Get-Content $pidFile))"; continue
        }
        $py = Get-VenvPython ".venv-$task"
        if (-not (Test-Path $py)) { throw ".venv-$task missing: run setup first" }
        # PYTHONPATH is inherited by the child: SF3D's repo for i23d, nothing for t2i
        if ($task -eq "i23d") { $env:PYTHONPATH = $Sf3dDir } else { Remove-Item env:PYTHONPATH -ErrorAction SilentlyContinue }
        $workerArgs = "-m arcdemo.worker --config $WorkerConfig --task $task --port $($Ports[$task])"
        $out = Join-Path $Demo "results\stdout-$task.log"
        Write-Host "+ .venv-${task}: python $workerArgs > results\stdout-$task.log"
        # Through cmd + ShellExecute (no -Redirect*), so the worker inherits none of this shell's
        # handles: a captured launcher (task scheduler, remote session) would otherwise never exit.
        $p = Start-Process -FilePath "cmd.exe" -WorkingDirectory $Demo -WindowStyle Hidden -PassThru `
            -ArgumentList "/d /s /c `"`"$py`" $workerArgs > `"$out`" 2>&1`""
        Set-Content -Path $pidFile -Value $p.Id
    }
    Remove-Item env:PYTHONPATH -ErrorAction SilentlyContinue
    foreach ($task in $Ports.Keys) {   # weights preload in the background; /health answers first
        $ok = $false
        for ($i = 0; $i -lt 60 -and -not $ok; $i++) { $ok = [bool](Get-Health $Ports[$task]); if (-not $ok) { Start-Sleep 1 } }
        if (-not $ok) { throw "$task worker did not answer on :$($Ports[$task]) - see results\stdout-$task.log" }
    }
    Show-Status
}

function Stop-Workers {
    foreach ($task in $Ports.Keys) {
        $pidFile = "results\worker-$task.pid"
        if (Test-Path $pidFile) {
            $workerPid = Get-Content $pidFile
            Write-Host "+ taskkill /T /F /PID $workerPid (${task}: cmd and its python child)"
            taskkill /T /F /PID $workerPid | Out-Null
            Remove-Item $pidFile
        } else { Write-Host "${task}: not running" }
    }
}

function Show-Status {
    Import-DemoSecrets
    foreach ($task in $Ports.Keys) {
        $h = Get-Health $Ports[$task]
        if (-not $h) { $h = "DOWN" }
        Write-Host "$task :$($Ports[$task])  $h"
    }
    Write-Host "logs: results\worker-<task>.log (worker), results\stdout-<task>.log (crashes before logging starts)"
}

function Open-Firewall {
    $admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $admin) { throw "run this action from an Administrator PowerShell" }
    New-NetFirewallRule -DisplayName "ARC demo workers" -Direction Inbound -Protocol TCP -LocalPort 8101, 8102, 8103 -Action Allow | Out-Null
    Write-Host "inbound TCP 8101, 8102, 8103 allowed"
}

switch ($Action) {
    "setup"    { Invoke-Setup }
    "start"    { Start-Workers }
    "stop"     { Stop-Workers }
    "status"   { Show-Status }
    "firewall" { Open-Firewall }
}
