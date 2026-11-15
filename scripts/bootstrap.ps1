[CmdletBinding()]
param(
    [ValidateSet('auto', 'cpu', 'cuda')]
    [string]$Device = 'auto',
    [Alias('Input')]
    [string]$InputPath,
    [Alias('Output')]
    [string]$OutputPath,
    [switch]$SetupOnly,
    [switch]$CheckOnly,
    [switch]$NoOpen
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$UvVersion = '0.12.13'
$PythonVersion = '3.11.16'
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$StampPath = Join-Path $ProjectRoot '.tools\quickstart-environment.json'

function Invoke-Checked {
    param([string]$Executable, [string[]]$CommandArguments)
    & $Executable @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit $LASTEXITCODE): $Executable $($CommandArguments -join ' ')"
    }
}

function Get-SetupFingerprint {
    param([string]$Flavor)
    $text = "bootstrap-v1|$UvVersion|$PythonVersion|torch=2.8.0|torchvision=0.23.0|$Flavor"
    foreach ($file in @('requirements/core-tested.txt', 'requirements/pose-download-tested.txt', 'pyproject.toml')) {
        $text += '|' + (Get-FileHash -LiteralPath (Join-Path $ProjectRoot $file) -Algorithm SHA256).Hash
    }
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($text)))).Replace('-', '').ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
}

function Test-ProjectEnvironment {
    if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { return $false }
    $probe = @'
import importlib,importlib.metadata,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
assert sys.version_info[:3]==(3,11,16), 'Expected Python 3.11.16'
for name,expected in [('torch','2.8.0'),('torchvision','0.23.0')]:
    assert importlib.metadata.version(name).split('+')[0]==expected, name+' version mismatch'
for file in ['requirements/core-tested.txt','requirements/pose-download-tested.txt']:
    for line in (root/file).read_text(encoding='utf-8').splitlines():
        line=line.split('#',1)[0].strip()
        if not line: continue
        name,expected=line.split('==')
        assert importlib.metadata.version(name)==expected, name+' version mismatch'
for name in ['torch','torchvision','numpy','cv2','yaml','sklearn','matplotlib','scipy','PIL','tqdm','gdown','requests','truststore']:
    importlib.import_module(name)
import fall_detect.config
assert fall_detect.config.ROOT==root, 'Editable install belongs to another project path'
'@
    $savedErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $PythonExe -B -X utf8 -c $probe $ProjectRoot 2>$null
        $probeExit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedErrorAction
    }
    return ($probeExit -eq 0)
}

function Get-LocalUv {
    foreach ($candidate in @('.tools\uv\uv.exe', '.tools\Scripts\uv.exe')) {
        $path = Join-Path $ProjectRoot $candidate
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            $version = & $path --version
            if ($LASTEXITCODE -eq 0 -and "$version" -match "^uv $([regex]::Escape($UvVersion))(\s|$)") {
                return $path
            }
        }
    }
    $toolDir = Join-Path $ProjectRoot '.tools\uv'
    New-Item -ItemType Directory -Path $toolDir -Force | Out-Null
    $archive = Join-Path $toolDir "uv-$UvVersion-windows-x64.zip"
    $checksum = "$archive.sha256"
    $base = "https://releases.astral.sh/github/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
    Write-Host "[1/4] Downloading project-local uv $UvVersion..."
    Invoke-WebRequest -UseBasicParsing -Uri "$base.sha256" -OutFile $checksum
    Invoke-WebRequest -UseBasicParsing -Uri $base -OutFile $archive
    $match = [regex]::Match((Get-Content -LiteralPath $checksum -Raw), '(?i)\b[0-9a-f]{64}\b')
    if (-not $match.Success) { throw 'The official uv checksum file is invalid.' }
    $actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
    if ($actual -ne $match.Value) { throw 'The uv archive checksum does not match. No downloaded program was executed.' }
    # Only extract the executable expected in the official archive. Preserve other local tools.
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($archive)
    try {
        $entries = @($zip.Entries | Where-Object { $_.FullName -eq 'uv.exe' })
        if ($entries.Count -ne 1) { throw 'The official uv archive has an unexpected layout.' }
        $destination = Join-Path $toolDir 'uv.exe'
        [IO.Compression.ZipFileExtensions]::ExtractToFile($entries[0], $destination, $true)
    } finally {
        $zip.Dispose()
    }
    $version = & $destination --version
    if ($LASTEXITCODE -ne 0 -or "$version" -notmatch "^uv $([regex]::Escape($UvVersion))(\s|$)") {
        throw 'The downloaded uv does not report the pinned version.'
    }
    return $destination
}

try {
    if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitOperatingSystem) {
        throw 'This launcher supports Windows x64. Use the documented manual setup on other platforms.'
    }
    $architecture = $env:PROCESSOR_ARCHITECTURE
    if ($env:PROCESSOR_ARCHITEW6432) { $architecture = $env:PROCESSOR_ARCHITEW6432 }
    if ($architecture -ne 'AMD64') { throw 'This launcher requires Windows x64 (AMD64).' }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Git is required. Install Git for Windows, then run start.cmd again.' }
    if ($SetupOnly -and $CheckOnly) { throw 'Choose either -SetupOnly or -CheckOnly.' }
    Set-Location -LiteralPath $ProjectRoot
    $env:PYTHONUTF8 = '1'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $ProjectRoot '.python'
    $env:UV_PYTHON_BIN_DIR = Join-Path $ProjectRoot '.tools\python-bin'
    $env:UV_PYTHON_NO_REGISTRY = '1'
    $env:UV_CACHE_DIR = Join-Path $ProjectRoot '.tools\uv-cache'
    $env:UV_NO_MODIFY_PATH = '1'

    $environmentReady = Test-ProjectEnvironment
    if ($CheckOnly -and -not $environmentReady) {
        throw 'The project environment is not ready. Run start.cmd once without -CheckOnly.'
    }
    if (-not $environmentReady) {
        $managedEnvironment = $false
        if (Test-Path -LiteralPath $StampPath -PathType Leaf) {
            try {
                $saved = Get-Content -LiteralPath $StampPath -Raw | ConvertFrom-Json
                $managedEnvironment = ($saved.managed_by -eq 'fall-detect-bootstrap-v1' -and $saved.root -eq $ProjectRoot)
            } catch { $managedEnvironment = $false }
        }
        $venvExists = Test-Path -LiteralPath (Join-Path $ProjectRoot '.venv')
        # Never replace a user-created environment. Only resume our own interrupted install.
        if ($venvExists -and -not $managedEnvironment) {
            throw 'An existing .venv has incompatible packages or belongs to another location. Preserve it by renaming it, or run start.cmd from a fresh clone. This launcher will not replace it.'
        }
        $uv = Get-LocalUv
        $flavor = 'cpu'
        if ($Device -eq 'cuda' -or ($Device -eq 'auto' -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue))) { $flavor = 'cu128' }
        $stamp = @{managed_by = 'fall-detect-bootstrap-v1'; root = $ProjectRoot; fingerprint = (Get-SetupFingerprint $flavor); python = $PythonVersion; flavor = $flavor; phase = 'installing'}
        $stamp | ConvertTo-Json | Set-Content -LiteralPath $StampPath -Encoding UTF8
        if (-not $venvExists) {
            Write-Host "[2/4] Installing project-local Python $PythonVersion..."
            Invoke-Checked -Executable $uv -CommandArguments @('--system-certs', 'python', 'install', $PythonVersion)
            Invoke-Checked -Executable $uv -CommandArguments @('--system-certs', 'venv', '--python', $PythonVersion, '.venv')
        } else {
            Write-Host "[2/4] Reusing the managed project Python $PythonVersion..."
            & $PythonExe -B -c 'import sys; sys.exit(0 if sys.version_info[:3]==(3,11,16) else 1)' 2>$null
            if ($LASTEXITCODE -ne 0) { throw 'The existing managed environment has a different or incomplete Python. Preserve it by renaming .venv, then retry.' }
        }
        Write-Host "[3/4] Installing pinned packages ($flavor). The first run can take several minutes..."
        Invoke-Checked -Executable $uv -CommandArguments @('--system-certs', 'pip', 'install', '--python', $PythonExe, 'torch==2.8.0', 'torchvision==0.23.0', '--index-url', "https://download.pytorch.org/whl/$flavor")
        Invoke-Checked -Executable $uv -CommandArguments @('--system-certs', 'pip', 'install', '--python', $PythonExe, '-r', 'requirements/core-tested.txt', '-r', 'requirements/pose-download-tested.txt')
        Invoke-Checked -Executable $uv -CommandArguments @('--system-certs', 'pip', 'install', '--python', $PythonExe, '--no-deps', '-e', '.')
        if (-not (Test-ProjectEnvironment)) { throw 'Installed packages failed the environment check.' }
        $stamp.phase = 'ready'
        $stamp | ConvertTo-Json | Set-Content -LiteralPath $StampPath -Encoding UTF8
    } elseif (-not $CheckOnly) {
        Write-Host '[1-3/4] Existing project environment is ready; skipping installation.'
    }

    $resolvedDevice = $Device
    if ($Device -ne 'cpu') {
        $cudaProbe = @'
import sys,torch
try:
    assert torch.cuda.is_available()
    x=torch.ones((2,3),device='cuda')
    torch.testing.assert_close((x@x.T).cpu(),torch.full((2,2),3.0))
except Exception:
    sys.exit(1)
'@
        & $PythonExe -B -X utf8 -c $cudaProbe 2>$null
        if ($LASTEXITCODE -eq 0) { $resolvedDevice = 'cuda' }
        elseif ($Device -eq 'cuda') { throw 'CUDA is not usable in this environment. Check the NVIDIA driver, or run start.cmd -Device cpu.' }
        else {
            $resolvedDevice = 'cpu'
            if (-not $CheckOnly) { Write-Host 'CUDA is not available; using CPU.' }
        }
    }
    $launchArguments = @('-B', '-X', 'utf8', 'scripts/quickstart.py', '--device', $resolvedDevice)
    if ($SetupOnly) { $launchArguments += '--setup-only' }
    if ($CheckOnly) { $launchArguments += '--check-only' }
    if ($NoOpen) { $launchArguments += '--no-open' }
    if ($InputPath) { $launchArguments += @('--input', $InputPath) }
    if ($OutputPath) { $launchArguments += @('--output', $OutputPath) }
    if (-not $CheckOnly) { Write-Host '[4/4] Preparing model resources and launching the project...' }
    Invoke-Checked -Executable $PythonExe -CommandArguments $launchArguments
    exit 0
} catch {
    [Console]::Error.WriteLine("ERROR: $($_.Exception.Message)")
    exit 1
}
