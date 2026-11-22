# Build upstream llama.cpp with CUDA (Ninja + MSVC + CUDA 12.1).
# ASCII only. Run from anywhere: powershell -ExecutionPolicy Bypass -File scripts\build.ps1
$ErrorActionPreference = 'Stop'

$work  = Split-Path -Parent $PSScriptRoot
$src   = Join-Path $work 'llama.cpp'
$build = Join-Path $src 'build'
$vs    = 'C:\Program Files\Microsoft Visual Studio\2022\Community'
$cuda  = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1'
$cmakeDir = Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
$ninjaDir = Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja'

# Import the MSVC x64 environment into this PowerShell session.
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}

$env:CUDA_PATH = $cuda
$env:PATH = "$cmakeDir;$ninjaDir;$cuda\bin;$env:PATH"

# CMake reads backslashes in cache values as escapes, so pass forward slashes.
$cudaFwd = $cuda -replace '\\', '/'

cmake --version
cmake -S $src -B $build -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DGGML_CUDA=ON `
    "-DCUDAToolkit_ROOT=$cudaFwd" `
    "-DCMAKE_CUDA_COMPILER=$cudaFwd/bin/nvcc.exe"
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed ($LASTEXITCODE)" }

cmake --build $build --config Release -j 16
if ($LASTEXITCODE -ne 0) { throw "cmake build failed ($LASTEXITCODE)" }

Write-Host 'BUILD OK'
