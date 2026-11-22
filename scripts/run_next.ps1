# Step 4 of PLAN-next: slot sharing off/on, cache size sweep, CPU share 0 and 0.95. ASCII only.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\run_next.ps1 [-Csv results\next\shared_raw.csv] [-Sizes 7296,5632,4096,2816,1472]
param(
    [string]$Csv = 'results\next\shared_raw.csv',
    [int[]]$Sizes = @(7296, 5632, 4096, 2816, 1472),
    [string[]]$Shares = @('0.95', '0')
)
$ErrorActionPreference = 'Continue'
$work = Split-Path -Parent $PSScriptRoot
Set-Location $work
New-Item -ItemType Directory -Force (Split-Path -Parent $Csv) | Out-Null

foreach ($size in $Sizes) {
    foreach ($share in $Shares) {
        foreach ($mode in @('off', 'on')) {
            if ($mode -eq 'on') { $env:LLAMA_MOE_CACHE_SHARE = '1' } else { $env:LLAMA_MOE_CACHE_SHARE = '' }
            $label = "N-$mode-$size-cpu$share"
            $srvArgs = @('-ngl', '99', '-cmoe', '-ub', '32', '--moe-cache-mib', "$size", '-lm', 'none')
            if ($share -ne '0') { $srvArgs += @('--moe-cache-cpu-share', $share) }
            python scripts\bench.py --label $label --csv $Csv -- @srvArgs
            if ($LASTEXITCODE -ne 0) { Write-Host "FAILED $label ($LASTEXITCODE)" }
        }
    }
}
$env:LLAMA_MOE_CACHE_SHARE = ''
Write-Host 'RUN_NEXT DONE'
