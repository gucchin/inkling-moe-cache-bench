# Step 4-6 of PLAN-next, second batch. ASCII only.
# 1) redo the 7296 MiB / CPU share 0.95 pair (other apps were closing during the first one)
# 2) CPU share 0.5 at 7296 MiB
# 3) restart spread: the best shared config, three more server starts
# 4) workloads with traces: off (0.95), on (0.95), on (0)
$ErrorActionPreference = 'Continue'
$work = Split-Path -Parent $PSScriptRoot
Set-Location $work

powershell -ExecutionPolicy Bypass -File scripts\run_next.ps1 -Csv results\next\shared_redo_raw.csv -Sizes 7296 -Shares 0.95
powershell -ExecutionPolicy Bypass -File scripts\run_next.ps1 -Csv results\next\shared_share05_raw.csv -Sizes 7296 -Shares 0.5

$base = @('-ngl', '99', '-cmoe', '-ub', '32', '--moe-cache-mib', '7296', '-lm', 'none')
$env:LLAMA_MOE_CACHE_SHARE = '1'
foreach ($k in 1..3) {
    python scripts\bench.py --label "N-on-7296-cpu0-restart$k" --csv results\next\restart_raw.csv -- @base
}

$env:LLAMA_MOE_CACHE_SHARE = ''
python scripts\run_trace.py --tag n_off -- @base --moe-cache-cpu-share 0.95
$env:LLAMA_MOE_CACHE_SHARE = '1'
python scripts\run_trace.py --tag n_on -- @base --moe-cache-cpu-share 0.95
python scripts\run_trace.py --tag n_on0 -- @base
$env:LLAMA_MOE_CACHE_SHARE = ''
Write-Host 'RUN_NEXT2 DONE'
