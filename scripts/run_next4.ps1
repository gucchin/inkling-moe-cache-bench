# PLAN-next step 5, follow-up: ubatch 64 keeps the prompt batches out of the cache. ASCII only.
# Checks: upstream (no sharing) with the same settings, VRAM overflow (mmap makes the shared-memory counter meaningful),
# and the three workloads with distinct prompts.
$ErrorActionPreference = 'Continue'
$work = Split-Path -Parent $PSScriptRoot
Set-Location $work
$csv = 'results\next\extra_raw.csv'

$env:LLAMA_MOE_CACHE_SHARE = ''
python scripts\bench.py --label 'N-off-4096-cpu0-ub64' --csv $csv -- -ngl 99 -cmoe -ub 64 --moe-cache-mib 4096 -lm none
python scripts\bench.py --label 'N-off-4096-cpu0.95-ub64' --csv $csv -- -ngl 99 -cmoe -ub 64 --moe-cache-mib 4096 --moe-cache-cpu-share 0.95 -lm none
$env:LLAMA_MOE_CACHE_SHARE = '1'
python scripts\bench.py --label 'N-on-4096-cpu0-ub64-mmap' --csv $csv -- -ngl 99 -cmoe -ub 64 --moe-cache-mib 4096
python scripts\bench.py --label 'N-on-2816-cpu0-ub64' --csv $csv -- -ngl 99 -cmoe -ub 64 --moe-cache-mib 2816 -lm none
python scripts\run_trace.py --tag n_on0_ub64 -- -ngl 99 -cmoe -ub 64 --moe-cache-mib 4096 -lm none
$env:LLAMA_MOE_CACHE_SHARE = ''
python scripts\run_trace.py --tag n_off0_ub64 -- -ngl 99 -cmoe -ub 64 --moe-cache-mib 4096 -lm none
Write-Host 'RUN_NEXT4 DONE'
