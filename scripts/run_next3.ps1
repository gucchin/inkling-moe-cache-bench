# PLAN-next steps 5-6: use of the freed VRAM, -lm dio, same-day layer offload reference. ASCII only.
$ErrorActionPreference = 'Continue'
$work = Split-Path -Parent $PSScriptRoot
Set-Location $work
$csv = 'results\next\extra_raw.csv'

# same-day reference: layer offload as in the first article
$env:LLAMA_MOE_CACHE_SHARE = ''
python scripts\bench.py --label 'N-b1-ncmoe13-ub16' --csv $csv -- -ngl 99 -ncmoe 13 -ub 16
python scripts\bench.py --label 'N-b1-ncmoe13-ub16-lmnone' --csv $csv -- -ngl 99 -ncmoe 13 -ub 16 -lm none

$env:LLAMA_MOE_CACHE_SHARE = '1'
# other load modes
python scripts\bench.py --label 'N-on-7296-cpu0-dio' --csv $csv -- -ngl 99 -cmoe -ub 32 --moe-cache-mib 7296 -lm dio
python scripts\bench.py --label 'N-on-7296-cpu0-mmap' --csv $csv -- -ngl 99 -cmoe -ub 32 --moe-cache-mib 7296
# smaller ubatch
python scripts\bench.py --label 'N-on-7296-cpu0-ub16' --csv $csv -- -ngl 99 -cmoe -ub 16 --moe-cache-mib 7296 -lm none
python scripts\bench.py --label 'N-on-7296-cpu0-ub8' --csv $csv -- -ngl 99 -cmoe -ub 8 --moe-cache-mib 7296 -lm none
# freed VRAM: smaller cache, larger ubatch (batches above 32 tokens do not use the cache)
python scripts\bench.py --label 'N-on-4096-cpu0-ub64' --csv $csv -- -ngl 99 -cmoe -ub 64 --moe-cache-mib 4096 -lm none
python scripts\bench.py --label 'N-on-4096-cpu0-ub128' --csv $csv -- -ngl 99 -cmoe -ub 128 --moe-cache-mib 4096 -lm none
$env:LLAMA_MOE_CACHE_SHARE = ''
Write-Host 'RUN_NEXT3 DONE'
