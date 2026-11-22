# moe-cache-tune.py output

Command (run through `scripts/moe_cache_tune_win.py`, which only replaces how the server process is stopped on Windows):

```
python scripts\moe_cache_tune_win.py --markdown --out runs\tune -- llama.cpp\build\bin\llama-server.exe -m models\Inkling-2-Mini-IQ4_NL.gguf -cmoe -ngl 99 -t 16 -c 32768 --parallel 1 --jinja -ub 32
```

`-ub 32` was added to the PLAN command. Without it the tuner stops with `no room on CUDA0 for a cache larger than 1472 MiB` (log: `runs/tune_ub512.out`), because the default ubatch of 512 reserves a 9948 MiB compute buffer.

Note: "28188 MiB of experts" counts the loop layers once per pass (44 + 4 cache layers). The file holds 15066 MiB of experts (sum of the `ffn_*_exps` tensors in the GGUF).

## Output

1/6 1472 MiB, share 0: 8.2 t/s
2/6 1472 MiB, share 0.5: 10.0 t/s
3/6 1472 MiB, share 0.95: 17.7 t/s
4/6 7296 MiB, share 0: 23.6 t/s
5/6 7296 MiB, share 0.5: 23.0 t/s
6/6 7296 MiB, share 0.95: 29.5 t/s
`Inkling-2-Mini-IQ4_NL.gguf`: 28188 MiB of experts, CUDA0 11056 MiB free, cache up to 7352 MiB

| cache MiB | share | t/s | hits |
|---:|---:|---:|---:|
| 1472 | 0 | 8.2 | 37% |
| 1472 | 0.5 | 10.0 | 37% |
| 1472 | 0.95 | 17.7 | 37% |
| 7296 | 0 | 23.6 | 83% |
| 7296 | 0.5 | 23.0 | 82% |
| 7296 | 0.95 | 29.5 | 79% |

machine: uploads 10.2 GiB/s, CPU 45.2 GiB/s, CPU part 6.68 ms, overlap 15.1 ms/GiB, rest 12.72 ms per token, fit within 1.2%

predicted t/s, warm cache:
| cache MiB | 0 | 0.5 | 0.8 | 0.9 | 0.95 | 0.98 |
|---:|---:|---:|---:|---:|---:|---:|
| 1408 | 9.8 | 12.2 | 17.8 | 20.8 | 22.6 | **23.8** |
| 1472 | 10.0 | 12.4 | 18.0 | 21.0 | 22.8 | **24.0** |
| 2816 | 14.0 | 16.3 | 22.3 | 25.3 | **27.1** | 28.2 |
| 5632 | 23.3 | 24.3 | 30.3 | 33.1 | **34.5** | 35.4 |
| 7296 | 29.2 | 28.6 | 34.3 | 36.7 | **38.0** | 38.7 |

recommended: `--moe-cache-mib 7296 --moe-cache-cpu-share 0.95` (38.0 t/s, 29.2 with share 0)
