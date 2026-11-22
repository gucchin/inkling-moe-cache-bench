# inkling-moe-cache-bench

Measurements of the llama.cpp MoE expert cache with Inkling 2 Mini on a 12 GB GPU, and an experimental patch that lets the passes of a looped layer share their cache entries.

日本語の説明は下にあります。記事: https://bench.gucchinworks.com/posts/2026-11-inkling-moe-cache/ （1 回め）、https://bench.gucchinworks.com/posts/2026-11-inkling-moe-cache-share/ （2 回め、パッチ）

## What is here

| Path | Content |
|---|---|
| `patches/moe-cache-share.diff` | The patch, against llama.cpp `6792507b7f611a6fc68bfae3eeae85e88017fc60`. One file, `src/llama-moe-cache.cpp` |
| `results/` | First round: layer offload vs the expert cache as it is upstream, routing analysis |
| `results/next/` | Second round: the cache with and without the patch |
| `scripts/` | The scripts that produced the numbers (Windows, Python 3.12) |

## The patch in short

Inkling 2 Mini runs layers 6-17 three times per token. llama.cpp gives each pass its own layer slot (48 slots for 24 layers), and the slots of a looped layer point to the same weight tensors. The MoE cache treats every slot as a separate layer: an expert used by two passes takes two cache slots and is uploaded twice, and the log reports 28188 MiB of host experts for 15066 MiB of weights.

With the patch, a layer whose expert tensors are the same as those of an earlier layer uses the cache entries of that layer. It is off by default. Set the environment variable `LLAMA_MOE_CACHE_SHARE=1` to turn it on.

This is an experiment, not a proposal for upstream as it is. It was tested on one machine with one GPU and one model. It is not tested with several GPUs, tensor split, LoRA, or other models.

## Results

RTX 4070 Ti (12 GB), i7-13700F, 96 GB RAM, Windows 11, CUDA 12.1. Model: `ggml-org/Inkling-2-Mini-GGUF`, IQ4_NL (17.6 GB).
Server arguments: `-ngl 99 -cmoe -ub 32 -c 32768 --parallel 1 --jinja -t 16 -lm none --moe-cache-mib N [--moe-cache-cpu-share F]`.
Prompt of 1150 tokens, 512 generated tokens, median of 3 runs. Source: `results/next/shared.csv`.

Generation, tokens per second:

| Cache | upstream, CPU share 0 | patch, CPU share 0 | upstream, CPU share 0.95 | patch, CPU share 0.95 |
|---:|---:|---:|---:|---:|
| 7296 MiB | 27.52 | 39.35 | 27.88 | 34.27 |
| 5632 MiB | 24.53 | 34.82 | 25.49 | 31.51 |
| 4096 MiB | 20.55 | 29.84 | 22.83 | 28.37 |
| 2816 MiB | 17.42 | 24.49 | 21.22 | 25.39 |
| 1472 MiB | 13.53 | 18.06 | 18.62 | 21.35 |

Hit rate of the cache (ubatch <= 8), CPU share 0:

| Cache | upstream | patch |
|---:|---:|---:|
| 7296 MiB | 81.54% | 94.77% |
| 5632 MiB | 75.02% | 90.89% |
| 4096 MiB | 65.17% | 84.66% |
| 2816 MiB | 54.25% | 75.33% |
| 1472 MiB | 34.14% | 57.40% |

Prompt processing at 7296 MiB, CPU share 0: 57.5 -> 148.3 tokens per second.

Three kinds of input, 8 prompts each, 768 generated tokens, 7296 MiB (`results/next/trace_runs_*.csv`):

| Input | upstream, CPU share 0.95 | patch, CPU share 0.95 | patch, CPU share 0 |
|---|---:|---:|---:|
| Coding | 27.98 t/s, 74.22% | 34.48 t/s, 89.39% | 39.61 t/s, 94.31% |
| Japanese questions | 29.59 t/s, 77.44% | 35.77 t/s, 91.50% | 40.69 t/s, 94.95% |
| Long text summaries | 28.05 t/s, 74.72% | 33.66 t/s, 89.52% | 38.43 t/s, 94.19% |

Correctness (`results/next/correctness.md`): with CPU share 0, greedy output is identical with and without the patch for three prompts. With a CPU share the output differs, also between two upstream runs with different shares, because the CPU and the GPU round differently.

For reference, layer offload on the same machine and the same day (`-ngl 99 -ncmoe 13 -ub 16`) gives 20.47 tokens per second (`results/next/extra.csv`).

## Broken output with large batches (upstream, not the patch)

On this build, Inkling 2 Mini answers with one repeated character (`????...` or `GGGG...`) when host experts are computed through the large-batch path, where the experts a batch needs are copied to the GPU. The cache is only used for batches of at most 32 tokens, so this happens with `-cmoe` and no cache at ubatch 32 and above, with a cache at ubatch 64 and above, and with `-ncmoe 13` at ubatch 64 and above. It also happens without any VRAM overflow. The same `-cmoe -ub 64` server answers a 25-token prompt correctly. See `results/next/garbage_diag.csv`.

Garbage output repeats its routing, so the cache hit rate and the speed look excellent: `-ub 64` with a 4096 MiB cache reported 48 tokens per second and a 99% hit rate. Every row of `results/next/shared.csv` and `results/next/extra.csv` therefore has an `output_ok` column, from a check of the saved responses. One row of the first round (`B2`, `-ngl 99 -cmoe` with the default ubatch) was measured on garbage output; `B2-ub16` is the valid number for that configuration.

The cause is not investigated, and it is not checked against a newer llama.cpp.

## Reproduce

1. Clone llama.cpp, check out `6792507b7f611a6fc68bfae3eeae85e88017fc60`, apply `patches/moe-cache-share.diff` (`git apply`), build with CUDA (`scripts/build.ps1` is what was used here).
2. Download `Inkling-2-Mini-IQ4_NL.gguf` from `ggml-org/Inkling-2-Mini-GGUF` into `models/`.
3. `powershell -ExecutionPolicy Bypass -File scripts\run_next.ps1`, then `python scripts\summarize_next.py results\next\shared_raw.csv --out results\next\shared.csv`.

The scripts expect the layout `llama.cpp/`, `models/`, `scripts/`, `results/` in one folder. `scripts/llsrv.py` and `scripts/build.ps1` hold the CUDA and Visual Studio paths of this machine.

## Notes

- With the default ubatch (512) the compute buffer alone is about 9.9 GB on this model and the cache has no room. `-ub 32` is needed on a 12 GB GPU.
- Without host memory pinning (`-lm none` or `-lm mlock`) uploads are slow. All numbers above are with `-lm none`.
- With the patch the clock of a looped layer advances three times per token, so the half-life of the usage score (128 tokens) is about 43 tokens for these layers.
- `-lm dio` gives the same speed as `-lm none` (39.63). Without pinning (mmap) the patched cache gives 33.57.
- Not measured: output quality, MTP drafting.

## 日本語

RTX 4070 Ti（12GB）で Inkling 2 Mini を、本家 llama.cpp の MoE エキスパートキャッシュで動かした計測と、実験用のパッチです。

Inkling 2 Mini は層 6-17 を 1 トークンにつき 3 回通ります。本家のキャッシュは、この 3 回を別の層として扱うので、同じエキスパートがパスごとに別の枠に入ります。パッチは、同じ重みを指す層に枠を共有させます。環境変数 `LLAMA_MOE_CACHE_SHARE=1` で有効になります（既定は無効で、本家と同じ動き）。

7296 MiB のキャッシュ、CPU 分担 0 で、ヒット率は 81.54% から 94.77% に、生成は 27.52 から 39.35 トークン/秒になりました。数字はすべてこの PC での実測です。なお、この版の llama.cpp では、キャッシュを通らない大きいバッチ（`-cmoe` で ubatch 64 以上など）で出力が壊れます。速さの表には、出力が正常だった構成だけを載せています。試したのは 1 台・1 GPU・1 モデルだけで、本家にそのまま入れられる形ではありません。
