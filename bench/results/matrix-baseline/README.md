# Community benchmark on Radeon Instinct MI50 32GB

Measured on 2026-11-18 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Radeon Instinct MI50 32GB, 32 GiB VRAM
- CPU: AMD Ryzen 5 5600GT with Radeon Graphics (6 cores, 12 threads); installed RAM: 34.6 GiB; storage: sda     GIGABYTE GP-GSTFS31256GTND sata; nvme0n1 CT500P1SSD8                nvme
- OS: Linux 6.12.0-211.66.1.el10_2.x86_64; GPU driver: 6.12.0-211.66.1.el10_2.x86_64; CUDA (driver supports up to): not read
- Strata: project 0.1.61, commit 17e4a08f; engine 0.1.61
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-gyro-m-tq2_0
- Context: 200000 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 42.3 (32.0-52.6) | 0.66 (0.56-0.88) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 224.6 (220.0-252.4) | not measured | 17.93 (15.96-18.31) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 321.9 (318.6-332.7) | not measured | 49.94 (48.31-50.45) |
| agent session, follow-up turns | 14453 (14448-14499) at the last turn | see below | up to 64 | 3 | see below | 37.6 (23.2-52.5) | 13.36 (9.15-17.38) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.7 % (70.7-70.7); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 13627 (13627-13629) new tokens of the files are read once); first turn (nothing cached) first token: 8.07 s (7.84-9.21).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 78.1 | 7.837 | 26.8 |
| 1 | 4735 | 626 | 4109 | 246.8 | 16.697 | 36.6 |
| 2 | 6267 | 4798 | 1469 | 160.3 | 9.203 | 38.1 |
| 3 | 9495 | 6330 | 3165 | 222.1 | 14.291 | 33.8 |
| 4 | 11578 | 9547 | 2031 | 169.2 | 12.043 | 52.5 |
| 5 | 14448 | 11590 | 2858 | 214.6 | 13.361 | 47.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 499 s.
