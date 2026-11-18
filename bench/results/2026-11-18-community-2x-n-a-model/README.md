# Community benchmark on N/A + N/A

Measured on 2026-11-18 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: N/A + N/A, VRAM not read
- CPU: AMD Ryzen 5 5600GT with Radeon Graphics (? cores, 12 threads); installed RAM: 34.6 GiB; storage: sda     GIGABYTE GP-GSTFS31256GTND sata; nvme0n1 CT500P1SSD8                nvme
- OS: Linux 6.12.0-211.66.1.el10_2.x86_64; GPU driver: not read; CUDA (driver supports up to): not read
- Strata: project 0.1.61, commit 04292329; engine not read
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: model not read
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 36.3 (30.4-48.1) | 0.78 (0.65-0.84) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 392.3 (338.9-437.5) | not measured | 10.29 (9.23-11.90) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 386.7 (315.3-393.0) | not measured | 41.58 (40.91-50.98) |
| agent session, follow-up turns | 14505 (14505-14508) at the last turn | see below | up to 64 | 3 | see below | 37.7 (33.5-42.0) | 10.15 (6.83-14.49) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.6 % (70.4-70.7); earlier-turn tokens read again per session: 75 (2-142) (earlier turns that the cache did not hold; the 13628 (13628-13629) new tokens of the files are read once); first turn (nothing cached) first token: 3.40 s (3.39-18.40).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 182.1 | 3.386 | 34.6 |
| 1 | 4738 | 629 | 4109 | 298.7 | 13.843 | 35.1 |
| 2 | 6271 | 4731 | 1540 | 201.4 | 7.7 | 37.7 |
| 3 | 9499 | 6334 | 3165 | 266.9 | 11.914 | 34.9 |
| 4 | 11594 | 9562 | 2032 | 230.4 | 8.873 | 39.3 |
| 5 | 14508 | 11650 | 2858 | 243.5 | 11.796 | 38.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 421 s.
