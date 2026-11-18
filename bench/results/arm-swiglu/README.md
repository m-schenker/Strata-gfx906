# Community benchmark on Radeon Instinct MI50 32GB

Measured on 2026-11-18 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Radeon Instinct MI50 32GB, 32 GiB VRAM
- CPU: AMD Ryzen 5 5600GT with Radeon Graphics (6 cores, 12 threads); installed RAM: 34.6 GiB; storage: sda     GIGABYTE GP-GSTFS31256GTND sata; nvme0n1 CT500P1SSD8                nvme
- OS: Linux 6.12.0-211.66.1.el10_2.x86_64; GPU driver: 6.12.0-211.66.1.el10_2.x86_64; CUDA (driver supports up to): not read
- Strata: project 0.1.61, commit 633568e0; engine 0.1.61
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 47.9 (35.2-65.1) | 0.52 (0.45-0.73) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 255.5 (240.1-256.1) | not measured | 15.76 (15.72-16.77) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 373.1 (372.8-378.6) | not measured | 43.08 (42.45-43.11) |
| agent session, follow-up turns | 14505 (14434-14524) at the last turn | see below | up to 64 | 3 | see below | 50.0 (23.5-57.0) | 17.17 (12.84-21.27) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.7 % (70.7-70.8); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 13629 (13628-13630) new tokens of the files are read once); first turn (nothing cached) first token: 10.62 s (9.43-10.72).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 64.8 | 9.433 | 33.7 |
| 1 | 4735 | 626 | 4109 | 193.5 | 21.273 | 46.6 |
| 2 | 6268 | 4798 | 1470 | 111.8 | 13.182 | 45.7 |
| 3 | 9496 | 6331 | 3165 | 160.4 | 19.764 | 45.5 |
| 4 | 11567 | 9536 | 2031 | 119.2 | 17.079 | 51.7 |
| 5 | 14434 | 11576 | 2858 | 166.8 | 17.174 | 23.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 536 s.
