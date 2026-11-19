#!/usr/bin/env python3
"""A fast prompt/decode probe against a running Strata server: one request, no suite, the engine's own timings.

    bench/gfx906_probe.py --url http://127.0.0.1:8095 --api-key "$(cat run/state/api-key)" --words 2900 --reps 2
    bench/gfx906_probe.py --url ... --url2 http://127.0.0.1:8096 --words 2900     # two servers, one line apart

Why: a full `strata_bench` round costs 10-15 minutes and a fresh 92 GB model load, so an idea that only needs "does the
prompt path change speed" is better answered by one long request.  The numbers are the same block `strata_bench` reads
(`timings.prompt_per_second`, `timings.predicted_per_second`), so a probe that moves is worth a real round and a probe
that does not is answered.

The prompt is a word list of the size you ask for (the suite's own generator), sent fresh every rep so nothing is
reused from the engine's prompt cache.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from ab_engine import long_prompt            # noqa: E402  the same word list the suite uses


def one(base: str, key: str, words: int, new_tokens: int, timeout: float) -> dict:
    body = {"model": "strata", "stream": False, "max_tokens": new_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none",
            # a unique tag in front of every rep: otherwise the engine's prompt cache answers the repeat and the
            # prompt number is meaningless (it reuses the earlier request's tokens)
            "messages": [{"role": "user", "content": f"[probe {os.getpid()} {time.time():.6f} {words}] "
                                                     + long_prompt(words)}]}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": "Bearer " + key} if key else {})})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    wall = time.perf_counter() - t0
    t = d.get("timings") or {}
    return {"prompt_n": t.get("prompt_n"), "prompt_tps": t.get("prompt_per_second"),
            "decode_tps": t.get("predicted_per_second"), "wall_s": round(wall, 2)}


def run(label: str, base: str, key: str, words: int, reps: int, new_tokens: int, timeout: float) -> dict:
    samples = []
    for i in range(reps):
        s = one(base, key, words, new_tokens, timeout)
        samples.append(s)
        print(f"  {label} rep {i + 1}: prompt_n {s['prompt_n']}, prompt {s['prompt_tps']:.1f} tok/s, "
              f"decode {s['decode_tps']:.1f} tok/s, wall {s['wall_s']}s", flush=True)
    med = {k: statistics.median([s[k] for s in samples]) for k in ("prompt_n", "prompt_tps", "decode_tps")}
    print(f"{label}: prompt {med['prompt_tps']:.1f} tok/s, decode {med['decode_tps']:.1f} tok/s "
          f"({med['prompt_n']:.0f} prompt tokens, {reps} rep(s))", flush=True)
    return med


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8095", help="the server to probe")
    ap.add_argument("--url2", help="a second server, for a side-by-side line")
    ap.add_argument("--label", default="A", help="the first label (default A)")
    ap.add_argument("--label2", default="B", help="the second label (default B)")
    ap.add_argument("--api-key", default=os.environ.get("STRATA_KEY", ""))
    ap.add_argument("--words", type=int, default=2900, help="prompt size in words (2900 ~ 4 K tokens)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--new-tokens", type=int, default=8, help="decoded tokens (few: this probe is about the prompt)")
    ap.add_argument("--timeout", type=float, default=3600.0)
    a = ap.parse_args()

    a_med = run(a.label, a.url.rstrip("/"), a.api_key, a.words, a.reps, a.new_tokens, a.timeout)
    if a.url2:
        b_med = run(a.label2, a.url2.rstrip("/"), a.api_key, a.words, a.reps, a.new_tokens, a.timeout)
        if a_med["prompt_tps"]:
            gain = (b_med["prompt_tps"] - a_med["prompt_tps"]) / a_med["prompt_tps"] * 100
            print(f"\n{a.label2} vs {a.label}: {gain:+.1f} % prompt, "
                  f"{(b_med['decode_tps'] - a_med['decode_tps']) / a_med['decode_tps'] * 100:+.1f} % decode")
    return 0


if __name__ == "__main__":
    sys.exit(main())
