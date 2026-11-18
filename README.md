# Strata for AMD gfx906 — a fork for MI50 / MI60 class cards

**This is a fork of [Strata](https://github.com/Niko1221/Strata) for the gfx906 series of AMD GPUs** — Instinct MI50,
Instinct MI60, Radeon Pro VII and Radeon VII (wave64, no matrix cores). Everything upstream ships is here; what this fork
adds is a container build that makes the engines actually run on these cards, and the source changes that requires.

## Where the changes live

**Every change is contained in the [`m-schenker/`](m-schenker/) folder.** The only file this fork touches outside that
folder is this `README.md`, which carries this notice. Nothing else in the source tree is edited: the container applies
its modifications as patch files and added source files at build time, from inside `m-schenker/gfx906/`, and the rest of
the tree is the upstream commit it was forked from. That keeps a rebase onto a newer Strata a matter of moving one
folder.

## What was changed

- **The Qwen backend was upgraded to Strata 0.2.15** (engine `0.1.61`, `qwen4exp`) and is built for gfx906 from the
  upstream sources as they are, with the handful of missing CUDA→HIP mappings carried as a patch.
- **The DeepSeek backend was ported to gfx906 as well, and is carried at version 0.2.13** — the last upstream release
  that still contained it. Upstream removed the DeepSeek engine from the tree in v0.2.14, so this fork vendors v0.2.13's
  `src/dsv41` engine (0.1.59), its `tools/dsv41` packer and its server package into `m-schenker/gfx906/`, wires it back
  into the 0.2.15 CMake build, and compiles it for gfx906 behind a compat layer: no matrix cores, no `mma.sync`, no
  `__ldcv`, and F32 forms of the low-bit expert kernels.
- Both engines live in one image and one container. The model's own GGUF metadata decides which engine serves it — a
  DeepSeek set goes to `strata-dsv41`, anything else to `strata` — so there is nothing to choose at run time.

The build and run guide is **[`m-schenker/gfx906/README.md`](m-schenker/gfx906/README.md)**; the full manual, including
how ROCm is obtained, every build argument and setting, what the build is verified to do and how to read a failure, is
**[`m-schenker/gfx906/REFERENCE.md`](m-schenker/gfx906/REFERENCE.md)**.

## AI disclaimer

**All changes in this fork were made with the help of DeepSeek-V4.1-Flash, run on an RTX Pro 6000 with Strata.** The
port, the patches, the container build and the documentation in the `m-schenker` folder were produced in that assisted
workflow, then built and exercised on real gfx906 hardware. Treat it as you would any contributed code: read the
patches, run the verification script, and take your own numbers — `m-schenker/gfx906/REFERENCE.md`, "Measuring on your
hardware", says how.

## Build and run, in short

All commands run from the repository root, the folder that contains `m-schenker/`.

```sh
# 1. build the image (both engines for gfx906; the long step, tens of minutes)
m-schenker/gfx906/build-image.sh                     # -> strata-gfx906:0.2.15

# 2. configure once: the .env goes BESIDE the compose file - that is where compose reads it
cp m-schenker/gfx906/.env.example m-schenker/gfx906/.env   # then set STRATA_MODEL_DIR / STRATA_DATA_DIR / MODEL_GGUF
                                                            # and, on rootless Docker, MEMLOCK (see below)

# 3. pack the model for the engine that will serve it (once per model)
docker compose -f m-schenker/gfx906/docker-compose.yml run --rm strata pack

# 4. serve it
docker compose -f m-schenker/gfx906/docker-compose.yml up -d
docker compose -f m-schenker/gfx906/docker-compose.yml logs -f       # "healthy" when the API answers

# 5. ask it something
KEY=$(docker compose -f m-schenker/gfx906/docker-compose.yml exec -T strata cat /data/state/api-key)
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"<model-id>","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

Before you start: a gfx906 card with a working `amdgpu` driver, Docker (rootless works), and the model as GGUF — the
container expects the host folder in `STRATA_MODEL_DIR` and mounts it read-only at `/models`. On rootless Docker also set
`MEMLOCK` in that `.env` to the user's hard limit in bytes (`ulimit -Hl` × 1024, usually `8388608`): the daemon cannot
raise it, and without it the container refuses to start with `error setting rlimits for ready process: error setting
rlimit type 8: operation not permitted`. Requirements, the full walkthrough and the sample configurations are in
[`m-schenker/gfx906/README.md`](m-schenker/gfx906/README.md).

---

*Everything below is the upstream Strata README, unchanged.*

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Run powerful AI on your own computer.</h3>
<p align="center">Chat, code and work with images, locally.</p>

<p align="center">
  <a href="#models">Models</a> · <a href="#install"><b>Install</b></a> ·
  <a href="#hardware">Hardware</a> · <a href="#tools">Tools</a> ·
  <a href="#help">Help</a> · <a href="#support-strata">Support</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Download Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?logo=githubsponsors&logoColor=fff" alt="Sponsor Strata on GitHub Sponsors"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Support Strata on Buy Me a Coffee"></a>
</p>

<p align="center"><sub><b>English</b> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

**Free, open source, and yours to run.** Use Strata on your desktop or with your favorite coding tools. No account needed.

## Models

We recommend **ISTA-DASLab first**, then **Unsloth** and other providers, where a compatible model fits.

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| Provider | Choices |
| --- | --- |
| **ISTA-DASLab · recommended** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S. [Coder: IQ1\_M](docs/MODELS.md#coder) for coding. |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF). Lower memory use but slower. |
| **Other options** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[Compare sizes, memory needs and compatibility](docs/MODELS.md).

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0. These are the currently documented packs for this model.

[Model guide and compatibility](docs/QWEN36.md).

</details>

<a id="the-other-sizes"></a>

Not sure what fits? [Choose a model for your computer](docs/MODELS.md#pick-by-memory).

## Install

<a id="option-a-the-strata-app"></a>

### Desktop app

**[Download Strata for Windows, Linux or Mac →](https://github.com/Niko1221/Strata/releases/latest)**

Install the app, choose a model, and start chatting. There is one installer for each system. On first start the app
downloads the engine for your graphics card. [Help with the installer](docs/INSTALL.md#the-app).

<sub>Files: `Strata-0.2.15-windows-x64-setup.exe` · `Strata-0.2.15-linux-x86_64.AppImage` · `Strata-0.2.15-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="Strata desktop app with a chat and live code preview side by side">
  <br><sub>Chat, files and a live preview in one window.</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### Terminal app

Prefer the terminal? Install Strata, then run **`strata launch`**.

<details open>
<summary><b>Install commands</b></summary>

**Linux / Mac**

```sh
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

On Windows, open a new terminal afterwards. Then:

```sh
strata launch
```

Follow the prompts to download and start a model. [Terminal guide](docs/TERMINAL_LAUNCH.md).

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="Strata terminal app showing model activity, speed, memory and recent requests">
  <br><sub>Watch your model and hardware while it runs.</sub>
</p>

<a id="build-from-source"></a>

[Build from source](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [Full install guide](docs/INSTALL.md)

<a id="what-you-need"></a>

## Hardware

| Supported hardware | Guide |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50 series** | [NVIDIA setup](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900, RX 9060 / 9070, AI PRO R9700** | [AMD setup](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · experimental** | [RX 6000 setup](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Intel setup](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 or newer** | [Mac setup](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Strix Halo setup](docs/STRIX_HALO.md) |
| **NVIDIA DGX Spark · Linux** | [DGX Spark setup](docs/DGX_SPARK_LINUX.md) |

<a id="using-it"></a>

## Tools

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**, and more through the OpenAI- and Anthropic-compatible APIs.

Run **`strata connect`** to see setup instructions for your tool. [Connection guide](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect).

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## Help

[All guides](docs) · [Troubleshooting](docs/TROUBLESHOOTING.md) · [Report a bug](https://github.com/Niko1221/Strata/issues/new/choose) ·
[Benchmarks](docs/COMMUNITY_BENCHMARKS.md) · [How it works](docs/HOW_IT_WORKS.md) · [Contribute](CONTRIBUTING.md)

## Support Strata

**Help keep local AI open to everyone.** A GitHub sponsorship or a coffee helps fund development and testing on more hardware.

<a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor_on_GitHub-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="GitHub Sponsors: sponsor Strata" height="48"></a>
<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: support Strata" height="48"></a>

### Loyal supporters

Thank you to everyone who supports Strata.

<!-- supporters:start (one line per public supporter, newest last; copy a line and change the GitHub name in all four places) -->
<a href="https://github.com/ryan4yin"><img src="https://github.com/ryan4yin.png?size=96" width="48" height="48" alt="ryan4yin" title="ryan4yin"></a>
<a href="https://github.com/themactep"><img src="https://github.com/themactep.png?size=96" width="48" height="48" alt="themactep" title="themactep"></a>
<a href="https://github.com/macintog"><img src="https://github.com/macintog.png?size=96" width="48" height="48" alt="macintog" title="macintog"></a>
<a href="https://github.com/bisonbet"><img src="https://github.com/bisonbet.png?size=96" width="48" height="48" alt="bisonbet" title="bisonbet"></a>
<a href="https://github.com/vizv"><img src="https://github.com/vizv.png?size=96" width="48" height="48" alt="vizv" title="vizv"></a>
<a href="https://github.com/kod88vn"><img src="https://github.com/kod88vn.png?size=96" width="48" height="48" alt="kod88vn" title="kod88vn"></a>
<a href="https://github.com/seoulworker"><img src="https://github.com/seoulworker.png?size=96" width="48" height="48" alt="seoulworker" title="seoulworker"></a>
<a href="https://github.com/jackyini"><img src="https://github.com/jackyini.png?size=96" width="48" height="48" alt="jackyini" title="jackyini"></a>
<a href="https://github.com/router0mail"><img src="https://github.com/router0mail.png?size=96" width="48" height="48" alt="router0mail" title="router0mail"></a>
<!-- supporters:end -->

[MIT license](LICENSE) · [Credits and model licenses](docs/HOW_IT_WORKS.md#credits) · [Privacy](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>For AI tools and automated readers: <a href="AGENTS.md">technical reference</a> · <a href="docs/AI_SETUP.md">setup instructions</a>.</sub>

**Sponsorships, hardware partnerships or collaborations:** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).

