# Multi-step computer use benchmark: setup on this machine

Ticket: find which multi-step computer use benchmark runs on this machine with a local Qwen3.5-4B agent in PyTorch through Hugging Face `transformers`.
Candidates: OSWorld-Verified, AndroidWorld, and two lighter browser candidates (MiniWoB++ through `miniwob`, BrowserGym).

Markers:

- [M] measured on this machine with a read-only command.
- [S] stated by a primary source. The URL follows the claim.
- [I] inference by the author of this note.

No install, clone, pull, or download ran for this note.
No command touched the GPU.

## Machine checks

| Check | Command | Result |
|---|---|---|
| KVM device | `ls -l /dev/kvm` | `crw-rw----+ 1 root kvm 10, 232`. The device exists and has an ACL. [M] |
| KVM ACL | `getfacl /dev/kvm` | The ACL gives the user `rw-`. [M] |
| KVM access | `test -r /dev/kvm && test -w /dev/kvm` | yes. The user can read and write `/dev/kvm` without the `kvm` group. [M] |
| CPU virtualization | `grep -c -E 'vmx\|svm' /proc/cpuinfo` | 48 [M] |
| Groups | `id -nG` | The user is in `sudo`, `docker`, `lxd`. The user is not in `kvm`. [M] |
| Docker version | `docker --version` | Docker version 29.1.3 [M] |
| Docker without sudo | `docker ps` | It works. No container runs. [M] |
| Docker host | `docker info` | 24 CPUs, 45.89 GiB memory, storage driver overlayfs. [M] |
| Android tools on PATH | `which emulator adb` | Neither is on PATH. [M] |
| Android SDK | `ls ~/Android/Sdk` | An SDK exists with `build-tools`, `cmdline-tools`, `ndk`, `platform-tools`, `platforms`. `platform-tools/adb` exists. The `emulator` package, `system-images`, and any AVD do not exist. [M] |
| Browser | `which google-chrome chromedriver` | `/usr/bin/google-chrome` exists. `chromedriver` does not exist. [M] |
| Other VM tools | `which qemu-system-x86_64 VBoxManage vmrun` | None is on PATH. [M] |
| Free disk | `df -h /home` | 653 GB free of 1.8 TB on `/`. [M] |
| Free memory | `free -g` | 45 GB total, about 24 GB available while the current experiment runs. [M] |

## OSWorld-Verified

| Fact | Value |
|---|---|
| Install | Clone the repository, then `pip install -r requirements.txt`. Python 3.10 or later. `pip install desktop-env` gives the environment without the tasks. [S] https://github.com/xlang-ai/OSWorld |
| VM providers | VMware, VirtualBox, Docker, Modal, Daytona, AWS. [S] https://github.com/xlang-ai/OSWorld |
| KVM | The authors recommend KVM for the Docker provider. [S] https://github.com/xlang-ai/OSWorld/blob/main/desktop_env/providers/docker/DOCKER_GUIDELINE.md |
| Download | `Ubuntu.qcow2.zip`, 12.3 GB, from the Hugging Face dataset `xlangai/ubuntu_osworld`. The Docker provider downloads it on the first run into `./docker_vm_data`. [S] https://huggingface.co/datasets/xlangai/ubuntu_osworld/tree/main and https://github.com/xlang-ai/OSWorld/blob/main/desktop_env/providers/docker/manager.py |
| Other downloads | A Docker image that runs QEMU. The fetched docs do not name it. A Google Drive cache holds init-state files. [S] https://github.com/xlang-ai/OSWorld. The size of the Docker image: not found. |
| Sudo on the host | Not stated for the Docker provider. [I] Docker without sudo and KVM access already work here, so the host needs no sudo. |
| Number of tasks | 369 Ubuntu tasks. 361 when the 8 Google Drive tasks are excluded. [S] http://osworld-v1.xlang.ai/. The task count of OSWorld-Verified alone: not found. |
| Steps per task | The paper uses a limit of 15 steps. [S] https://arxiv.org/html/2404.07972. The verified leaderboard reports several step settings, for example 100 steps. [S] https://xlang.ai/blog/osworld-verified |
| Time per task | The median human time is 111.94 s per task. [S] https://arxiv.org/html/2404.07972. The agent time per task: not found. The authors state that a full run takes "10+ hours" without parallel AWS runs. [S] https://xlang.ai/blog/osworld-verified |
| Observation | `screenshot`, `a11y_tree`, `screenshot_a11y_tree`, `som`. The screen is 1920x1080. [S] https://github.com/xlang-ai/OSWorld/tree/main/mm_agents and https://arxiv.org/html/2404.07972 |
| Action | `pyautogui` Python code, or `computer_13` enumerated actions. [S] https://github.com/xlang-ai/OSWorld/tree/main/mm_agents |
| Custom agent | Implement `reset()` and `predict(instruction, obs)`, which returns `response, actions`. Import the agent in `run.py` or `scripts/python/run_multienv.py`. [S] https://github.com/xlang-ai/OSWorld |
| Qwen agent in repo | `mm_agents/qwen3vl_agent.py` calls DashScope or an OpenAI-compatible endpoint through `OPENAI_BASE_URL`. It emits `pyautogui` actions on a 0 to 999 relative grid. It keeps 4 history steps. [S] https://github.com/xlang-ai/OSWorld/blob/main/mm_agents/qwen3vl_agent.py. The file has no local `transformers` path. [I] A local agent needs a new `predict()` that calls the PyTorch model in process, or a local OpenAI-compatible server. |
| Licence | Apache-2.0. [S] https://github.com/xlang-ai/OSWorld |
| Qwen3.5-4B score | OSWorld-Verified 35.6. [S] https://huggingface.co/Qwen/Qwen3.5-4B. The step limit for that score: not found. |

## AndroidWorld

| Fact | Value |
|---|---|
| Install | Install Android Studio. Create an AVD named `AndroidWorldAvd`, Pixel 6, API level 33 (Tiramisu). Start it with `emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554`. Clone the repository, then `pip install -r requirements.txt` and `python setup.py install`. Python 3.11 or later. Install `ffmpeg`. [S] https://github.com/google-research/android_world |
| Docker option | Experimental: "has not been extensively tested". The run command uses `--privileged`. [S] https://github.com/google-research/android_world. The Dockerfile installs the SDK command line tools, the emulator, an x86_64 Google APIs image at API 33, and Python 3.11.3. [S] https://raw.githubusercontent.com/google-research/android_world/main/Dockerfile |
| KVM | Not stated in the README or the Dockerfile. [I] The x86_64 emulator needs KVM for usable speed. KVM works here. |
| Download | Android Studio or the SDK emulator package, plus the API 33 system image, from Google. Size: not found. The footprint is 2 GB memory and 8 GB disk. [S] https://arxiv.org/html/2405.14573 |
| Sudo on the host | Not stated. The README uses `sudo apt` for `ffmpeg`. [S] https://github.com/google-research/android_world. [I] The Docker route needs no host sudo here. |
| API keys | The README sets `OPENAI_API_KEY` and `GCP_API_KEY` for its sample agents. [S] https://github.com/google-research/android_world. [I] A local agent does not need them. |
| Number of tasks | 116 tasks across 20 apps, with random parameters. It also has 92 MobileMiniWoB++ tasks. [S] https://arxiv.org/html/2405.14573 |
| Steps per task | The step budget is about 2x the human step count of each task. [S] https://github.com/google-research/android_world. The average step count: not found. |
| Time per task | M3A takes 3.9 min per task on average. The text-only version takes 2.5 min. [S] https://arxiv.org/html/2405.14573. [I] These times use API models. A local 4B model sets its own time. |
| Observation | A full-resolution screenshot (2400x1080x3) and a UI tree from the accessibility service. [S] https://arxiv.org/html/2405.14573 |
| Action | JSON actions from `json_action.py`: tap, swipe, type, navigation buttons. `adb` executes them. [S] https://arxiv.org/html/2405.14573 |
| Custom agent | Subclass `EnvironmentInteractingAgent`. Implement `step()`. Return `AgentInteractionResult` with `done`. Add the agent to `_get_agent` in `run.py`. [S] https://github.com/google-research/android_world |
| Licence | Apache-2.0. [S] https://github.com/google-research/android_world |
| Qwen3.5-4B score | AndroidWorld 58.6. [S] https://huggingface.co/Qwen/Qwen3.5-4B. The agent setup for that score: not found. |
| Qwen2.5-VL-3B score | AndroidWorld_SR 90.8, MobileMiniWob++_SR 67.9. [S] https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct. [I] The 90.8 is not consistent with 35% for Qwen2.5-VL-72B in the report (https://arxiv.org/html/2502.13923). Do not trust it without a check. |

## MiniWoB++ (`miniwob` from PyPI)

| Fact | Value |
|---|---|
| Install | `pip install miniwob`. Version 1.1.0, wheel 1.9 MB, Python 3.10 or later. [S] https://pypi.org/project/miniwob/ |
| Browser | Chrome or Chromium, and a matching ChromeDriver on PATH. Selenium drives the browser. [S] https://miniwob.farama.org/content/getting_started/ and https://github.com/Farama-Foundation/miniwob-plusplus |
| On this machine | Chrome exists. ChromeDriver does not exist. [M] |
| HTML tasks | Bundled in the wheel: not found. [I] The 1.9 MB wheel probably holds the HTML, but verify. |
| Licence | MIT. [S] https://pypi.org/project/miniwob/ |
| Number of tasks | "over 100" environments. [S] https://github.com/Farama-Foundation/miniwob-plusplus. 125 task types. [S] https://arxiv.org/html/2604.07776 |
| Episode length | The default episode limit is 10 s of browser time (`EPISODE_MAX_TIME = 10000`). The reward decays with time when `time_proportional` is set. [S] https://raw.githubusercontent.com/Farama-Foundation/miniwob-plusplus/main/miniwob/html/core/core.js. The step count per task: not found. [I] Most tasks take 1 to 10 actions. [I] A slow local model can hit the 10 s limit. Check how the Gymnasium wrapper handles the limit before use. |
| Observation | `utterance`, `fields`, `screenshot` (RGB array), `dom_elements`. [S] https://miniwob.farama.org/content/observation_space/ |
| Action | 14 types, for example `CLICK_COORDS`, `CLICK_ELEMENT`, `TYPE_TEXT`, `PRESS_KEY`, `SCROLL_UP_COORDS`. [S] https://miniwob.farama.org/content/action_space/ |
| Custom agent | A Gymnasium loop: `reset()`, then `step(action)`. [S] https://github.com/Farama-Foundation/miniwob-plusplus. [I] The model runs in the same process. No server is needed. |
| Small Qwen scores | Not on the Qwen3.5-4B card. [S] https://huggingface.co/Qwen/Qwen3.5-4B. A 2026 paper reports MiniWoB 61.1 for Qwen3.5-4B and 63.2 for Qwen3.5-9B through BrowserGym and AgentLab. [S] https://arxiv.org/html/2604.07776. The observation mode for those scores: not found. |

## BrowserGym (MiniWoB through `browsergym-miniwob`)

| Fact | Value |
|---|---|
| Install | `pip install browsergym` or `pip install browsergym-miniwob`, then `playwright install chromium`. [S] https://github.com/ServiceNow/BrowserGym |
| Not PyPI only | The MiniWoB setup clones `miniwob-plusplus` at commit `7fd85d71a4b60325c6585396ec4f48377d049838` and sets `MINIWOB_URL` to a `file://` path. [S] https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/miniwob/README.md. [I] `playwright install chromium` downloads a browser from the Playwright CDN, not from PyPI. |
| Licence | Apache-2.0. [S] https://raw.githubusercontent.com/ServiceNow/BrowserGym/main/LICENSE |
| Benchmarks | MiniWoB, WebArena, WebArenaVerified, VisualWebArena, WorkArena, AssistantBench, WebLINX, OpenApps, TimeWarp, Knows. [S] https://github.com/ServiceNow/BrowserGym. [I] WebArena and WorkArena need servers or accounts, so only MiniWoB is light. |
| Episode length | Max steps: not found in the fetched pages. |
| Observation and action | Not found in the fetched README. [I] BrowserGym exposes an accessibility tree, DOM, and screenshot, with `click(bid)` style actions. Verify in the docs. |
| Custom agent | AgentLab runs agents on BrowserGym. The demo agent uses OpenAI `gpt-4o-mini`. [S] https://github.com/ServiceNow/BrowserGym. The integration steps: not found. |
| Small Qwen scores | Qwen3.5-4B 61.1 on MiniWoB through BrowserGym and AgentLab. [S] https://arxiv.org/html/2604.07776 |

## Setup cost comparison

| Candidate | Downloads | Needs sudo | Needs VM | Time per task | Blocker on this machine |
|---|---|---|---|---|---|
| OSWorld-Verified | Git repo, PyPI requirements, 12.3 GB `Ubuntu.qcow2.zip` from Hugging Face, a Docker image of unknown size [S] | No for the Docker route [I], because Docker and KVM work without sudo [M] | Yes: an Ubuntu QEMU VM inside Docker with KVM [S] | Human median 111.94 s [S]. Agent time not found. 15 to 100 steps [S] | No hard blocker found [I]. Costs: 12.3 GB plus the image, one VM of guest RAM while about 24 GB is free [M], and a new `predict()` for local `transformers` [I]. |
| AndroidWorld | Git repo, PyPI requirements, Android emulator package and API 33 image from Google (size not found), or a Docker build from the Dockerfile [S] | No for the Docker route [I]. The native route uses `sudo apt` for `ffmpeg` [S] | Yes: an Android emulator, 2 GB memory and 8 GB disk [S] | 3.9 min average for M3A [S] | The emulator package, a system image, and an AVD are absent [M]. The Docker route is experimental [S]. A new agent class is needed [S]. |
| MiniWoB++ (`miniwob`) | 1.9 MB wheel from PyPI [S], plus a ChromeDriver that matches the Chrome version [S] | No [I] | No | 10 s browser limit per episode by default [S]. Steps not found | ChromeDriver is absent [M]. The 10 s limit can fail a slow local model [I]. |
| BrowserGym MiniWoB | PyPI packages, a Playwright Chromium, a git clone of `miniwob-plusplus` [S] | No [I] | No | Not found | Not PyPI only, because of the git clone and the Playwright browser [S]. |

## Open questions

1. Which step limit and agent setup give the Qwen3.5-4B scores of 35.6 on OSWorld-Verified and 58.6 on AndroidWorld? The model card does not say.
2. How large is the OSWorld Docker image, and how much guest RAM does one VM take? The fetched docs do not say.
3. How long does one OSWorld task take with a local 4B model? No primary source gives an agent time per task.
4. Does the `miniwob` wheel include the HTML tasks, and does its Gymnasium wrapper apply the 10 s limit to agent think time?
5. Is the Qwen2.5-VL-3B AndroidWorld_SR value of 90.8 on its model card an error?
6. Can the long GPU experiment and a benchmark VM share the 45 GB RAM? About 24 GB is free now [M].
7. Which observation mode (screenshot or accessibility tree) does the harness need? The choice changes which agent each benchmark needs. This is a decision for the user.
