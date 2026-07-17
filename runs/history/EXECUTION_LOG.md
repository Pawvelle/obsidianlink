# Execution Log

This is the English historical validation record for the original local
development work. Commands, acceptance results, artifact paths, hashes, and
safety conclusions are retained; the source runs remain in `runs/history/`.

## 2026-07-16 — Phase 0

### Completed

- Host: Apple M4 / arm64 / 16 GB / macOS 26.5.2.
- Shallow-cloned MineRL into `vendor/minerl` at
  `dev@cdeae668c2f334e3c9117adf651b5a94436b45f8`.
- Reviewed the README, requirements, setup, BASALT environments,
  action/observation interfaces, and MCP build scripts.
- Created the `mc-agent` Conda environment with Python 3.10.20 and arm64
  OpenJDK 1.8.0_472.
- Installed MineRL-compatible dependencies including Gym 0.23.1, NumPy 1.23.5,
  OpenCV 4.8.1.78, and pyglet 1.5.27, plus Qwen dependencies PyTorch 2.13.0,
  torchvision 0.28.0, Transformers 4.57.6, and Accelerate 1.14.0.
- `pip check` reported `No broken requirements found.`
- An unsandboxed MPS check confirmed `mps_built=True` and
  `mps_available=True`, and successfully created an `mps:0` tensor.
- Qwen3-VL completed an MPS smoke inference with a synthetic red image:
  13.607 s load, 2.940 s inference, output `Red`.
- Downloaded model commit `89644892e4d85e24eaac8bacfd4f463576704203`:
  12 files, 4,266,648,961 bytes. `model.safetensors` was 4,255,140,312 bytes
  with verified SHA-256
  `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`.
- Local MineRL patches upgraded LWJGL 3.2.1 to 3.3.1 and added
  `-XstartOnFirstThread` to JVM startup.

### Safety gate

The MineRL installation script clones `Hexeption/MCP-Reborn@1.16.5-20210115`
from the network and then executes third-party Gradle build code. The initial
unsandboxed request was declined for that risk; no bypass was used. Subsequent
Phase 1 work began only after explicit user approval.

## 2026-07-16 — Documentation-plan calibration

- Reviewed the official Installation, Versions, Environment, BASALT, FAQ,
  Performance, and First Agent pages.
- The `latest` documentation was labelled 0.4.0 while mixing 1.x content. The
  plan therefore established the authority order: pinned source first,
  documentation for interfaces and task semantics.
- Confirmed that v1.x compiles Minecraft during installation and needs JDK 8.
  This headed macOS host does not use the Linux-headless xvfb/VirtualGL route.
- Confirmed FindCave uses 3,600 steps, starts with an empty inventory, ends
  with `ESC=1` after a cave is found, forbids digging straight down, and has no
  predefined BASALT reward success signal.
- Added non-blocking planner/step loops, termination evidence, and manual
  review requirements to `MASTER_PLAN.md` v1.0.1 without changing the overall
  sequence.

## 2026-07-16 — Phase 1

### Build and compatibility

- The user explicitly approved execution of MCP-Reborn third-party Gradle code.
- Audited and pinned `Hexeption/MCP-Reborn@1.16.5-20210115` at commit
  `1e71be5bd4c49bc4d6ab0ee559c31b298b7697a3`.
- Pinned `ForgeGradle:3.0.+` to verified `3.0.190`; the build stages used
  ForgeGradle 3.0.190 and the MineRL patch-specified 4.1.10 respectively.
- Preserved and SHA-256-verified MCP `output.zip` as
  `65669e3413666b4634f00f876efbfc36bf5659b43078068caa9c4e32158fd139` to
  prevent drift in the 2021 remote patch input.
- Applied Apple Silicon fixes: LWJGL 3.3.1, `-XstartOnFirstThread`, removed the
  window-icon call, disabled `checkGlfwError`, and generated `schemas.index`
  relative to the project. Added `natives-macos-arm64` for LWJGL core, GLFW,
  jemalloc, OpenAL, OpenGL, STB, and tinyfd; Maven Central HEAD checks returned
  HTTP 200.
- `setup_mcp.sh` and `patch_mcp.sh` gained fail-fast behavior, clone retries,
  non-interactive patching, and hash gates. `clean build shadowJar` succeeded;
  the macOS arm64 MineRL wheel was built and installed into `mc-agent`.

### Artifacts and acceptance

- Wheel: `artifacts/phase1/minerl-1.0.2-cp310-cp310-macosx_11_0_arm64.whl`,
  about 1.3 GB; SHA-256
  `a208729207ad459e4bc6d4b5f441b9eb2f6ead633cec2dee2025d80c03108f56`.
- First real frame: `artifacts/phase1/findcave-reset.png`, 640 × 360;
  SHA-256 `4b37801a16f1d090d7237dd2c58a7fad77f3f728f3bf6d9591807ed3cfd89a5e`.
- Logs: `logs/phase1/minerl-build-arm64-natives.log` and
  `logs/phase1/findcave-real-smoke.log`.
- MineRL 1.0.2 imported and `MineRLBasaltFindCave-v0` registered. The supported
  `MineRLNavigate-v0` fake environment reset and stepped once. Its `close()`
  has the upstream `NotImplementedType.client_socket` issue; the test records
  compatibility only for that exact exception.
- A real FindCave environment reset, saved its first frame, ran 10 steps, and
  closed. `pov` was `[360, 640, 3]`, with no early `done`. No Minecraft or
  mcprec process remained.
- `pip check` only noted MineRL metadata requiring `typing>=3.6.6`; Python 3.10
  includes `typing`, and installing its obsolete backport would shadow the
  standard library. The note did not affect import, fake-environment, or real
  environment acceptance.

**Conclusion: passed.** Work advanced to Phase 2 only.

## 2026-07-16 — Phase 2

- Loaded the model only from `models/Qwen3-VL-2B-Instruct` at commit
  `89644892e4d85e24eaac8bacfd4f463576704203`, with `local_files_only=True`.
  Device and precision remained Apple MPS / FP16 and generation used
  `do_sample=False`.
- The initial 448 × 448 / `max_new_tokens=64` setup passed 10/10 JSON outputs
  but left only about 2.14 GB of available system memory. The pinned final
  configuration is 336 × 336 / `max_new_tokens=48`, without quantization or a
  model change. The action schema is `action/yaw/pitch/duration_ticks` and is
  strictly parsed without connecting MineRL or executing actions.
- A synthetic red image returned `Red` in 0.838 s. The Phase 1 real frame was
  inferred 10 times with no OOM; strict JSON and field/type/enum/range checks
  passed 10/10. Every output was
  `{"action":"look","yaw":0,"pitch":0,"duration_ticks":10}`.
- Real-frame inference averaged 3.845 s (median 3.943, P90 4.522, min 2.928,
  max 4.540). Model load took 9.069 s; peak RSS was 4,342,857,728 bytes and
  MPS driver peak was 5,036,163,072 bytes. Available memory bottomed at
  2,183,233,536 bytes, above the 2 GiB gate but with limited margin.
- Benchmark script: `scripts/benchmark_qwen_phase2.py`, SHA-256
  `4dbdc4d6a9d521d890b1f1fee750b09c0266ae9aa7f22af3071fd307734ddf48`.
  Result and final-log SHA-256:
  `0d39f72bfc65bb70fdfae2da9377040a0348c94b7aa15da71def1e18ec3f0f0e`.

**Conclusion: passed.** Phase 2 did not implement or execute MineRL actions.

## 2026-07-16 — Phase 3

- Added `MineRLEnvAdapter` for the legacy Gym 0.23.1 reset/step API, validated
  `pov` as `uint8 (360,640,3)`, and enforced creator-thread-only
  `reset/step/close` calls.
- Added a strict macro-action protocol: one JSON object only; allowlisted
  `wait/look/turn/move_forward`; unknown fields, Markdown, `ESC`, non-finite
  values, and invalid types fall back to a one-tick no-op. Fixed limits are
  duration 1–40 ticks and pitch/yaw -30°–30°, with a 160-character reason cap.
- `MacroExecutor` builds from `action_space.no_op()`, forces `ESC=0`, sends
  camera delta only on a macro action's first tick, and permits sprint only for
  `move_forward`. Parser and executor both clamp direct malformed actions.
- Added a capacity-one `LatestActionMailbox`, a thread-safe watchdog, and
  flush-on-write `config.json`, `events.jsonl`, and `metrics.json` logs.
- All 16 standard-library tests passed, including malformed input, direct-action
  clamping, mailbox, thread ownership, legacy Gym normalization, structured
  logging, and watchdog behavior. After aborting a 40-tick forward action, the
  next `next_tick()` immediately became a no-op.
- A real 20-tick preflight passed, with different initial/final screenshot
  hashes. The 500-tick acceptance command was
  `PYTHONPATH=src python -m mc_agent.cli phase3-smoke --ticks 500` in
  `mc-agent`; it alternated 10-tick stationary look/wait actions without Qwen.
  The run at `artifacts/phase3/runs/20260716-150715` completed 500/500 ticks,
  had reward 0.0, took 34.822 s including Minecraft startup/shutdown, and
  achieved 14.359 ticks/s. All 500 ticks had
  `ESC/attack/forward/jump/sprint=0` and `done=false`; no process remained.
- Key hashes: `events.jsonl`
  `50cbbf5698184c0ae4fe594f53d2bbe95e8f5a6aec92abb381813e6e2321c6b9`,
  `metrics.json`
  `8d07e2f712ac27e1576a04dd71e08b15edf35f25836eea056304bf0a7a10f35c`.

**Conclusion: passed.** Qwen integration was deferred to Phase 4.

## 2026-07-16 — Phase 4 MVP

- Kept Qwen3-VL in an independent daemon worker with capacity-one
  observation/decision mailboxes. Model output continued through the strict
  parser, allowlist, numeric clamps, and `MacroExecutor`; source had no
  `eval`, `exec`, shell, or subprocess execution path for model output.
- The first 5 × 800 run at `artifacts/phase4/runs/20260716-153300` completed
  but had no valid decisions in episodes 2, 4, and 5 (`accepted=false`). Queue
  clearing could not cancel in-flight inference, so stale decisions crossed an
  episode boundary and were discarded.
- Added `idle`, `wait_until_idle()`, and a generation-based
  `begin_episode()` barrier to `QwenPlannerWorker`. It invalidates the old
  generation, clears queued observations, waits for the single in-flight
  inference, clears stale state, and only then allows reset/submit. The running
  MineRL loop remains non-blocking.
- A controllably blocking fake-planner test confirmed barrier behavior. A later
  run removed stale decisions but was still too fast after warm-up. A monotonic
  20 Hz rate limiter was added without reading planner state or waiting on
  Qwen, and acceptance required at least one `accepted_decision` per episode.
- The final preflight at `artifacts/phase4/runs/20260716-230402` passed 800
  ticks with 11/11 accepted decisions, zero rejected or stale decisions,
  `ESC=0`, and three action signatures. Qwen inference averaged/maxed
  3.888/4.438 s; `env.step()` P95/max was 0.024/0.215 s, confirming decoupling.
- The final command was
  `/opt/anaconda3/bin/conda run -n mc-agent env PYTHONPATH=src python -m mc_agent.cli phase4-eval --episodes 5 --ticks 800 --observation-interval 40`.
  Run `artifacts/phase4/runs/20260716-230542` passed: five complete 800-tick
  episodes; 31/31 accepted, zero rejected/stale decisions; six action
  signatures; yaw -20, -15, 0, 10, 15, and 20; 4,000 ticks with `ESC=0`; and
  no residual Minecraft, mcprec, watcher, GradleStart, or planner process.
  Its SHA-256 is
  `b6dc5e830201eb4420aef868f3cd03b10a28c862788616546443950f3dd3563e`.
  Maximum Qwen latency was 22.376 s but each episode's maximum `env.step()`
  stayed at or below 0.407 s. Available memory bottomed near 1.348 GB.
- A 60-second five-chapter sampled replay was built from that run without AI
  interpolation: `artifacts/phase4/replays/20260716-230542-phase4-replay.mp4`.
  It is H.264 High, 1280 × 720, 30 fps, `yuv420p`, no audio, and faststart;
  SHA-256 `095ff2b9b1cecc5f5d9dffccbcd9cc07b7f52e5e1ee1c08f8872664398e8bc46`.

**Conclusion: passed.** The Qwen/MineRL closed-loop MVP passed replayability,
safety, non-blocking, logging, and cleanup gates; no Phase 5 behavior work was
performed at that point.

## 2026-07-16 to 2026-07-17 — Phase 5 variable studies

All studies kept the MineRL 20 Hz step loop, Qwen worker, episode barrier,
action allowlist, numeric clamps, and `ESC=0` safety boundary intact. The
documented commands used the fixed `mc-agent` Conda environment and
`PYTHONPATH=src`; all artifacts remain under `runs/history/`.

### Frame-change detection — retained

- Calibrated 100 Phase 4 decision frames by cropping the lower 60 HUD rows,
  sampling world pixels at 8-pixel intervals, and converting to grayscale.
  LOW means normalized mean absolute difference `<0.005` and changed-pixel
  ratio `<1%`, with a significant-pixel threshold of 20/255.
- `FrameChangeDetector` ran in shadow mode for A and supplied only a bounded
  `LOW/CHANGED` status to B. A three-seed, paired 5,101/5,102/5,103 run at
  `artifacts/phase5/frame-change-ab/20260716-235054` completed all six
  episodes: A had 14/14 accepted decisions and 28.57% invalid decisions; B
  had 11/11 and 27.27%. B reduced the rate by 4.55% at 3.07% additional
  inference cost; detection took at most about 2.53 ms.
- `accepted=true`, `advance_recommended=true`; summary SHA-256
  `409fed3a2a734018443b969e7a089847f64c585fcd2ec09efba4229d2b6895e0`.

### Turning-loop detection — not retained

- `TurningLoopDetector` becomes active after three accepted non-zero-yaw
  `look/turn` actions totaling at least 30°; forward, wait, zero-yaw, and
  pitch-only actions break the sequence. B receives a bounded prompt only when
  active.
- The three-seed run at `artifacts/phase5/turning-loop-ab/20260717-000810`
  completed six episodes with zero stale decisions and `ESC=0`. B never met
  the trigger condition, so the treatment was never exposed; yaw-only decision
  rate increased 14.29%. `accepted=true`, `advance_recommended=false`; SHA-256
  `07549b9e016c3ce8708f77976ae332f553bb394560e4aba858465eb64fcbe376`.

### Repetition penalty — not retained; decision acknowledgement retained

- `RepetitionDetector` recorded consecutive accepted identical action names.
  The B prompt prohibited choosing the preceding action but did not change the
  parser or executor.
- The audit found the worker could take a stale pre-action observation before
  the environment loop consumed its decision. An asynchronous decision-ack
  handoff was added: the worker waits in the background for acknowledgement,
  the environment continues to step, then acknowledgement clears the old
  observation before the worker accepts a new frame. The step loop never waits
  for Qwen.
- The three-seed run at `artifacts/phase5/repetition-ab/20260717-002634` had
  B treatment exposure on four decisions, but all still chose `look`; both
  action-name repetition rates were 100%. `accepted=true`,
  `advance_recommended=false`; SHA-256
  `ef065c9d0ee8e1d7c7260d2d2c26e01bae8f181f9017f05b13b1528ec00c6075`.

### Safe recovery macro — retained

- When a parsed accepted action was `wait` or zero-angle `look/turn`, B
  replaced it with a one-tick camera-only `look`, alternating yaw +20°/-20°.
  It kept pitch 0 and `attack/jump/sprint/ESC=false`, passed the normal clamps,
  and never replaced meaningful model actions.
- The three-seed run at `artifacts/phase5/recovery-ab/20260717-004851` applied
  all 7/7 B recovery opportunities safely. Executed invalid-action rate fell
  from 38.46% to 0%, 3/5 observable follow-ups were meaningful, low-change and
  no-op-tick rates improved, and inference cost rose 4.76%. No unsafe recovery
  or meaningful action overwrite occurred. `accepted=true`,
  `advance_recommended=true`; SHA-256
  `2a617055c35ac329f540d9dd1f14689dcfa8d218ad73360b6e95be971a5d5dde`.

### Orientation summary — not retained

- `OrientationMemory` accumulated actual executed yaw from zero on reset,
  wrapped to `[-180,180)`, and bucketed at 20°. B received at most three recent
  sampled views with `LOW/CHANGED` status and an adjacent lower-visit
  suggestion; A's prompt remained byte-for-byte the retained baseline.
- In `artifacts/phase5/orientation-ab/20260717-010559`, B read the summary for
  11 decisions and reduced revisit rate from 80.70% to 75.44%, below the 10%
  predeclared gate. It did not produce `move_forward`. `accepted=true`,
  `advance_recommended=false`; SHA-256
  `cfabd2417fd204fbc3a89187f3075b7eed46cf32f7ddd2580e7e1ce88b813d49`.

### Hierarchical prompt — not retained

- B organized the same visual decision as observe left/center/right, assess
  center risk, then choose an action. The final single-seed preflight tightened
  the action requirement to use 6- or 16-tick `move_forward` unless a concrete
  center danger existed.
- B reduced raw model invalid decisions from 70% to 40%, but low-change rate
  worsened from 26.32% to 36.84% and neither group moved forward. The preflight
  at `artifacts/phase5/hierarchical-ab/20260717-011940` was
  `accepted=true`, `advance_recommended=false`, SHA-256
  `1fd79e4e3b97e0b34096312038d75633a35e23a6367e9da9845d2a664de05ced`.

### Constrained forward-probe recovery — not retained

- B replaced a semantic no-op with exactly one `move_forward` tick only after
  two consecutive LOW observation windows. Its camera delta was zero and
  `attack/jump/sprint/ESC=false`; each trigger consumed the LOW streak.
- The single-seed preflight at
  `artifacts/phase5/forward-probe-ab/20260717-152510` executed one safe
  forward tick, whose next observation was `CHANGED`, but B's no-op tick rate
  (99.50%) was higher than A's (99.25%). It was therefore not advanced to
  three seeds: `accepted=true`, `advance_recommended=false`, SHA-256
  `7d381b5c8c9b818d4f65d1d18b61c4b153ee707b93b69b62e54d487c297343b4`.

**Phase 5 conclusion:** frame-change feedback, safe camera recovery, and the
asynchronous decision-ack correctness fix were retained. The fixed-budget runs
still did not establish a material reduction in being stuck or reliably produce
forward movement, so Phase 5 acceptance did not pass and Phase 6 did not begin.

## 2026-07-17 — Workspace consistency and live observation

- Updated `MASTER_PLAN.md` to use the current workspace path
  `/Users/joey/Desktop/Projects/mc-agent` rather than the former Documents path.
  Pinned dependencies, architecture, Phase 5 state, and the next step did not
  change.
- Moved `OrientationMemory`, `OrientationState`, and `OrientationView` from
  `mc_agent.perception` to the planned `mc_agent.memory` module and left a
  compatibility forwarder at the old import path. The outer repository was
  initialized on `main` and committed as
  `cdcc8742badc46d673ee7adfd0fb6a396c4836ca`; no remote was configured or
  pushed. The outer `.gitignore` excludes `/vendor/minerl/`, leaving the nested
  repository and its upstream origin intact.
- Regression checks completed 53/53 tests without launching Minecraft,
  rebuilding MineRL, or downloading a model.
- Added optional `--watch` to render MineRL's existing `render(mode="human")`
  output in a `MineRL Render` first-person window. It is disabled by default,
  runs only on the environment owner thread after reset and steps, and does not
  alter Qwen, the action protocol, or loop concurrency. Static checks passed,
  `--help` listed the flag, and the unit suite passed 43/43.
- The visible run at `runs/episodes/20260717-213029` completed 800/800 ticks:
  7/7 accepted decisions, six model forward decisions, 46 forward ticks, zero
  stale decisions, `ESC=0`, no planner error, and `accepted=true`. Rendering
  worked through reset and every step, then shut down cleanly.
