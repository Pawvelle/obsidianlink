# mc-agent Roadmap

`mc-agent` is a personal, local Minecraft Agent project. MineRL provides the
Minecraft 1.16.5 environment; Qwen3-VL-2B-Instruct reads the first-person view
and produces constrained JSON macro actions; a local executor safely converts
those actions into keyboard, mouse, and camera controls.

## Pinned technology stack

- Python 3.10.20 in the `mc-agent` Conda environment
- OpenJDK 8.0.472
- MineRL 1.0.2 / Minecraft 1.16.5 / Gym 0.23.1 / NumPy 1.23.5
- Qwen3-VL-2B-Instruct with pinned local weights, Apple MPS / FP16
- MineRL source in the independent nested repository `vendor/minerl`

Do not casually upgrade these versions during ordinary feature work or
structural maintenance.

## Phase 1 — Run Minecraft and MineRL

**Status: complete.**

- Apple Silicon MineRL build and patches are complete.
- `MineRLBasaltFindCave-v0` can reset, step, and close.
- Minecraft closes automatically with no residual process.

## Phase 2 — Let Qwen interpret Minecraft frames

**Status: complete.**

- Qwen loads from the pinned local model.
- The first Minecraft frame has been processed on Apple MPS.
- Output passes strict JSON action parsing; no model-generated code or command
  is executed.

## Phase 3 — Let the Agent control the player

**Status: complete.**

- The MineRL step loop and Qwen inference are decoupled, so slow model
  inference does not block the environment loop.
- The action allowlist, numeric clamps, watchdog, episode barrier,
  decision acknowledgement, lightweight memory, and structured logs are in
  place.
- Several replayable closed-loop episodes have completed, with `ESC` disabled
  by default.

## Phase 4 — Complete a simple task

**Status: in progress.**

The current task remains `MineRLBasaltFindCave-v0`.

Completed personal project milestones:

- The safe-forward prompt and acceptance gate are in place; zero-angle
  `look`/`turn` actions no longer count as meaningful model actions.
- The 2026-07-17 5 × 800-tick acceptance run passed: 25 meaningful Qwen
  forward decisions, 290 forward ticks, at least two meaningful action types
  per episode, `ESC=0`, and zero stale decisions.
- JSON actions now include fail-closed `cave_visible` handling. A claim is
  recorded as a cave candidate only when it provides evidence of darkness,
  rock, an opening, and direction.
- The historical dirt-wall false-positive sample and an 800-tick integration
  episode passed. Ordinary plains frames did not produce cave candidates and
  did not break the safe-forward loop.

Phase 4 is not complete: the 506 historical decision frames and newest replay
contain no trustworthy positive cave-entrance sample. Positive detection cannot
therefore be validated and FindCave cannot be marked successful. The next step
is only to obtain a genuine cave-entrance frame or episode, then review the
candidate decision and approach action. Do not add a paper-style A/B runner,
complex long-term memory, or another model backend.

Local evidence:

- `runs/phase4-forward-acceptance/20260717-205706/summary.json`
- `runs/phase4-cave-preflight/20260717-211041/summary.json`

## Long-term safety boundaries

1. Model output must pass JSON parsing, an action allowlist, and numeric
   clamping.
2. Only the thread running the step loop may operate the MineRL environment.
3. The Qwen worker must not block a running MineRL step loop.
4. Model weights and MineRL, Python, and JDK versions remain pinned.
5. `vendor/minerl` is an independent Git repository; the outer repository must
   not commit or rewrite its history.
6. MineRL Gradle builds execute third-party code and require explicit user
   approval before rebuilding.

Detailed historical validation records are in `runs/history/EXECUTION_LOG.md`.
Older run assets remain in `runs/history/artifacts/` and `runs/history/logs/`.
