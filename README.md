# mc-agent

[简体中文](README.zh-CN.md) | **English**

A Minecraft Agent for personal learning and long-term development. MineRL runs
Minecraft 1.16.5, Qwen3-VL-2B-Instruct interprets the first-person view and
generates constrained JSON macro actions, and a local executor safely sends
those actions to the environment.

The core loop is straightforward:

```text
Minecraft frame -> Qwen planning -> JSON validation and clamping -> MineRL action -> new frame
```

## Project layout

```text
mc-agent/
├── README.md
├── README.zh-CN.md
├── AGENTS.md
├── ROADMAP.md
├── environment.yml
├── model.lock.json
├── mc_agent/
│   ├── main.py       # Command-line entry point
│   ├── agent.py      # Agent main loop
│   ├── env.py        # MineRL lifecycle wrapper
│   ├── qwen.py       # Asynchronous Qwen visual planner
│   ├── actions.py    # JSON action protocol, executor, and watchdog
│   ├── memory.py     # Simple directional memory and frame-change detection
│   └── logger.py     # Replayable JSONL logs
├── scripts/          # Environment checks and smoke tests
├── tests/            # Core unit tests
├── models/           # Local model; not committed to the outer Git repository
├── runs/             # New run results and local history
└── vendor/minerl/    # Independent upstream MineRL repository
```

## Environment

The validated environment is pinned to:

- Conda environment: `mc-agent`
- Python 3.10.20 / OpenJDK 8.0.472
- MineRL 1.0.2 / Gym 0.23.1 / NumPy 1.23.5
- Qwen3-VL-2B-Instruct / Apple MPS / FP16

Activate the existing environment directly; do not reinstall MineRL or
download the model again:

```bash
conda activate mc-agent
python scripts/check_environment.py
```

`environment.yml` records the Python dependency versions. `model.lock.json`
is the sole configuration for the model repository, commit, and weight
checksums. Run `scripts/download_model.sh` only if the local weights are
actually missing.

The local Apple Silicon build of MineRL has already been verified. Do not run
`pip install ./vendor/minerl` or rebuild with Gradle casually: the build runs
third-party code and may overwrite the current working artifacts.

## Tests

Run the core unit tests:

```bash
python -m unittest discover -s tests -v
```

Run focused checks when needed:

```bash
python scripts/smoke_test_minerl.py --mode fake
python scripts/smoke_test_qwen.py
python scripts/smoke_test_minerl.py --mode real --steps 10
python scripts/smoke_test_agent.py --frame runs/smoke/findcave-reset.png
python scripts/smoke_test_agent.py --frame <frame.png> --after-forward
python scripts/smoke_test_agent.py --frame <non-cave.png> --expect-no-cave
```

## Run the Agent

Start with one episode to observe a run:

```bash
python -m mc_agent.main --watch --episodes 1 --ticks 800 --observation-interval 40
```

`--watch` opens a real-time first-person observation window named `MineRL
Render`. It only displays frames received by the Agent and does not take over
the keyboard or mouse. Press `Ctrl+C` in the terminal to stop the Agent. Omit
this flag when no observation window is needed.

Add episodes only when continuous replay is needed:

```bash
python -m mc_agent.main --episodes 5 --ticks 800 --observation-interval 40
```

For a genuine cave search, the stock BASALT task's three-minute mission limit
is usually too short. Use an explicit local time budget; this leaves the
pinned MineRL package and global Gym registration untouched, while preserving
the same FindCave world and action constraints:

```bash
python -m mc_agent.main --episodes 1 --ticks 18000 --mission-ticks 18000 \
  --observation-interval 40 --seed 45 --output-root runs/phase4-cave-search
```

`18000` ticks is 15 minutes of MineRL simulation at 20 Hz. The configured
mission limit is recorded in both the run summary and the episode config.

The default model remains the pinned 2B baseline. A larger local model must be
kept behind its own checked lock and selected explicitly, for example:

```bash
python -m mc_agent.main --model-lock model.experiments/qwen3-vl-4b.lock.json
```

Results are written to `runs/episodes/<timestamp>/` by default. Each episode
contains initial and final frames, decision frames, per-tick events, and
summary metrics. Qwen runs in a separate worker and never blocks the MineRL
step loop. When a bounded model macro finishes, its post-action frame starts
the next asynchronous inference immediately; fixed-interval frames remain as
a fallback. Before an episode changes, a barrier waits for old inference to
finish and clears stale observations and decisions.

The agent's restricted movement vocabulary includes forward progress plus
bounded `retreat`, `sidestep_left`, and `sidestep_right` escape macros. They
cannot press `ESC`, attack, or jump; water and low-progress guards use the
escape macros only to leave a visible local hazard. `ESC` is not present in
the model schema. The environment owner may emit exactly one local `ESC=1`
tick only after two independently validated cave frames in the same short-lived
target direction, separated by at least 12 real forward ticks.

Model actions include a fail-closed `cave_visible` field. A missing field is
treated as `false`. A decision frame is recorded as a cave candidate only when
the model explicitly reports a cave and its rationale includes evidence of
darkness, rock, an opening, and direction. The first validated frame establishes
a bounded target bearing for the next few decisions; a second matching validated
frame after real approach progress is the only path to the local completion tick.
All candidate evidence and any completion request still require manual review.

## Current scope

The Agent has passed the 5 × 800-tick safe-forward acceptance test, with
model-driven forward ticks and meaningful action variation. Cave negative
samples and run integration have also been verified, but no existing replay
contains a trustworthy positive cave-entrance sample. Therefore, finding a
cave is not yet complete. Continue Phase 4 in [ROADMAP.md](ROADMAP.md) by
validating only genuine cave positives and approach actions. Historical
validation records are in `runs/history/EXECUTION_LOG.md`; large historical
run artifacts stay local and are not committed to the outer Git repository.

`vendor/minerl` is an independent Git repository. The outer project does not
commit, delete, or rewrite its Git history.
