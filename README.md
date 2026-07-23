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

### Manual evidence collection

To explore a real FindCave world yourself without loading Qwen, start the
manual controller:

```bash
python scripts/manual_findcave.py --seed 101 --ticks 18000 --mission-ticks 18000
```

Focus the `MineRL Manual FindCave` window. `W`, `A`, `S`, and `D` toggle the
corresponding movement (opposite directions replace one another); `I`, `J`, `K`, and `L` look; Space performs one jump;
`R` toggles sprint; `C` saves a candidate frame; and `Q` or Esc ends the
session. Qwen is never loaded, attack and `ESC` are always disabled, forward
movement pauses when a center water hazard is detected, and all candidate
frames plus an action log are saved below `runs/manual-findcave/` for review.

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
bounded `retreat`, `sidestep_left`, and `sidestep_right` escape macros. It
cannot press `ESC` or attack. A jump may accompany `move_forward` only when the
model identifies a visible one-block ledge on an otherwise safe route; the
executor emits it for exactly one local tick. Water and low-progress guards use
the escape macros only to leave a visible local hazard. `ESC` is not present in
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

## Phase 4 completion

The Agent has passed the 5 × 800-tick safe-forward acceptance test, with
model-driven forward ticks and meaningful action variation. Phase 4 completed
in the autonomous seed-101 run at
`runs/phase4-true-entrance-approach/20260723-142315/episode-01/`: two manually
reviewed genuine entrance frames were separated by 114 real forward ticks, then
the local owner emitted exactly one `ESC=1` completion tick. The model never
had access to `ESC`. Historical validation records are in
`runs/history/EXECUTION_LOG.md`; large historical run artifacts stay local and
are not committed to the outer Git repository.

`vendor/minerl` is an independent Git repository. The outer project does not
commit, delete, or rewrite its Git history.
