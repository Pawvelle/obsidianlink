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

**Status: complete.**

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
- A bounded cave-target memory and local completion gate are covered by unit
  tests: only a first candidate that passes the existing text, direction, and
  frame gates may establish a target; only a second matching candidate after
  at least 12 real forward ticks can cause the environment owner to emit one
  `ESC=1` tick. The model protocol still cannot request `ESC`. This change has
  not yet been validated against a genuine cave-positive MineRL frame. Its
  seed-101 800-tick integration smoke
  (`runs/phase4-cave-target-smoke/20260722-220148`) passed with 7 accepted
  decisions, 468 forward ticks, zero stale decisions, and zero target or ESC
  triggers in the absence of cave evidence.

The earlier absence of a trustworthy positive has now been resolved by a
human-operated, real MineRL session: seed 101 frame
`runs/manual-findcave/20260723-110256/candidates/candidate-tick-03026.png`
shows an enterable dark stone entrance. Qwen emitted `cave_visible=true` with
the fixed entrance-evidence reason; its initial `left` wording did not match
the directional image band, so the bounded local resolver selected the
unambiguous overlapping `center` band and the full candidate gate passed with
`cave_direction_source=local_dark_region`. Its review is recorded in
`runs/manual-findcave/20260723-110256/positive_validation.md`.

Phase 4 completed on 2026-07-23 in the real seed-101 episode
`runs/phase4-true-entrance-approach/20260723-142315/episode-01`. The Agent
autonomously established a target from a reviewed genuine entrance, executed
114 real forward ticks, re-confirmed the same entrance at observation tick 235,
and emitted exactly one locally gated `ESC=1` tick at environment tick 334.
Manual review confirms both candidate frames are the same natural stone-bounded
entrance; the model never controlled `ESC`. See
`runs/phase4-true-entrance-approach/20260723-142315/episode-01/manual_review.md`.

A post-jump 1,800-tick seed-101 closed loop
(`runs/phase4-jump-validation/20260723-112335`) did exercise the entire local
completion flow at tick 606, including one `ESC=1` tick after 56 recorded
forward ticks. It is **not** a completion: manual review found that the prior
broad darkness veto mistook shadowed grass for left-side cave evidence at the
two decisive observations. An initial 2% neutral stone-context calibration
also failed manual review at tick 704: a shallow trench and a dirt wall still
passed as openings. The final completion-grade gate combines a 9% stone-context
minimum with one coherent neutral-dark connected region; it rejects those false
frames and retains the human-collected real entrance center band.
Its real 800-tick seed-101 regression completed normally with 8 raw cave
claims, zero validated candidates, and `ESC=0`. These are safety regressions,
not success evidence. Reviews:
`runs/phase4-jump-validation/20260723-112335/episode-01/manual_review.md`,
`runs/phase4-stone-gate-regression/20260723-113052/episode-01/manual_review.md`,
and `runs/phase4-stone-gate-final/20260723-113350/episode-01/review.md`.
The final predicate also requires one coherent neutral-dark connected region;
its separate real 800-tick regression again produced zero candidates and
`ESC=0` despite six raw cave claims. See
`runs/phase4-opening-geometry-regression/20260723-135401/episode-01/review.md`.

Seed reconnaissance (`scripts/capture_findcave_starts.py`, camera-only panorama,
no Qwen) has screened 160 seeds across ten batches. Nine candidate seeds have
since completed a full Qwen long episode and were manually reviewed frame by
frame; eight remain negative and one (seed 101's first attempt) was
inconclusive due to a camera-pitch runaway, described below:

- Seed 7 (`runs/phase4-cave-search/20260720-161104`): 18,000/18,000 ticks
  completed, zero deaths, but the character stayed stuck against the same dirt
  terrace the whole episode; zero cave candidates were even raised.
- Seed 3 (`runs/phase4-cave-search/20260720-164452`): 3 cave candidates were
  raised and passed the keyword gate, but all three were confirmed by manual
  frame review to be a sunlit sandstone-wall hallucination; the episode ended
  in drowning at tick 7577, unrelated to any cave.
- Seed 30 (`runs/phase4-cave-search/20260721-163330`): seed reconnaissance
  found a genuine rocky highland (real exposed gray stone, not sandstone or
  dirt), the strongest terrain candidate screened so far. However the episode
  drowned at tick 1479 in a water crossing before ever reaching that terrain;
  the 2 cave candidates raised were both confirmed by manual review to be an
  underwater dirt-riverbed hallucination, not a cave.
- Seed 23 (`runs/phase4-cave-search/20260721-164500`): reconnaissance flagged a
  dark square depression as ambiguous (possibly man-made). 18,000/18,000 ticks
  completed, zero deaths. The character stayed against a dirt terrace the whole
  episode; the single cave candidate raised was confirmed by manual review to
  be a dirt-arch hallucination, not a cave. A small dark gap visible in the
  far background of `final.png` was never approached and is not usable
  evidence.
- Seed 47 (`runs/phase4-cave-search/20260721-172422`): seed reconnaissance
  found the strongest land-accessible natural gray stone wall screened so far
  (no water crossing required). The episode still raised 63 raw
  `cave_visible=true` claims, but the fail-closed keyword gate correctly
  rejected 62 of them; the 1 candidate that passed the gate was confirmed by
  manual review to be a dirt-pit close-up hallucination (an isolated gray
  block, not a wall). The episode ended at tick 14696, killed by a skeleton at
  night, before clear evidence the character ever reached the scouted stone
  wall.
- Seed 54 (`runs/phase4-cave-search/20260721-211512`): seed reconnaissance
  found the largest and clearest natural exposed-rock mountain screened across
  all four batches, directly in front of spawn with no water visible in any of
  the four cardinal panorama directions. The run was rejected by the internal
  acceptance gate after only 1353/18000 ticks: the character fell into a
  narrow ravine stream hidden between two dirt terraces (invisible from the
  fixed four-direction panorama angles) and drowned before ever reaching the
  scouted rock slope. Zero cave candidates passed the evidence gate.
- Seed 75 (`runs/phase4-cave-search/20260721-213452`): seed reconnaissance
  flagged a stone structure with a dark square hole as ambiguous (likely a
  man-made well/ruin base rather than a natural cave). The run reached
  17497/18000 ticks (`accepted=false`, run-statistics gate) and ended when the
  character was killed by a zombie at night. This episode raised an unusually
  high 151/162 raw `cave_visible=true` claims; manual review shows this was
  driven by widespread nighttime darkness being misread as cave darkness (a
  new false-positive source distinct from sunlit sandstone or dirt-pit
  hallucinations). The keyword gate still correctly rejected 149 of the 151
  claims; the 2 that passed were confirmed negative (flat night-dark gradient,
  and a real rock corner with no opening).

- Seed 92 (`runs/phase4-cave-search/20260721-220616`): seed reconnaissance
  found a natural gray-stone patch with a dark square recess right at spawn
  (yaw000), directly ahead with no water in any of the four cardinal
  directions — the strongest "looks like a real opening" candidate screened so
  far. The run reached 17288/18000 ticks (`accepted=false`, run-statistics
  gate) and ended with the character killed by a zombie at night. Sampled
  frames across the whole episode show the character wandered away across
  dirt terraces and never once re-approached the scouted stone structure after
  spawn; the single frame that passed the evidence gate (`tick-13027.png`) is
  a plain nighttime dirt-terrace shot with no rock or opening evidence.

- Seed 101 (`runs/phase4-cave-search/20260721-224034` and `-230035`, two
  attempts): seed reconnaissance found the visually strongest candidate
  screened across all seven batches — a large mountain directly at spawn with
  a distinctly darker recess in its exposed rock face, no water in any of the
  four cardinal directions. Both 18,000-tick attempts suffered reproducible
  camera-pitch drift and did not observe the terrain meaningfully. The later
  cumulative camera-pitch guard corrected that failure: a guarded 1,800-tick
  Qwen validation completed with 773 forward ticks and one guard correction,
  but zero cave claims. Targeted no-Qwen collection then established a more
  specific remaining limit. A 192-tick forward/sprint approach could not climb
  the first one-block ledge; a separately bounded jump-assisted capture reached
  the rock shelf but met a close wall rather than an enterable opening. Offline
  Qwen review of the closest recess frame returned `cave_visible=false`.
  Seed 101 is now a negative directed sample, not a cave positive. Its evidence
  is recorded in `runs/phase4-target-approach/20260723-104621/review.md`.

Drowning in an early water crossing (seeds 3, 30, and 54) and death by hostile
mob at night (seeds 47, 75, and 92) are repeated failure modes that can end an
episode before it ever reaches promising rocky terrain. Seed 54 specifically
shows that a clean four-direction panorama is not sufficient to rule out a
hidden ravine/stream along the actual walking path. Seed 92 shows that even
when reconnaissance finds a strong-looking target directly at spawn, the
model's walking path during the long episode can drift away from it entirely
before night falls. Seed 75 additionally shows that nighttime ambient darkness
itself can be misread by the model as cave darkness, though the keyword
evidence gate still filters out nearly all such claims. Seed 101's earlier
camera-pitch failure has been repaired by the cumulative pitch guard and its
guarded follow-up observed terrain normally; however, its targeted approach
  exposed an independent step-height limit. The current Qwen policy now permits
  a one-tick jump only alongside `move_forward` for a visibly one-block ledge
  on an otherwise safe route; it is clamped off for every other macro. No model-generated ESC,
water-safety, or completion boundary was relaxed during the research.

Local evidence:

- `runs/phase4-forward-acceptance/20260717-205706/summary.json`
- `runs/phase4-cave-preflight/20260717-211041/summary.json`
- `runs/cave-starts/20260720-160032/review.md` (seeds 1-16 reconnaissance)
- `runs/cave-starts/20260721-162624/review.md` (seeds 17-32 reconnaissance)
- `runs/cave-starts/20260721-170628/review.md` (seeds 33-48 reconnaissance)
- `runs/cave-starts/20260721-210819/review.md` (seeds 49-64 reconnaissance)
- `runs/cave-starts/20260721-212753/review.md` (seeds 65-80 reconnaissance)
- `runs/cave-starts/20260721-215628/review.md` (seeds 81-96 reconnaissance)
- `runs/cave-starts/20260721-223334/review.md` (seeds 97-112 reconnaissance)
- `runs/cave-starts/20260722-213644/review.md` (seeds 113-128 reconnaissance)
- `runs/cave-starts/20260722-214058/review.md` (seeds 129-144 reconnaissance)
- `runs/cave-starts/20260723-103437/review.md` (seeds 145-160 reconnaissance)
- `runs/phase4-target-approach/20260723-104621/review.md` (seed 101 directed
  approach review)
- `runs/manual-findcave/20260723-110256/positive_validation.md` (seed 101
  human-operated genuine cave-positive validation)
- `runs/phase4-jump-validation/20260723-112335/episode-01/manual_review.md`
  (seed 101 automatic completion-flow false-positive regression)
- `runs/phase4-stone-gate-regression/20260723-113052/episode-01/manual_review.md`
  (intermediate stone-context false-positive regression)
- `runs/phase4-stone-gate-final/20260723-113350/episode-01/review.md`
  (final stone-context 800-tick no-ESC regression)
- `runs/phase4-opening-geometry-regression/20260723-135401/episode-01/review.md`
  (coherent-opening 800-tick no-ESC regression)
- `runs/phase4-true-entrance-approach/20260723-142315/episode-01/manual_review.md`
  (autonomous genuine cave completion)
- `runs/phase4-cave-search/20260720-161104/manual_review.md` (seed 7)
- `runs/phase4-cave-search/20260720-164452/manual_review.md` (seed 3)
- `runs/phase4-cave-search/20260721-163330/manual_review.md` (seed 30)
- `runs/phase4-cave-search/20260721-164500/manual_review.md` (seed 23)
- `runs/phase4-cave-search/20260721-172422/manual_review.md` (seed 47)
- `runs/phase4-cave-search/20260721-211512/manual_review.md` (seed 54)
- `runs/phase4-cave-search/20260721-213452/manual_review.md` (seed 75)
- `runs/phase4-cave-search/20260721-220616/manual_review.md` (seed 92)
- `runs/phase4-cave-search/20260721-224034/manual_review.md` (seed 101, first
  attempt — camera-pitch runaway)
- `runs/phase4-cave-search/20260721-230035/manual_review.md` (seed 101, second
  attempt — camera-pitch runaway reproduced)

## Phase 5 — Bounded cave-entry verification

**Status: code complete and unit-tested; one real-MineRL 800-tick
safety run was performed on 2026-07-29 (seed 101, `--cave-entry-phase`
enabled). The entry phase stayed in its `idle` state, no `ESC` tick
was emitted, and no post-entry evidence frame was written. The run
validates that the new control layer does not trigger spuriously under
real MineRL conditions; it does **not** constitute a Phase 5
completion. A controlled true-entrance reproduction is still required
before this phase can be marked complete.**

The Phase 4 double-confirmation gate already proved that the agent can
*find* a real cave entrance and stop. Phase 5 adds the next safe step:
after the gate fires, walk a short, locally driven forward block into the
validated opening, record one post-entry evidence frame, then emit the
single local `ESC` tick. The current task is still
`MineRLBasaltFindCave-v0`; the goal is evidence-grade entry proof, not
general navigation or any model/dependency upgrade.

### Design summary

- The new layer is opt-in via `--cave-entry-phase` (default off). When
  disabled, the agent behaves exactly as Phase 4. When enabled, the
  reconfirmation path no longer immediately calls
  `executor.request_cave_completion`; it instead activates a
  `CaveEntryPhase` state machine.
- `CaveEntryPhase` is a small, single-shot state machine with five
  states: `idle` → `entering` → `entered` | `aborted` | `unverified`.
  `entered` means the bounded forward block ran to completion **and**
  the post-entry frame passed the local plausibility check; in that
  case the single local `ESC` tick is emitted and
  `cave_completion_requested` becomes `True`. `aborted` means a
  safety guard tripped (water hazard, low-progress, turn-scan,
  environment-done, watchdog, max-ticks) — the bounded block did
  not run to completion, no `ESC` is emitted. `unverified` means the
  bounded block ran to completion but the post-entry frame did
  **not** pass the local plausibility check — the post-entry
  evidence frame is still written for human review, but no `ESC` is
  emitted and `cave_completion_requested` stays `False`. All three
  terminal states are non-recoverable: re-activation is forbidden
  after any of them.
- Activation requires (a) the existing `cave_target_reconfirmations
  ≥ 1` gate, (b) the same `forward_ticks_after_acquisition ≥
  CAVE_COMPLETION_MIN_FORWARD_TICKS` precondition, and (c)
  `cave_completion_requested` still `False`. Re-activation is
  forbidden.
- The entry block borrows the deterministic forward-continuation
  pattern: a bounded local forward macro runs in the executor while
  per-tick safety guards (water hazard, low-progress, turn-scan,
  environment-done, watchdog, max-ticks) remain in force. If any
  guard trips, the entry phase is `aborted` and `ESC` is **not**
  emitted. When the budget is exhausted cleanly (last macro
  finished), the phase routes to one of two terminal states
  depending on the post-entry frame: `entered` (plausibility passed)
  triggers the single local `ESC` tick; `unverified` (plausibility
  failed) leaves the phase sealed with the evidence frame saved
  for human review and no `ESC` emitted.
- The forward budget is hard-capped at
  `CAVE_ENTRY_PHASE_MAX_TICKS = 30` ticks (≈ 1.5 s at 20 Hz, ≈ 3 m of
  in-world travel) and is *not* configurable beyond that via the
  model. A second command-line override
  (`--cave-entry-phase-max-ticks`) is available for offline
  experiments but never changes the protocol.
- The model is never given a new privileged action. During the
  entering phase any new model decision is acknowledged so the
  planner worker never blocks, but it is dropped: it cannot turn,
  jump, attack, extend the entry budget, or run its own forward
  macro. Entry and `forward_continuation` are intentionally exclusive
  in this window.
- The single local `ESC` tick is emitted by
  `executor.request_cave_completion()` **only** when the bounded
  block ran to completion **and** the post-entry frame passed the
  local plausibility check; otherwise no `ESC` is emitted and
  `cave_completion_requested` stays `False`. The post-entry evidence
  frame is always recorded before that decision is taken, in
  `entry_evidence/post-tick-NNNN.png` next to `decision_frames/`,
  regardless of which terminal state the phase lands in.

### Acceptance criteria (covered by `tests/test_cave_entry.py` and
`tests/test_memory.py`)

1. `cave_entry_phase_enabled = False` preserves the Phase 4 path
   bit-for-bit: same `cave_completion_requested`, same single `ESC`
   tick, no `entry_evidence/` directory.
2. `cave_entry_phase_enabled = True` with no double confirmation never
   activates the entry phase: state stays `idle`, `esc_nonzero_ticks
   == 0`.
3. With double confirmation and sufficient forward progress, the
   entry phase reaches `entered`, `entry_forward_ticks` equals the
   configured budget, and exactly one local `ESC` tick fires.
4. The post-entry evidence frame is written under `entry_evidence/`
   and the run still terminates on a single `ESC` tick with
   `termination_reason == "cave_completion_requested"`.
5. A water hazard detected during entry aborts the phase; no `ESC` is
   emitted; `cancellation_reason == "water_hazard"`.
6. Two consecutive low-progress stalls during entry abort the phase
   with `cancellation_reason` in `{low_progress, turn_scan}`; no
   `ESC` is emitted.
7. A second cave candidate that arrives after the entry phase has
   reached `entered` does **not** re-activate the phase and does not
   re-emit `ESC` (`esc_nonzero_ticks == 1`).
8. A model decision that arrives during the entry phase is
   acknowledged and counted but not executed, and it does not extend
   the entry forward budget.
9. `entry_forward_ticks` is always `≤ max_ticks` and `≤ forward_ticks`.
10. **P1 — Local plausibility is a gate, not an annotation.** The
    post-entry frame is checked against the coarse luminance rule
    (post world luminance below `50` or at least 30% lower than the
    pre-entry frame). When the rule fails, the phase lands in
    `unverified`; the evidence frame is still written under
    `entry_evidence/`, but `cave_completion_requested` stays
    `False`, `request_cave_completion()` is **not** called, and
    `esc_nonzero_ticks` stays `0`. The episode falls through to
    `tick_budget` or `environment_done` for the termination reason.
    The phase cannot be re-activated afterwards.
11. The terminal state is `entered`, `aborted`, or `unverified`; once
    any of them is reached, no further transition is possible. The
    `unverified` state carries the same single-shot guarantee as
    `entered` and `aborted`: `cave_entry_phase.activate()` raises if
    called again.
12. A reproduction of the `cave_completion_requested` path that calls
    `request_cave_completion()` twice is impossible by construction
    (entry phase owns the single ESC, Phase 4 path is suppressed
    when the entry phase is in flight or terminal).
13. Activating the entry phase interrupts any forward macro already
    in flight in the executor (typically a leftover
    `forward_continuation` block): the leftover macro's remaining
    forward ticks are not counted toward `entry_forward_ticks` and
    do not extend the entry phase beyond its configured budget.

### Test surface

- `tests/test_memory.py` — `CaveEntryPhaseTests`: state machine,
  input validation, single-shot activation, abort idempotence,
  plausibility routing to `entered` vs `unverified`, explicit
  `mark_unverified` path, re-activation refusal from `unverified`.
- `tests/test_cave_entry.py` — thirteen integration tests covering the
  acceptance list above using a fake `MineRLEnvAdapter` and the same
  `FakePlanner` / `_DelayedDecisionMailbox` machinery as
  `tests/test_agent.py`. No real MineRL session, no Qwen inference,
  no network.

### Real-MineRL safety run — 2026-07-29

A single 800-tick MineRL run was executed on seed 101 with
`--cave-entry-phase` explicitly enabled. The run's purpose was to
validate the new control layer end-to-end under real MineRL
conditions; it was **not** a Phase 5 completion attempt, because no
real entrance was found and the entry phase never activated.

| Field | Value |
|---|---|
| Run dir | `runs/phase5-cave-entry-validation/20260729-220922/` |
| Manual review | `runs/phase5-cave-entry-validation/20260729-220922/manual_review.md` |
| Seed | 101 |
| Tick budget | 800 |
| Completed ticks | 800 / 800 |
| `termination_reason` | **`tick_budget`** |
| `esc_nonzero_ticks` | **0** |
| `cave_completion_requested` | **`False`** |
| `cave_target_acquisitions` | 1 |
| `cave_target_reconfirmations` | **0** |
| `cave_entry_phase.state` | **`idle`** (never activated) |
| `cave_entry_phase.is_terminal` | `False` |
| `cave_entry_phase.activation_tick` | `None` |
| `cave_entry_phase.entry_forward_ticks` | 0 |
| `cave_entry_phase.evidence_frame` | **`None`** (no `entry_evidence/` directory was created) |
| `cave_entry_phase.plausible` | `None` |
| Entry evidence frames written | **0** |
| Cave-completion evidence frames | **0** |

The agent walked 327 real forward ticks across a 40-tick
observation interval; Qwen emitted 7 `cave_visible=true` claims. One
of those claims passed the text + directional stone-bounded
dark-opening frame gate as a `cave_candidate` at observation tick 0,
but `target_before_decision.active` was `False` there, so it became
a `cave_target.acquire` (not a reconfirmation) and accumulated only
`forward_ticks_after_acquisition=1`, far below the
`CAVE_COMPLETION_MIN_FORWARD_TICKS=12` gate. The remaining 5 claims
were vetoed at the stone-bounded dark-opening frame gate; one further
claim was rejected at the text-evidence step (missing direction
word). No double confirmation ever occurred, so the entry phase
preconditions were never met. No `ESC` tick was emitted, no
post-entry evidence frame was written, and the `unverified` branch
was never reached — the entry phase was correctly sealed in
`idle` for the full 800 ticks.

This run validates the Phase 5 control layer: the entry phase does
not activate spuriously, the new ESC suppression on the
`unverified` branch was never needed, and the existing Phase 4
guards (water hazard, low-progress, turn-scan, environment-done,
watchdog, max-ticks, camera-pitch guard) remained off for the
whole episode. It does **not** constitute a Phase 5 completion: no
`ESC` was emitted, no post-entry evidence was recorded, and no
cave-completion evidence was produced. A controlled true-entrance
reproduction is still the remaining step before Phase 5 can be
marked complete.

### What is *not* in this phase

- No general navigation, combat, underwater, or path-planning work.
- No model or dependency upgrade (MineRL, Qwen, Python, JDK, model
  commit are all untouched).
- No weakening of the existing text/frame/direction/stone-context
  gates, the camera-pitch guard, the water-hazard guard, the
  low-progress recovery, the ESC policy, or the
  `forward_continuation` safety layer.
- No new privileged action for the model.
- No real MineRL run has been executed for the controlled
  true-entrance reproduction itself; that step requires another
  explicit user approval and would write to a new
  `runs/phase5-controlled-real-entrance-repro/<timestamp>/`
  directory.

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
7. `ESC` is never model-generated. The environment owner may emit it once only
   through the double-confirmed cave-completion gate, with evidence logged for
   manual review.

Detailed historical validation records are in `runs/history/EXECUTION_LOG.md`.
Older run assets remain in `runs/history/artifacts/` and `runs/history/logs/`.
