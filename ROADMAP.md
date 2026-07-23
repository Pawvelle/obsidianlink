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
