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

Phase 4 is not complete: the 506+ historical decision frames and newest replays
contain no trustworthy positive cave-entrance sample. Positive detection cannot
therefore be validated and FindCave cannot be marked successful. The next step
is only to obtain a genuine cave-entrance frame or episode, then review the
candidate decision and approach action. Do not add a paper-style A/B runner,
complex long-term memory, or another model backend.

Seed reconnaissance (`scripts/capture_findcave_starts.py`, camera-only panorama,
no Qwen) has screened 112 seeds across seven batches. Nine candidate seeds have
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
  a distinctly darker, deeper recess visible in its exposed rock face, no
  water in any of the four cardinal directions. Both independent attempts ran
  the full 18000/18000 ticks (`accepted=true`, character survived both times)
  but raised **zero** raw `cave_visible=true` claims in either run. Frame
  sampling across both whole episodes shows the camera pitch drifted upward
  starting around tick 640 and stayed locked on sky/clouds (later, night
  stars) for essentially the rest of each run; the character never saw the
  ground, the mountain, or any terrain again after the first ~2000 ticks. This
  reproduced identically on a second independent attempt, so it is treated as
  a repeatable behavior for this seed/scene rather than a transient glitch.
  The mountain's dark recess was never re-examined with visual evidence in
  either run, so it is neither confirmed nor rejected — it is marked as an
  unusable candidate under the current agent behavior rather than counted as a
  positive or negative sample, and no further long episodes are planned for
  this seed. This is a further confirmation that `accepted=true` alone does
  not indicate a meaningful search attempt, in the same spirit as
  `cave_visible=true` alone not indicating success.

Drowning in an early water crossing (seeds 3, 30, and 54) and death by hostile
mob at night (seeds 47, 75, and 92) are repeated failure modes that can end an
episode before it ever reaches promising rocky terrain. Seed 54 specifically
shows that a clean four-direction panorama is not sufficient to rule out a
hidden ravine/stream along the actual walking path. Seed 92 shows that even
when reconnaissance finds a strong-looking target directly at spawn, the
model's walking path during the long episode can drift away from it entirely
before night falls. Seed 75 additionally shows that nighttime ambient darkness
itself can be misread by the model as cave darkness, though the keyword
evidence gate still filters out nearly all such claims. Seed 101 shows a
distinct, reproducible failure mode across two independent attempts: the
camera pitch itself can drift upward and stay locked on the sky for nearly an
entire 18000-tick episode, which the run-statistics acceptance gate does not
detect. These are noted as known limitations; no water-avoidance, combat,
camera, or safety-boundary code was changed to address them without explicit
user approval.

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

Detailed historical validation records are in `runs/history/EXECUTION_LOG.md`.
Older run assets remain in `runs/history/artifacts/` and `runs/history/logs/`.
