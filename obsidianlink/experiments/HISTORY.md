# Experiment records

Git tracks this HISTORY, experiment scripts, and small evidence
summaries under `evidence/`. Live raw JSON / PNG under `runs/` stay
on the local machine and are gitignored. Deleting Git blobs is not
deleting the experimental conclusions below.

## Status of historical results

All files that already existed before the 2026-08-19 architecture reset
are **historical debugging records**. They must not be used as formal
Benchmark capability conclusions.

### L1 results — invalid for L1 capability conclusion

These L1 artifacts changed task semantics, used an unreliable
`ObservationFromGrid` truth channel, and/or ran a Reactive agent that
did not pass the RGB frame to the vision model:

* `l1_oracle_1ep_20260819_093744Z.json`
* `l1_oracle_1ep_20260819_094353Z.json`
* `l1_reactive_Qwen3-VL-2B-Instruct_1ep_20260819_094710Z.json`
* matching `*_frames/` directories

Reasons:

1. L1 semantics were changed: Casting + Frame Construction were moved
   into the scene; the agent only attempted Ignition + Nether Entry.
2. Evaluator world truth from `ObservationFromGrid` was unreliable
   (POV showed obsidian; grid reported 0/14).
3. The Reactive L1 run discarded `Observation` (`del observation`) and
   called the text-only model path.

Do not treat `success=False, reason=portal_frame_incomplete` from these
runs as an Agent capability result.

### L1 technical feasibility spike — 2026-08-19

`l1_spike_20260819_124538Z.json` and `l1_spike_20260819_124538Z_frames/` are a **mechanic feasibility** record, not an Agent capability result (`valid_for_l1_agent_conclusion: false`).

Live evidence (no pre-built portal frame):

* `InventoryAgentStart` can give water_bucket + 14 lava_bucket + flint_and_steel
* Placement after warmup: `(0.5, 101.0, 0.5)`
* hotbar select + `use` pours lava; lava+water casts a new obsidian block; water can be picked up
* `EquipAction` (`equip none`) crashes this MineRL 1.0.2 / MCP-Reborn stack
* Oracle did **not** complete a 10-block frame, ignition, or Nether entry (timeout during extra casts; later frames are invalid)

The earlier `l1_spike_20260819_124215Z.json` is the EquipAction crash repro.

### L1 controlled environment v0.1 smoke — 2026-08-19

The first live smoke (`l1_env_smoke_20260819_174234Z`) used an obsidian
courtyard floor because DrawingDecorator can only draw `lava` /
`obsidian`. That floor was replaced: an Agent could treat the ground as
portal material.

Current live smoke: `l1_env_smoke_20260819_175245Z.json` and
`l1_env_smoke_20260819_175245Z_frames/`:

* spawn `(0.5, 4.0, 0.5)` on superflat **grass** (not obsidian)
* DrawBlock is lava-pool only; no obsidian floor or walls
* 4×4 lava pool visible in POV (`lava_frac≈0.056`, `grass_frac≈0.76`)
* inventory and `hotbar.1-5` unchanged
* no EquipAction, no ObservationFromGrid, no pre-built portal

This is **not** an L1 Agent or Oracle capability result (`oracle_or_agent_run: false`).

### L1 mechanical interaction — 2026-08-20

Canonical live run `l1_mechanics_20260820_033330Z` (local `runs/` only). Git keeps `evidence/l1_mechanics_summary.json`. Not an Oracle, Evaluator, or Agent capability result (`oracle_or_agent_run: false`).

Live evidence:

* env `MineRLL1Controlled-v0`; 266 steps; wall time ≈ 30.3s
* empty bucket scooped a lava source → `lava_bucket` (Hot Stuff); pool missing one source
* native `use` placed lava, then water; lava_frac dropped and POV shows a new obsidian block under water
* cobblestone placed via hotbar + `use` (no PlaceBlock): 64 → 63
* iron pickaxe `attack` broke it and the drop was picked up: 63 → 64
* no EquipAction, no ObservationFromGrid, no DrawBlock obsidian, no preloaded lava_bucket
* **NEW OBSIDIAN = TRUE**

Earlier attempts `032340Z` (scoop miss on grass rim), `032545Z` / `032740Z` / `032915Z` (obsidian yes, cobble break no because crosshair hit water / current pushed the player) are local debugging records, not git-tracked.

### L1 Evaluator — 2026-08-20

Live smoke `run_l1_evaluator_smoke.py` (local `runs/` only; no summary
JSON needed, the assertions are the evidence). Confirms on this MineRL
1.0.2 / MCP-Reborn / Malmo 0.37.0 stack:

* `RewardForTouchingBlockType(nether_portal)` registers and steps without
  crashing; its reward reaches `Environment.hidden_state["reward"]`
  (never `Observation`).
* `ObservationFromCurrentLocation` exposes `biome_id`, `can_see_sky`,
  `light_level`, etc. in `location_stats` (both gym `info` and raw obs).
* With no portal touched, `L1Evaluator.evaluate(...)` returns
  `success=False, reason=nether_entry_not_confirmed` — fails closed.

Not an Oracle or Agent run.

### Full Scripted Oracle — 2026-08-20, blocked at Gate 1

`run_l1_oracle.py` / `l1_oracle.py`. Local `runs/l1_oracle_*` only (raw
JSON + PNG, not git-tracked).

Live evidence:

* Reference geometry: cornerless 10-block frame at `base_x=-1, base_y=4,
  z=3` (bottom row rests on the already-validated y=3 grass floor).
* First attempt (no mold): pouring lava on open grass spread across
  several adjacent cells instead of staying in the single target cell —
  POV evidence contradicts the earlier mechanics test's "NEW
  OBSIDIAN=TRUE" being a precise geometric proof; it was only a visual
  heuristic. Fix: cobblestone mold walls (left/right of the 2-cell
  bottom row) placed via the same proven hotbar+`use` mechanic, before
  pouring.
* Mold construction and at least one full lava-bucket scoop succeeded
  live. Full Gate 1 (both bottom-row cells poured, watered, and
  obsidian-confirmed) was not completed in one clean episode: two
  independent runs hit `TimeoutError` → `RuntimeError: Attempted to
  step an environment server with done=True` at ~270-280s wall time,
  even after cutting the action/retry budget substantially between
  attempts.
* Diagnostic: a pure-`WAIT` loop ran 93,200 steps / 340s with **no**
  timeout, ruling out a fixed episode wall-clock cap. `minerl/env/
  _multiagent.py` hardcodes `SOCKTIME = 240s` per-step socket recv
  timeout. The two Oracle failures did not share the same triggering
  action type (one on a mold `use`, one on a return-trip `move`),
  consistent with progressive Minecraft-server-side slowdown from
  fluid-simulation load rather than one deterministic bad action.
* No EquipAction, PlaceBlock, ObservationFromGrid, DrawBlock portal,
  teleport, command, prebuilt frame, or inventory injection were used.
* **ORACLE SUCCESS = False**, stopped at Gate 1 per the no-fake-success
  rule. Root cause is a stack-level stability limit, not a geometry or
  aiming bug.

### Water recovery isolation — 2026-08-20

Live runs (local `runs/` only, not git-tracked):

* `water_recovery_iso_20260820_105237Z` (Run 1)
* `water_recovery_iso_20260820_105355Z` (Run 2)

Not Gate 1. Not an Oracle or Agent capability result. Protocol: place one water source on grass, recover with a **single** `USE`, then 20 WAIT-only ticks (no USE / ATTACK / MOVE / HOTBAR / CAMERA). Mapped MineRL `use` was recorded every tick.

Phenomenon:

* Before pour: `bucket=1`, `water_bucket=1`
* Pour `USE` (tick 3): inventory stable immediately at `bucket=2`, `water_bucket=0` (held through 8 fluid-wait ticks)
* Recover `USE` (tick 13): `water_bucket` first appears; `bucket=1`, `water_bucket=1`, `selected_item=water_bucket`
* Consecutive hold: 21 ticks (recover + 20 WAIT). Did **not** disappear
* Wait window: all `wait`, `minerl.use=0`. reward / done / pose / selected_item unchanged
* Run 2 reproduced Run 1 exactly

Root cause of the hypothesized WAIT-only rollback: **not confirmed, because it did not happen**. `water_bucket=1` after a single legal recover is not a transient observation in this setting.

Fix: none. No `N=3` stable-state confirmation — live evidence says the same-tick inventory delta is already authoritative here.

Limitation: this isolation forbids CAMERA after recover, so it does not explain a Gate 1 flip that only appears once the camera moves. Remaining suspects for that earlier symptom are a multi-tick `USE` burst or a later CAMERA aimed at leftover flowing water / the lava-mold scene — not delayed Malmo inventory sync under WAIT.

### Gate 1 one obsidian — 2026-08-20

Live runs (local `runs/` only):

* `l1_oracle_20260820_113730Z` — **invalid**: water replaced the lava source
  (`lava_frac` 0.38→0, `obsidian_frac` unchanged). Heuristic `ok` was a
  false positive because it treated lava disappearance as obsidian.
* `l1_oracle_20260820_113909Z` — **valid**: neighbor-cell water after
  lava settle. `obsidian_frac` 0.0009→0.041, `obsidian_visual_rose=True`.
  65 steps, 3 `USE`, ~28s.
* `l1_oracle_20260820_114030Z` — **valid**, reproduced. `obsidian_frac`
  0.0009→0.023. Same 65-step / 3-`USE` sequence.

Sequence: scoop lava (`USE`×1) → place lava (`USE`×1 sneak) → wait 8 →
yaw nudge → place water beside lava (`USE`×1 sneak) → look away 2 ticks
→ wait until `obsidian_visual_rose`. Starting `water_bucket` is used
directly (no extra collect/recover before the pour).

Not a portal frame, ignition, or Nether entry. `L1Evaluator.success`
stays False (`nether_entry_not_confirmed`). Lava still spreads on open
grass; this is not geometric proof of a specific frame cell.

### Other historical pilots

D1 / D2 / D3 live JSON files remain useful as pipeline / scene-validity
evidence. They were collected under the previous package layout. They
are still **pilot / not capability conclusions**, matching the research
plan's Prototype vs Benchmark Evaluated distinction.
