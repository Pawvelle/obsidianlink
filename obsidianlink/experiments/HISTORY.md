# Experiment records

This directory stores live-run JSON and frames.

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

### Other historical pilots

D1 / D2 / D3 live JSON files remain useful as pipeline / scene-validity
evidence. They were collected under the previous package layout. They
are still **pilot / not capability conclusions**, matching the research
plan's Prototype vs Benchmark Evaluated distinction.
