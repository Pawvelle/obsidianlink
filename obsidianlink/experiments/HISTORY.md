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

### Other historical pilots

D1 / D2 / D3 live JSON files remain useful as pipeline / scene-validity
evidence. They were collected under the previous package layout. They
are still **pilot / not capability conclusions**, matching the research
plan's Prototype vs Benchmark Evaluated distinction.
