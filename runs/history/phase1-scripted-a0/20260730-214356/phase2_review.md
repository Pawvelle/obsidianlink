# Phase 2 Evaluator Review (independent of Phase 1)

This file records the Phase 2 evaluator status for the
`phase1-scripted-a0/20260730-214356` run. It is intentionally
**independent of the Phase 1 manual review** and never modifies the
original Phase 1 evidence. See `manual_review.md` for the Phase 1 review
(unchanged) and `phase2_evaluator_replay.json` for the machine-readable
replay verdict.

## Verdict

Phase 2 evaluator has **insufficient evidence** to replay this run.

## Round 3 code-only fixes (2026-07-30)

The 7 audit issues from Round 3 have been addressed at the offline
code level. **None of them produce new MineRL evidence**; the replay
status is still `insufficient_evidence` because the historical
artefacts do not contain the required per-step ground truth.

Code-level changes:

1. **External-structure attribution** — the backend now computes
   `external_structure_candidate_count` and exposes it in the
   evidence dict. A geometrically valid frame whose required cells
   are not all in the agent attribution set (either fully external
   or mixed attributed+external) is classified as
   `frame_not_built_by_episode` instead of `frame_never_valid`.
2. **Observation-bound attribution** — accepted placement credits are
   valid only for the current post-step observation. Fresh delta count
   must equal the credit count; ambiguous batches fail closed as
   external, credits expire at the boundary, and external cells cannot
   later be re-attributed.
3. **External Nether entry** — position is only a sanity check. A
   positive verdict requires typed bridge transition evidence whose
   interior offsets exactly match the latched frame identity. Missing
   evidence is unknown and cannot produce success.
4. **Partial-frame structural rule** — confirmed in the unit tests:
   three obsidian on different edges of a hypothetical frame
   (`(1,0,1)`, `(0,2,1)`, `(3,3,1)`) do **not** trigger
   `build_site_selected`; the rule requires ≥3 on one edge or an
   L-shape with a shared corner.
5. **Latched timestamp enforcement** — `EvaluationState.__post_init__`
   raises `ValueError` when a milestone step is set without a
   matching `latched_timestamps` key. Multiple emissions return
   identical timestamps. Per-agent Nether timestamps use
   `agent_entered_nether:<agent_id>` keys so they do not collide.
6. **`has_missing_truth` is real** — the property is computed in
   `FrameDetectionResult` from `missing_frame_cell_count` /
   `missing_interior_cell_count` aggregates populated at
   construction time. `_name_at()` placeholder no longer exists.
7. **`time.time()` fallback removed** — `milestone_events()` only
   reads `latched_timestamps`; missing keys raise instead of being
   silently filled in at emission time.

Round 4 tightened these contracts further:

- attribution credits expire at the post-step observation boundary,
  ambiguous delta batches fail closed, and external cells can never be
  re-attributed;
- atSpawn grid offsets are converted through the reset-time world anchor;
- proximity alone no longer proves portal-entry causality. A positive
  verdict requires typed bridge transition evidence for the exact latched
  frame; missing evidence is unknown;
- terminated false/unknown entry correlations receive explicit failure
  types instead of `failure_type=None`.

These changes add regression coverage for old-external re-attribution,
stale no-op credits, nearby external dimension flips, and world-anchor
conversion. The offline suite passes 121 / 121 tests. These changes still
produce no new MineRL evidence.

The follow-up bridge source now defines typed server-side transition evidence
(`sequence`, from/to dimension, and source portal block world position) and
an explicit grid world origin. It is persisted through an idempotent MCP patch
script. A user-approved `./gradlew compileJava` attempt stopped during
ForgeGradle project configuration because the default PATH exposed Java
25.0.3 instead of the required Java 8; Java compilation did not begin and
Minecraft was not started. A read-only follow-up located the pinned OpenJDK
8.0.472 under `/opt/anaconda3/envs/mc-agent`; a second `compileJava` invocation
then completed successfully (five tasks: four executed and one up-to-date).
Minecraft was not started, so the bridge remains unvalidated in a real MineRL
run.

## Recorded facts (from the historical artefacts)

- 14 obsidian placements, 2 dirt placements, 1 flint-and-steel use
- `first_obsidian_step=6`, `last_obsidian_step=148`
- `first_flint_step=158` (note: this is the action step, not the
  Nether transition step; the historical events do not record when
  `dimension` changed)
- Final dimension: `minecraft:the_nether`
- `max_obsidian_added=14`, `portal_activated_latched=true`

## Missing evidence for Phase 2

The new Phase 2 evaluator requires:

1. **Per-step portal grid snapshots**, not just the collapsed
   `portal_grid_changes` list. The historical artefacts only carry
   the aggregated change list, and `obsidian` / `fire` blocks were
   normalised to `other` by the bridge.
2. **Block-change attribution** that ties each obsidian delta to a
   specific `place_block` action. The historical artefacts record
   step-level actions but do not record which grid cell was produced
   by which action.
3. **A precise Nether transition step** for each agent
   (`first_nether_step_by_agent`). The historical artefacts only
   record the final dimension flag.
4. **Pre-transition agent position** in world coordinates, so the
   evaluator can verify the agent stepped through the latched portal
   rather than an external teleport.
5. **Explicit termination signal** (`episode_terminated`,
   `terminated_step`, `terminated_reason`). The historical run ended
   by max-step exhaustion but the step is not recorded as a structured
   termination event.

Without (1)–(5) the new evaluator cannot honestly report
`success=True` for the historical run, and the new contract explicitly
forbids fabricating these facts.

## Phase 2 status

Phase 2 remains **in progress** as recorded in `ROADMAP.md` and
`README.md`. Closing the gap requires either:

- a Java bridge extension that supplies typed portal-transition identity
  plus per-step grid/block-delta evidence, followed by
- a fresh controlled MineRL run (both bridge/Gradle work and the run require
  explicit user approval).

Until either is done, **Phase 2 is not complete** and **Phase 3 must
not start**.

## Files

- `manual_review.md` — original Phase 1 review (untouched)
- `phase2_review.md` — this file
- `phase2_evaluator_replay.json` — machine-readable replay verdict
- `events.jsonl`, `summary.json` — original Phase 1 artefacts
