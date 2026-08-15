# ObsidianLink v2.0 项目状态

更新时间：2026-08-15

## Scope decision

v2 已完成 scope freeze、architecture reset 与 legacy quarantine：

- 唯一研究主题为 Nether Portal Construction；
- Diagnostic、End-to-End、Generalization & Recovery 是评测维度；
- Single-Agent / Multi-Agent 是正交 execution modes；
- Benchmark kernel 与 Agent/baseline 解耦；
- 正式 end-to-end success 必须是当前 episode portal 的可归因 Nether entry；
- 旧 deterministic progression 只作为 scripted oracle、calibration 或 regression fixture。

旧完整状态记录位于 [docs/legacy/v1/PROJECT_STATUS_V1.md](docs/legacy/v1/PROJECT_STATUS_V1.md)。该文件是历史证据，不是 active scope。

## P0.1 architecture cleanup（offline complete）

- Roadmap phase 保持 P0–P8，Environment Validation 保持 E0–E12，Diagnostic level 保持 D1–D6；
- End-to-End difficulty 已统一为 L1–L4；所有 L-level 仍以 attributed Nether entry 为最终成功；
- 历史 `TaskInstance` 是 v1 compatibility surface（`LegacyTaskInstance` alias）；v2 taxonomy 使用 `TaskIdentity`；
- architecture guards 覆盖 benchmark/tasks 对 deterministic drivers、legacy evaluators、model-specific agents 与 legacy TaskInstance 的依赖。

## 当前唯一 active task

`P1-REAL-MINERL-ENVIRONMENT-VALIDATION`

P1 real MineRL environment validation remains the only active phase.

目标：建立并真实验证最小、可重复、可审计的 MineRL/Minecraft 实验仪器。冻结清单：E0 reset/close、E1 RGB、E2 inventory、E3 selected item、E4 camera、E5 movement、E6 block placement、E7 bucket、E8 block truth、E9 fluid truth、E10 vanilla water-lava -> obsidian、E11 portal activation、E12 dimension transition。

E10 是 calibration，不是正式 benchmark task。Evaluator 必须观察 server-side `air/lava/... -> obsidian`。

## P1-E0 / E1 / E2 / E3 / E4 / E5 / E6 / E7 status

- E0–E5 contract / offline runtime / MineRL adapter：`unit_verified`；各有一次已审查真实成功；均 `integration_verified`: NO
- E0 RGB/lifecycle evidence: `runs/p1_e0_reset_close/e0-live-20260813-130313`；E1 RGB 360×640×3 uint8: `runs/p1_e1_rgb_observation/e1-live-20260813-162733`
- E2 inventory match `{"dirt": 7, "obsidian": 4, "flint_and_steel": 1}`: `runs/p1_e2_inventory_observation/e2-live-20260813-232125`
- E3 selected item `flint_and_steel`: `runs/p1_e3_selected_item_observation/e3-live-20260814-001`
- E4 one `look(pitch=0°, yaw=+20°)`，FullStats yaw/pitch，tolerance 1.0°: `runs/p1_e4_camera_control/e4-live-20260814-001`, `camera_ok`
- E5 one `move(forward=1.0)`, FullStats xpos/ypos/zpos；#1 reset_failed `p1-e5-live-001` / `runs/p1_e5_movement/e5-live-20260814-001`（JVM SIGSEGV / Malmo EOF，不是 movement_failed）；#2 `p1-e5-live-002` / `runs/p1_e5_movement/e5-live-20260815-002`, `movement_ok`, delta z≈0.098
- E6 contract / offline runtime: complete / `unit_verified`
- E6 MineRL adapter / live bridge: implemented / offline tested
- E6 calibration: `dirt`; one `place_block(dirt)`, `duration_ticks=1`; spawn world `(0, 4, 0)`, yaw `0`, pitch `60`; target world `(0, 4, 1)` = atSpawn grid `(0, 0, 1)`; before `air` → after `dirt` from evaluator-only `portal_grid`
- E6 coordinate audit: ObservationFromGrid(`atSpawn=true`) is spawn-relative; `_cell_index_in_grid` indexes that namespace. Offline fix `unit_verified`
- E6 real attempt #1: episode `p1-e6-live-001` / `runs/p1_e6_block_placement/e6-live-20260815-001`; `real_execution_performed=true`; `success=true`; `outcome=placement_ok`; before world `(0, 4, 1)=air`; one `place_block(dirt)` `duration_ticks=1` `translation_accepted=true` `tested_action_count=1`; after world `(0, 4, 1)=dirt`; `world_changed=true`; `intended_block_present=true`; close returned; `process_release_proven=false`
- E6 reviewed real success: YES
- E6 `integration_verified`: NO
- E7 contract / offline runtime: complete / `unit_verified`
- E7 water calibration: offline verified; `water_bucket` → `bucket` and none → water
- E7 lava calibration: offline verified; `lava_bucket` → `bucket` and none → lava
- E7 MineRL adapter / live bridge: implemented / offline tested
- E7 calibration: closed WATER / LAVA variants; each is a fresh episode with exactly one `use_item`; spawn world `(0, 4, 0)`, yaw `0`, pitch `60`; target world `(0, 4, 1)` = atSpawn grid `(0, 0, 1)`; success requires public inventory transition AND evaluator-only fluid effect
- E7 WATER real attempt #1: episode `p1-e7-water-live-001` / `runs/p1_e7_bucket_usage/water/e7-water-live-20260815-001`; `real_execution_performed=true`; `success=true`; `outcome=bucket_ok`; before inventory `water_bucket=1` `bucket=0`; before fluid `none` at world `(0, 4, 1)` / atSpawn grid `(0, 0, 1)`; one `use_item(water_bucket)` `duration_ticks=1` `translation_accepted=true` `tested_action_count=1`; after inventory `water_bucket=0` `bucket=1`; after fluid `water`; `inventory_changed=true`; `bucket_consumed=true`; `empty_bucket_produced=true`; `fluid_changed=true`; `intended_fluid_present=true`; close returned; `process_release_proven=false`
- E7 WATER reviewed real success: YES
- E7 LAVA real execution: NOT RUN
- E7 `integration_verified`: NO
- E8–E12: NOT STARTED
- P1 Hard Gate: NOT PASSED
- P2: NOT STARTED

None of E0–E7 is `integration_verified`; `process_release_proven=false`. E0–E7 remain solver-independent. Typed evaluator truth never enters Agent-visible surfaces. Imports, `--check`, and unit tests do not start MineRL.

## 本次 v2 refactor 已完成

- `unit_verified`：v2 taxonomy、verification vocabulary、P1 manifest、E0–E7 contracts/offline runtimes/MineRL bridges、solver-independent kernel interfaces、catalog quarantine 与文档一致性合同；
- legacy infrastructure 保持原 import，可继续运行离线 regression；
- active catalog 不包含旧 C1–C5 正式 benchmark entries；
- `python -m obsidianlink --check` 与 `scripts/check_environment.py` 使用 v2/P1 语义，不把离线结果提升为真实 integration。标准本地运行时为 `environment.yml` 的 `mc-agent`。

## 尚未验证

- E0–E5 各有一次审查过的真实 success evidence，但稳定重复性与 OS-level process release 未证明，`integration_verified`: NO；
- E6: `unit_verified`; one reviewed real success (`p1-e6-live-001`, `placement_ok`); `integration_verified`: NO；
- E7: `unit_verified`; water/lava calibrations offline verified; adapter/live bridge implemented / offline tested; one reviewed WATER real success (`p1-e7-water-live-001`, `bucket_ok`); LAVA real execution NOT RUN; `integration_verified`: NO；
- E8–E12: NOT STARTED；P1 Hard Gate: NOT PASSED；P2: NOT STARTED；
- E10、portal activation、dimension transition 尚未真实验证；
- L1–L4、Diagnostic instances、Generalization/Recovery 与 Multi-Agent gameplay 尚未实现；没有 `benchmark_evaluated` 结果。

## P1 hard gate

P1 Hard Gate 尚未通过。进入 P2 前必须完成真实环境 validation suite，并达到稳定重复成功、`truth_missing=0`、无人工干预。规划建议至少 20 个 fresh episodes。每次真实 MineRL/Minecraft 运行与每次 Gradle 构建仍需用户单独授权。

## 下一精确任务

E7 contract, offline runtime, and MineRL adapter/live bridge are `unit_verified`. E7 WATER has one reviewed real success (`p1-e7-water-live-001`, `bucket_ok`). E7 LAVA real execution: NOT RUN. E7 `integration_verified`: NO. E0–E6 reviewed real success history is preserved. P1 Hard Gate: NOT PASSED. E8–E12: NOT STARTED. P2: NOT STARTED. Request explicit authorization for exactly ONE E7 LAVA real MineRL/Minecraft bucket calibration run. Do not start E8 automatically. Do not rerun WATER.
