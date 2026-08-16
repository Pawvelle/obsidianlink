# ObsidianLink v2.0 项目状态

更新时间：2026-08-16

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

E10 是 calibration，不是正式 benchmark task。Evaluator 必须同时观察 water-pour cell 出现 water/source，以及相邻 lava source 变为 obsidian。

## P1 E0–E11 active status

All E0–E9 contracts, offline runtimes, and MineRL adapters are `unit_verified`; each has at least one reviewed real success, but none is `integration_verified` and `process_release_proven=false` remains. E10 contract/offline runtime/MineRL bridge is `unit_verified`; E10 geometry real verified: YES; E10 real conversion reviewed success: YES; `integration_verified`: NO. E11 contract/offline runtime/MineRL bridge is `unit_verified`; E11 runtime geometry deployed / real verified: YES; E11 real activation: NOT RUN; `integration_verified`: NO.

- E0 lifecycle: `runs/p1_e0_reset_close/e0-live-20260813-130313`; E1 RGB: `runs/p1_e1_rgb_observation/e1-live-20260813-162733`.
- E2 inventory: `runs/p1_e2_inventory_observation/e2-live-20260813-232125`; E3 selected item: `runs/p1_e3_selected_item_observation/e3-live-20260814-001`; E4 camera: `runs/p1_e4_camera_control/e4-live-20260814-001`.
- E5 movement: historical attempt `p1-e5-live-001` failed during reset with JVM `SIGSEGV` / Malmo EOF; reviewed success `p1-e5-live-002` produced `movement_ok`. The failure was infrastructure, not movement capability.
- E6 placement: `p1-e6-live-001`, `placement_ok`, target `air -> dirt` from evaluator-only grid truth.
- E7 bucket use: `p1-e7-water-live-001` and `p1-e7-lava-live-001`, both `bucket_ok`; WATER + LAVA coverage complete.
- E8 block truth: historical attempt `p1-e8-live-001` failed before validation with the E5 `liblwjgl_stb` / Sound engine / `STBVorbis` fingerprint; `p1-e8-live-002` produced `block_truth_ok` with `truth_missing_count=0`.
- E9 fluid truth: historical WATER attempt `p1-e9-water-live-001` failed before validation with the same native fingerprint; `p1-e9-water-live-002` and `p1-e9-lava-live-001` produced `fluid_truth_ok`, preserving source/flowing distinction with `truth_missing_count=0`.
- Detailed calibration fields remain in the referenced run evidence and [P1 Environment Validation](docs/architecture/P1_ENVIRONMENT_VALIDATION.md); startup failure diagnosis is preserved in [P1 startup reliability root cause](docs/architecture/P1_STARTUP_RELIABILITY_ROOT_CAUSE.md).
- E10: contract/offline runtime/MineRL bridge implemented / `unit_verified`；controlled Mission geometry deployment: VERIFIED；real conversion reviewed success: YES (`p1-e10-live-001`, `obsidian_conversion_ok`)；before water=`air/none`, target=`lava/source`；stimulus 1 × `use_item(water_bucket)`；after water=`water/source`, target=`obsidian`；`truth_missing_count=0`；`integration_verified`: NO.
- E11: contract/offline runtime/MineRL bridge implemented / `unit_verified`；prebuilt obsidian frame is a **calibration fixture**, not Agent construction；runtime fixture deployment: VERIFIED；geometry real verified: YES (`e11-geometry-20260816-001`, `e11_geometry_ready`, frame 14/14 obsidian, interior 6/6 air, portal=0, fire=0)；real activation: NOT RUN；`integration_verified`: NO。
- E12: NOT STARTED; P1 Hard Gate: NOT PASSED; P2: NOT STARTED.

None of E0–E11 is `integration_verified`; `process_release_proven=false`. E0–E11 remain solver-independent. Typed evaluator truth never enters Agent-visible surfaces. Imports, `--check`, and unit tests do not start MineRL.

## 本次 v2 refactor 已完成

- `unit_verified`：v2 taxonomy、verification vocabulary、P1 manifest、E0–E11 contracts/offline runtimes/MineRL bridges、solver-independent kernel interfaces、catalog quarantine 与文档一致性合同；
- legacy infrastructure 保持原 import，可继续运行离线 regression；
- active catalog 不包含旧 C1–C5 正式 benchmark entries；
- `python -m obsidianlink --check` 与 `scripts/check_environment.py` 使用 v2/P1 语义，不把离线结果提升为真实 integration。标准本地运行时为 `environment.yml` 的 `mc-agent`。

## 尚未验证

- E0–E5 各有一次审查过的真实 success evidence，但稳定重复性与 OS-level process release 未证明，`integration_verified`: NO；
- E6: `unit_verified`; one reviewed real success (`p1-e6-live-001`, `placement_ok`); `integration_verified`: NO；
- E7: `unit_verified`; water/lava calibrations offline verified; adapter/live bridge implemented / offline tested; one reviewed WATER real success (`p1-e7-water-live-001`, `bucket_ok`); one reviewed LAVA real success (`p1-e7-lava-live-001`, `bucket_ok`); real calibration coverage WATER + LAVA complete; `integration_verified`: NO；
- E8: `unit_verified`; #1 `p1-e8-live-001` `reset_failed` SAME_FINGERPRINT native crash as E5 #1; #2 `p1-e8-live-002` `block_truth_ok`; reviewed real success: YES; `integration_verified`: NO；
- E9: `unit_verified`; WATER #1 `p1-e9-water-live-001` `reset_failed` SAME_FINGERPRINT native crash as E5 #1 / E8 #1; WATER #2 `p1-e9-water-live-002` `fluid_truth_ok`; LAVA #1 `p1-e9-lava-live-001` `fluid_truth_ok`; real calibration coverage WATER + LAVA complete; reviewed real success: YES; `integration_verified`: NO；
- E10: `unit_verified`；geometry VERIFIED；real conversion reviewed success YES (`p1-e10-live-001`, `obsidian_conversion_ok`)；`integration_verified`: NO；
- E11: `unit_verified`；runtime geometry deployed / real verified YES；real activation NOT RUN；`integration_verified`: NO；
- E12: NOT STARTED；P1 Hard Gate: NOT PASSED；P2: NOT STARTED；
- E11 offline contract 已实现，runtime geometry 已真实验证，但 flint_and_steel activation 尚未运行，不能当作 portal activation capability；
- E12 dimension transition 尚未开始；
- L1–L4、Diagnostic instances、Generalization/Recovery 与 Multi-Agent gameplay 尚未实现；没有 `benchmark_evaluated` 结果。

## P1 hard gate

P1 Hard Gate 尚未通过。进入 P2 前必须完成真实环境 validation suite，并达到稳定重复成功、`truth_missing=0`、无人工干预。规划建议至少 20 个 fresh episodes。每次真实 MineRL/Minecraft 运行与每次 Gradle 构建仍需用户单独授权。

## 下一精确任务

Before audio mitigation, P1 startup reliability calibration completed 20 fresh-process attempts: 18 success / 2 infrastructure failures (90% first-attempt success). Attempt-006 was a confirmed `liblwjgl_stb` / `Sound engine` SIGSEGV; attempt-014 was a separate unresolved mission/reset reply failure. After deploying the narrowly scoped `disable-client-audio.patch`, the independent post-mitigation validation completed 20 fresh-process attempts: 20 success / 0 failure (100% observed first-attempt success rate), with `max_reset_attempts=1` and no validation action. Failure fingerprints: none. Native crashes: 0; Malmo EOF: 0; timeouts: 0; cleanup failures: 0. No startup infrastructure failure was observed in these 20 post-mitigation attempts, but this finite sample does not prove absolute reliability. `process_release_proven=false` remains for all attempts.

E10 contract/offline runtime/MineRL bridge is `unit_verified`. Controlled Mission geometry is VERIFIED. One real E10 conversion (`p1-e10-live-001`) produced `obsidian_conversion_ok` from lava source + one `use_item(water_bucket)` to water source + obsidian, `truth_missing_count=0`. Compact evidence: `runs/history/p1-e10-live-20260816-001/`. `integration_verified`: NO. E11 contract/offline runtime/MineRL bridge is `unit_verified`. E11 runtime fixture deployment: VERIFIED. E11 geometry real verified: YES (`e11-geometry-20260816-001`, `e11_geometry_ready`, 14/14 obsidian, 6/6 air, portal=0, fire=0). Compact evidence: `runs/history/p1-e11-geometry-20260816-001/`. E11 real activation: NOT RUN. `integration_verified`: NO. Next exact task: one explicitly authorized real E11 flint_and_steel → vanilla portal activation calibration. Do not start it automatically. E12: NOT STARTED. P1 Hard Gate: NOT PASSED. P2: NOT STARTED. Attempt-014 remains historically unexplained.
