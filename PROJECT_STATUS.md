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
- End-to-End difficulty 已统一为 L1–L4，不再复用 Roadmap 的 P-prefix；所有 L-level 仍以 attributed Nether entry 为最终成功；
- 历史 `TaskInstance` 明确为 v1 compatibility surface，并提供 `LegacyTaskInstance` alias；v2 canonical taxonomy 使用 `TaskIdentity`；
- architecture guards 覆盖 benchmark/tasks 对 deterministic drivers、legacy evaluators、model-specific agents 与 legacy TaskInstance 的依赖；
- 本次 cleanup 不改变 active phase，不创建 task instance，也不新增 capability 或 verification claim。

## 当前唯一 active task

`P1-REAL-MINERL-ENVIRONMENT-VALIDATION`

P1 real MineRL environment validation remains the only active phase.

目标：建立并真实验证最小、可重复、可审计的 MineRL/Minecraft 实验仪器，而不是继续修补旧 36-step C1 solver。

冻结清单：

- E0 reset / close
- E1 RGB observation
- E2 inventory observation
- E3 selected item
- E4 camera control
- E5 movement
- E6 block placement
- E7 bucket usage
- E8 server-side block truth
- E9 water/lava fluid truth
- E10 vanilla water-lava -> obsidian
- E11 portal activation
- E12 dimension transition

E10 允许预置合法 support/trench，并由 deterministic calibration script 只执行最小流体交互。Evaluator 必须观察 server-side `air/lava/... -> obsidian`。它验证 `MineRL Action -> Minecraft Server -> Vanilla Mechanics -> Evaluator-only World Truth -> Verdict`，不是正式 benchmark task。

## P1-E0 / E1 / E2 / E3 / E4 / E5 status

- E0 contract: complete
- E0 offline runtime: `unit_verified`
- E0 MineRL integration bridge: implemented / offline tested
- E0 real MineRL execution: one authorized success reviewed
- E0 `integration_verified`: NO
- E1 contract / adapter / offline runtime / live bridge: `unit_verified`
- E1 real MineRL execution: one authorized RGB success reviewed
- E1 RGB: 360×640×3 uint8
- E1 `integration_verified`: NO
- E2 contract / offline runtime: complete / `unit_verified`
- E2 MineRL adapter / live bridge: implemented / offline tested
- E2 real MineRL execution: one authorized inventory success reviewed
- E2 observed inventory: `{"dirt": 7, "obsidian": 4, "flint_and_steel": 1}`
- E2 `inventory_matches_expected`: true
- E2 `integration_verified`: NO
- E3 contract / offline runtime: complete / `unit_verified`
- E3 MineRL adapter / live bridge: implemented / offline tested
- E3 expected selected item: `flint_and_steel`
- E3 observed selected item: `flint_and_steel`
- E3 `selected_item_matches_expected`: true
- E3 observed selected item source: backend public `Observation.selected_item`
- E3 real MineRL execution: one authorized success reviewed
- E3 `integration_verified`: NO
- E4 contract / offline runtime: complete / `unit_verified`
- E4 MineRL adapter / live bridge: implemented / offline tested
- E4 calibration: exactly one `look(pitch=0°, yaw=+20°)` action
- E4 evaluator-only orientation source: MineRL FullStats `info.location_stats.yaw/pitch`, retained by the backend and never copied into Agent-visible observations
- E4 yaw/pitch tolerance: 1.0° inclusive; yaw difference uses signed shortest-angle normalization across ±180°
- E4 real MineRL execution: one authorized success reviewed (`camera_ok`); requested yaw/pitch `20.0/0.0`, before `0.0/0.0`, after `20.000002/0.0`, normalized yaw delta `20.000001999999995°`, pitch delta `0.0°`
- E4 `integration_verified`: NO
- E5 contract / offline runtime: complete / `unit_verified`
- E5 MineRL adapter / live bridge: implemented / offline tested
- E5 calibration: exactly one `move(forward=1.0, strafe=0.0, sprint=false, jump=false)`, `duration_ticks=1`; no settle step
- E5 evaluator-only truth: MineRL FullStats `info.location_stats.xpos/ypos/zpos`; reset yaw comes from the existing FullStats orientation bridge and defines Minecraft forward as `(-sin(yaw), cos(yaw))` in X/Z
- E5 frozen bounds: minimum horizontal/forward `0.02` block; maximum lateral `0.02`, horizontal `0.5`, vertical `0.25` block
- E5 authorized real attempt reviewed: episode `p1-e5-live-001`, evidence `runs/p1_e5_movement/e5-live-20260814-001`
- E5 attempt execution state: `real_execution_performed=true`, `success=false`, `outcome=reset_failed`, `reset_completed=false`, `initial_state_present=false`, `tested_action_count=0`
- E5 movement action not executed; reviewed real success: NO. The reset failure is not a movement capability negative result
- E5 attempt #2: episode `p1-e5-live-002`, evidence `runs/p1_e5_movement/e5-live-20260815-002`; real execution, `success=true`, `movement_ok`
- E5 #2 truth: before `(0.5,4.0,0.5,yaw=0.0)`, after `(0.5,4.0,0.5980000033676625)`, delta `(0.0,0.0,0.09800000336766246)`; horizontal/total/forward `0.09800000336766246`, lateral `0.0`
- E5 #2: one action accepted; movement/direction/lateral/vertical/teleport verdicts true; cleanup signals true; `process_release_proven=false`; one reviewed real movement success
- E5 `integration_verified`: NO
- E6–E12: NOT STARTED
- P1 Hard Gate: NOT PASSED
- P2: NOT STARTED

Reviewed live evidence (not modified):

- E0: `runs/p1_e0_reset_close/e0-live-20260813-130313`, episode `p1-e0-live-002`, `lifecycle_ok`.
- E1: `runs/p1_e1_rgb_observation/e1-live-20260813-162733`, episode `p1-e1-live-001`, `rgb_ok`, 360×640×3 uint8.
- E2: `runs/p1_e2_inventory_observation/e2-live-20260813-232125`, episode `p1-e2-live-001`, exact expected/observed inventory match.
- E3: `runs/p1_e3_selected_item_observation/e3-live-20260814-001`, episode `p1-e3-live-001`, exact `flint_and_steel` match.
- E4: `runs/p1_e4_camera_control/e4-live-20260814-001`, episode `p1-e4-live-001`, `camera_ok`.
- E5: `runs/p1_e5_movement/e5-live-20260814-001`, episode `p1-e5-live-001`, production path reached; reset failed before initial state or movement action.

E0–E4 each have one reviewed success. E5 #2 is its first reviewed movement success. None is `integration_verified`; `process_release_proven=false`.

E5 root cause: Minecraft/JVM native `SIGSEGV`, `Sound engine`, `liblwjgl_stb.dylib`; then Malmo socket EOF, surfaced in MineRL Python as `TypeError: a bytes-like object is required, not 'NoneType'`. It is `reset_failed`, not `movement_failed`.

E0/E1/E2/E3/E4 are not promoted to `integration_verified`. Existing policy: live runtimes and `p1_validation_manifest()` / `--check` are fail-closed surfaces and never emit that claim; AGENTS.md forbids declaring `integration_verified` before the P1 Hard Gate; a single reviewed success is recorded evidence, not suite promotion. `benchmark_evaluated` is not claimed.

E0–E5 validation remains solver-independent. E4 submits one bounded look; E5 submits one bounded move. Their observed truth independently comes from backend-retained FullStats. Raw `location_stats` and typed evaluator truth never enter Agent-visible surfaces. Imports, `--check`, and unit tests do not start MineRL.

## 本次 v2 refactor 已完成

- `unit_verified`：v2 taxonomy、verification vocabulary、P1 manifest、E0–E5 contracts/offline runtimes/MineRL bridges、solver-independent kernel interfaces、catalog quarantine 与文档一致性合同；
- legacy infrastructure 保持原 import，可继续运行离线 regression；
- active catalog 不包含旧 C1–C5 正式 benchmark entries，也没有批量创建空壳 v2 task instances；
- `python -m obsidianlink --check` 与 `scripts/check_environment.py` 使用 v2/P1 语义，不执行 deterministic drivers，也不把离线 E0 提升为真实 integration。

## 本次离线验收

- 历史 reviewed live evidence：E0 lifecycle、E1 RGB、E2 inventory、E3 selected-item、E4 camera 各一次成功；均不提升为 `integration_verified`，`process_release_proven=false`；
- E5 movement contract、FullStats truth、offline runtime、MineRL adapter/live bridge 已实现并保持 `unit_verified`；
- E5 reset-failure audit hardening 已完成：planned/actual action count 分离；`failure_stage`、`original_exception_type`、chained traceback、`reset_attempt_count`、`environment_launch_count`、runtime log manifest/SHA-256、fail-closed cause classification；
- hardening 验收：E5 targeted 25 项、E0–E5 regression 214 项、完整离线回归 1474 项及 CLI/environment/compile/diff checks 通过；
- 标准本地运行时为 `environment.yml` 的 `mc-agent`；环境检查 fail closed 验证该身份。

## 尚未验证

- E0 已有一次审查过的真实 lifecycle success evidence，但不是 `integration_verified`；OS-level process release 未证明；
- E1 已有一次审查过的真实 RGB success evidence（360×640×3 uint8），但不是 `integration_verified`；
- E2 已有一次审阅通过的真实 inventory success evidence，但不是 `integration_verified`；
- E3 已有一次审阅通过的真实 selected-item success evidence，但稳定重复性与 OS-level process release 尚未证明，`integration_verified`: NO；
- E4 已有一次审阅通过的真实 camera success evidence，但稳定重复性与 OS-level process release 尚未证明，`integration_verified`: NO；
- E5 now has one reviewed real movement success, but stable repetition and OS-level process release remain unverified; E5 `integration_verified`: NO；
- E6–E12: NOT STARTED；P1 Hard Gate: NOT PASSED；P2: NOT STARTED；
- 真实 MineRL/Minecraft casting 不是 `integration_verified`；
- E10、portal activation、dimension transition 尚未真实验证；
- L1–L4 正式 end-to-end task 尚未实现；
- Diagnostic task instances、Generalization/Recovery engine 与 Multi-Agent gameplay 尚未实现；
- 没有 `benchmark_evaluated` 结果或正式数据集。

## P1 hard gate

P1 Hard Gate 尚未通过。进入 P2 前必须完成真实环境 validation suite，并达到稳定重复成功、`truth_missing=0`、无人工干预。规划建议至少 20 个 fresh episodes；最终次数、seed 和失败处理规则在真实实验合同冻结时确定。每次真实 MineRL/Minecraft 运行与每次 Gradle 构建仍需用户单独授权。

## 下一精确任务

E5 has one reviewed real movement success and remains `unit_verified`; `integration_verified`: NO. P1 Hard Gate: NOT PASSED. E6–E12: NOT STARTED. P2: NOT STARTED. Do not start E6. Any further real MineRL/Minecraft run requires new explicit single-run authorization.
