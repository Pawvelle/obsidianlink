# ObsidianLink v2.0 项目状态

更新时间：2026-08-14

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

## P1-E0 / E1 / E2 / E3 / E4 status

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
- E5–E12: not started / not run
- P1 Hard Gate: NOT PASSED
- P2: NOT STARTED

Reviewed live evidence (not modified):

- E0: `runs/p1_e0_reset_close/e0-live-20260813-130313`, episode `p1-e0-live-002`, `lifecycle_ok`.
- E1: `runs/p1_e1_rgb_observation/e1-live-20260813-162733`, episode `p1-e1-live-001`, `rgb_ok`, 360×640×3 uint8.
- E2: `runs/p1_e2_inventory_observation/e2-live-20260813-232125`, episode `p1-e2-live-001`, exact expected/observed inventory match.
- E3: `runs/p1_e3_selected_item_observation/e3-live-20260814-001`, episode `p1-e3-live-001`, exact `flint_and_steel` match.
- E4: `runs/p1_e4_camera_control/e4-live-20260814-001`, episode `p1-e4-live-001`, `camera_ok`.

All five have `success=true`, `real_execution_performed=true`, clean lifecycle/cleanup, and `process_release_proven=false`; none is `integration_verified`.

E0/E1/E2/E3/E4 are not promoted to `integration_verified`. Existing policy: live runtimes and `p1_validation_manifest()` / `--check` are fail-closed surfaces and never emit that claim; AGENTS.md forbids declaring `integration_verified` before the P1 Hard Gate; a single reviewed success is recorded evidence, not suite promotion. `benchmark_evaluated` is not claimed.

E0–E4 validation remains solver-independent. E4 submits one protocol-validated `MacroAction("look")`; its before/after truth independently comes from backend-retained FullStats. Raw `location_stats` and typed camera truth never enter Agent-visible surfaces. Imports, `--check`, and unit tests do not start MineRL.

## 本次 v2 refactor 已完成

- `unit_verified`：v2 taxonomy、verification vocabulary、P1 E0–E12 manifest、E0 lifecycle runtime、E0 MineRL integration bridge、E1 RGB observation runtime/adapter/live bridge、E2 inventory observation runtime/adapter/live bridge、E3 selected-item contract/runtime/adapter/live bridge、E4 camera contract/runtime/adapter/live bridge、solver-independent kernel interfaces、catalog quarantine 与文档一致性合同；
- legacy infrastructure 保持原 import，可继续运行离线 regression；
- active catalog 不包含旧 C1–C5 正式 benchmark entries，也没有批量创建空壳 v2 task instances；
- `python -m obsidianlink --check` 与 `scripts/check_environment.py` 使用 v2/P1 语义，不执行 deterministic drivers，也不把离线 E0 提升为真实 integration。

## 本次离线验收

- 2026-08-12：离线测试 1250 项通过；该结果只支持 `unit_verified` 声明，不代表真实 Minecraft 能力；
- 2026-08-13：P0.1 cleanup 后完整离线回归 1259 项通过；仍只支持 `unit_verified` 声明；
- 2026-08-13：P1-E0 MineRL integration bridge 后完整离线回归 1292 项通过；E0 当时仍仅为 `unit_verified`；
- 2026-08-13：授权真实 E0 lifecycle run `p1-e0-live-002` 成功并已审查；`process_release_proven=false`，E0 仍不是 `integration_verified`；
- 2026-08-13：P1-E1 RGB observation contract/adapter/offline runtime/live bridge 已实现；
- 2026-08-13：授权真实 E1 RGB run `p1-e1-live-001` 成功并已审查（360×640×3 uint8）；`process_release_proven=false`，E1 仍不是 `integration_verified`；
- 2026-08-13：P1-E2 inventory observation contract/adapter/offline runtime/live bridge 已实现；
- 2026-08-13：唯一一次授权真实 E2 inventory run `p1-e2-live-001` 成功并已审阅；expected/observed inventory 精确一致，`process_release_proven=false`，E2 仍不是 `integration_verified`；
- 2026-08-14：P1-E3 selected-item contract、offline runtime、MineRL adapter 与显式授权 live bridge 已实现；E3 targeted tests 32 项、E0–E3 regression 166 项通过；仅支持 `unit_verified`；
- 2026-08-14：E3 完成后完整离线回归 1426 项通过；该结果不代表真实 Minecraft capability；
- 2026-08-14：唯一一次授权真实 E3 selected-item run `p1-e3-live-001` 成功并已审阅；expected/observed 均为 `flint_and_steel`，`selected_item_matches_expected=true`，可观察 cleanup 成功，`process_release_proven=false`，E3 仍不是 `integration_verified`；
- 2026-08-14：P1-E4 camera contract、offline runtime、evaluator-only FullStats orientation bridge、MineRL adapter 与显式授权 live bridge 已实现；E4 targeted tests 23 项、E0–E4 regression 189 项、action/backend regression 44 项通过；仅支持 `unit_verified`；
- 2026-08-14：E4 offline 完成时完整离线回归 1449 项通过；当时未运行真实 MineRL/Minecraft；
- 2026-08-14：唯一一次授权真实 E4 camera run `p1-e4-live-001` 成功并已审阅；一次 bounded look、FullStats before/after、方向、1° tolerance、lifecycle 与 cleanup 均通过；`process_release_proven=false`，E4 仍不是 `integration_verified`；
- v2 CLI 自检与 P1 环境状态检查脚本通过，均报告 E0–E12 为 `not_run`；
- Python compile check 与 `git diff --check` 通过。
- 标准本地运行时冻结为 `environment.yml` 中的 Conda 环境 `mc-agent`；Python 命令与测试不得使用系统 Python 或其他环境，环境检查会 fail closed 验证该身份。

## 尚未验证

- E0 已有一次审查过的真实 lifecycle success evidence，但不是 `integration_verified`；OS-level process release 未证明；
- E1 已有一次审查过的真实 RGB success evidence（360×640×3 uint8），但不是 `integration_verified`；
- E2 已有一次审阅通过的真实 inventory success evidence，但不是 `integration_verified`；
- E3 已有一次审阅通过的真实 selected-item success evidence，但稳定重复性与 OS-level process release 尚未证明，`integration_verified`: NO；
- E4 已有一次审阅通过的真实 camera success evidence，但稳定重复性与 OS-level process release 尚未证明，`integration_verified`: NO；
- E5–E12 尚未实现或尚未运行；P1 Hard Gate 未通过；
- 真实 MineRL/Minecraft casting 不是 `integration_verified`；
- E10、portal activation、dimension transition 尚未真实验证；
- L1–L4 正式 end-to-end task 尚未实现；
- Diagnostic task instances、Generalization/Recovery engine 与 Multi-Agent gameplay 尚未实现；
- 没有 `benchmark_evaluated` 结果或正式数据集。

## P1 hard gate

P1 Hard Gate 尚未通过。进入 P2 前必须完成真实环境 validation suite，并达到稳定重复成功、`truth_missing=0`、无人工干预。规划建议至少 20 个 fresh episodes；最终次数、seed 和失败处理规则在真实实验合同冻结时确定。每次真实 MineRL/Minecraft 运行与每次 Gradle 构建仍需用户单独授权。

## 下一精确任务

P1 E4 camera-control is complete through one reviewed authorized real MineRL calibration success. Stop here; do not start E5.

本次唯一授权并已执行的 E4 live command（不得在没有新授权时重跑）：

```bash
/opt/anaconda3/bin/conda run -n mc-agent python -m obsidianlink.env.integration.e4_run --execution-mode authorized_live_e4 --authorized-live-run e4_camera_control --episode-id p1-e4-live-001 --output-dir /Users/joey/Documents/Projects/ObsidianLink/ObsidianLink/runs/p1_e4_camera_control/e4-live-20260814-001
```

E4 has one reviewed live success but is not `integration_verified`. E5 movement is not started. P1 Hard Gate has not passed and P2 must not begin.
