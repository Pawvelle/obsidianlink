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

## P1-E0 / E1 / E2 / E3 status

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
- E3 observed selected item source: backend public `Observation.selected_item`
- E3 real MineRL execution: NOT RUN / awaiting explicit authorization
- E3 `integration_verified`: NO
- E4–E12: not started / not run
- P1 Hard Gate: NOT PASSED
- P2: NOT STARTED

Reviewed E0 live evidence (not modified): `runs/p1_e0_reset_close/e0-live-20260813-130313`, episode `p1-e0-live-002`. `success=true`, `outcome=lifecycle_ok`, `real_execution_performed=true`; opened/reset/initial state/closed are true. Observable cleanup succeeded; `process_release_proven=false`.

Reviewed E1 live evidence (not modified): `runs/p1_e1_rgb_observation/e1-live-20260813-162733`, episode `p1-e1-live-001`. `success=true`, `outcome=rgb_ok`, `real_execution_performed=true`; `opened=true`, `reset_completed=true`, `initial_state_present=true`, `rgb_present=true`, `rgb_height=360`, `rgb_width=640`, `rgb_channels=3`, `rgb_dtype=uint8`, `closed=true`, `error=null`. Observable cleanup `close_returned`, `backend_marked_closed`, `environment_reference_cleared`, and `owner_cleared` are true. `process_release_proven` remains false and is not rewritten.

Reviewed E2 live evidence (not modified): `runs/p1_e2_inventory_observation/e2-live-20260813-232125`, episode `p1-e2-live-001`. `success=true`, `outcome=inventory_ok`, `real_execution_performed=true`; expected and observed inventory both equal `{"dirt": 7, "obsidian": 4, "flint_and_steel": 1}`, and `inventory_matches_expected=true`. Opened/reset/initial state/closed are true; observable cleanup succeeded and `process_release_proven=false`.

E0/E1/E2/E3 are not promoted to `integration_verified`. Existing policy: live runtimes and `p1_validation_manifest()` / `--check` are fail-closed surfaces and never emit that claim; AGENTS.md forbids declaring `integration_verified` before the P1 Hard Gate; a single reviewed success is recorded evidence, not suite promotion. `benchmark_evaluated` is not claimed.

The solver-independent E0/E1/E2/E3 runtime remains in `obsidianlink/env/validation/`. MineRL adapters live in `obsidianlink/env/integration/`. E3's adapter projects only backend public `Observation.selected_item`; it does not read initial inventory, hotbar mapping, action intent, raw `equipped_items`, or expected calibration state. Expected and observed values remain independent. Authorized E3 command is recorded below but has not been run. Import, `--check`, and unit tests do not start MineRL.

## 本次 v2 refactor 已完成

- `unit_verified`：v2 taxonomy、verification vocabulary、P1 E0–E12 manifest、E0 lifecycle runtime、E0 MineRL integration bridge、E1 RGB observation runtime/adapter/live bridge、E2 inventory observation runtime/adapter/live bridge、E3 selected-item contract/runtime/adapter/live bridge、solver-independent kernel interfaces、catalog quarantine 与文档一致性合同；
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
- v2 CLI 自检与 P1 环境状态检查脚本通过，均报告 E0–E12 为 `not_run`；
- Python compile check 与 `git diff --check` 通过。
- 标准本地运行时冻结为 `environment.yml` 中的 Conda 环境 `mc-agent`；Python 命令与测试不得使用系统 Python 或其他环境，环境检查会 fail closed 验证该身份。

## 尚未验证

- E0 已有一次审查过的真实 lifecycle success evidence，但不是 `integration_verified`；OS-level process release 未证明；
- E1 已有一次审查过的真实 RGB success evidence（360×640×3 uint8），但不是 `integration_verified`；
- E2 已有一次审阅通过的真实 inventory success evidence，但不是 `integration_verified`；
- E3 offline implementation 已完成，但真实 MineRL/Minecraft E3 尚未运行，observed selected item 尚无 live evidence，`integration_verified`: NO；
- E4–E12 尚未实现或尚未运行；P1 Hard Gate 未通过；
- 真实 MineRL/Minecraft casting 不是 `integration_verified`；
- E10、portal activation、dimension transition 尚未真实验证；
- L1–L4 正式 end-to-end task 尚未实现；
- Diagnostic task instances、Generalization/Recovery engine 与 Multi-Agent gameplay 尚未实现；
- 没有 `benchmark_evaluated` 结果或正式数据集。

## P1 hard gate

P1 Hard Gate 尚未通过。进入 P2 前必须完成真实环境 validation suite，并达到稳定重复成功、`truth_missing=0`、无人工干预。规划建议至少 20 个 fresh episodes；最终次数、seed 和失败处理规则在真实实验合同冻结时确定。每次真实 MineRL/Minecraft 运行与每次 Gradle 构建仍需用户单独授权。

## 下一精确任务

E3 offline implementation ready -> waiting for explicit authorized real MineRL run.

唯一后续 E3 live command（尚未执行，且每次执行仍需用户单独明确授权）：

```bash
/opt/anaconda3/bin/conda run -n mc-agent python -m obsidianlink.env.integration.e3_run --execution-mode authorized_live_e3 --authorized-live-run e3_selected_item --episode-id p1-e3-live-001 --output-dir /Users/joey/Documents/Projects/ObsidianLink/ObsidianLink/runs/p1_e3_selected_item_observation/e3-live-20260814-001
```

Do not start E4. P1 Hard Gate has not passed and P2 must not begin.
