# ObsidianLink v2.0 项目状态

更新时间：2026-08-13

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

## P1-E0 status

- E0 contract: complete
- E0 offline runtime: `unit_verified`
- E0 MineRL integration bridge: implemented / offline tested
- E0 cleanup fail-closed semantics: hardened
- E0 real MineRL execution: one authorized success reviewed
- E0 `integration_verified`: NO
- E1 contract / adapter / offline runtime / live bridge: implemented / `unit_verified`
- E1 real MineRL execution: `not_run`
- E1 `integration_verified`: NO
- E2–E12: not started / not run
- P1 Hard Gate: NOT PASSED
- P2: NOT STARTED

Reviewed E0 live evidence (not modified): `runs/p1_e0_reset_close/e0-live-20260813-130313`, episode `p1-e0-live-002`. Fields: `success=true`, `outcome=lifecycle_ok`, `real_execution_performed=true`, `opened=true`, `created=true`, `reset_completed=true`, `initial_state_present=true`, `closed=true`, `error=null`, observable cleanup `close_returned`, `backend_marked_closed`, `environment_reference_cleared`, and `owner_cleared` are true. `cleanup.process_release_proven` remains false; `close()` returning is not OS-level process-release proof. There is no automatic verification promotion path; this single reviewed success does not mark E0 `integration_verified`. CLI `--check` and `p1_validation_manifest()` remain offline and still report cases as `not_run`.

The solver-independent E0/E1 runtime remains in `obsidianlink/env/validation/`. MineRL adapters live in `obsidianlink/env/integration/`. Authorized E0 entrypoint: `python -m obsidianlink.env.integration.e0_run --execution-mode authorized_live_e0 --authorized-live-run e0_reset_close`. Authorized E1 entrypoint (not executed this round): `python -m obsidianlink.env.integration.e1_run --execution-mode authorized_live_e1 --authorized-live-run e1_rgb_observation`. Import, `--check`, and unit tests do not start MineRL. Real-run success fails closed when observable cleanup signals are explicitly false.

## 本次 v2 refactor 已完成

- `unit_verified`：v2 taxonomy、verification vocabulary、P1 E0–E12 manifest、E0 lifecycle runtime、E0 MineRL integration bridge、E1 RGB observation runtime/adapter/live bridge、solver-independent kernel interfaces、catalog quarantine 与文档一致性合同；
- legacy infrastructure 保持原 import，可继续运行离线 regression；
- active catalog 不包含旧 C1–C5 正式 benchmark entries，也没有批量创建空壳 v2 task instances；
- `python -m obsidianlink --check` 与 `scripts/check_environment.py` 使用 v2/P1 语义，不执行 deterministic drivers，也不把离线 E0 提升为真实 integration。

## 本次离线验收

- 2026-08-12：离线测试 1250 项通过；该结果只支持 `unit_verified` 声明，不代表真实 Minecraft 能力；
- 2026-08-13：P0.1 cleanup 后完整离线回归 1259 项通过；仍只支持 `unit_verified` 声明；
- 2026-08-13：P1-E0 MineRL integration bridge 后完整离线回归 1292 项通过；E0 当时仍仅为 `unit_verified`；
- 2026-08-13：授权真实 E0 lifecycle run `p1-e0-live-002` 成功并已审查；`process_release_proven=false`，E0 仍不是 `integration_verified`；
- 2026-08-13：P1-E1 RGB observation contract/adapter/offline runtime/live bridge 已实现；E1 真实 MineRL 未执行；
- v2 CLI 自检与 P1 环境状态检查脚本通过，均报告 E0–E12 为 `not_run`；
- Python compile check 与 `git diff --check` 通过。
- 标准本地运行时冻结为 `environment.yml` 中的 Conda 环境 `mc-agent`；Python 命令与测试不得使用系统 Python 或其他环境，环境检查会 fail closed 验证该身份。

## 尚未验证

- E0 已有一次审查过的真实 lifecycle success evidence，但不是 `integration_verified`；OS-level process release 未证明；
- E1 contract/offline runtime/live bridge 为 `unit_verified`；真实 MineRL E1 尚未执行，不是 `integration_verified`；
- E2–E12 尚未实现或尚未运行；P1 Hard Gate 未通过；
- 真实 MineRL/Minecraft casting 不是 `integration_verified`；
- E10、portal activation、dimension transition 尚未真实验证；
- L1–L4 正式 end-to-end task 尚未实现；
- Diagnostic task instances、Generalization/Recovery engine 与 Multi-Agent gameplay 尚未实现；
- 没有 `benchmark_evaluated` 结果或正式数据集。

## P1 hard gate

P1 Hard Gate 尚未通过。进入 P2 前必须完成真实环境 validation suite，并达到稳定重复成功、`truth_missing=0`、无人工干预。规划建议至少 20 个 fresh episodes；最终次数、seed 和失败处理规则在真实实验合同冻结时确定。每次真实 MineRL/Minecraft 运行与每次 Gradle 构建仍需用户单独授权。

## 下一精确任务

Authorized real MineRL E1 RGB observation execution. P1 real MineRL environment validation still requires explicit user authorization; P1 Hard Gate has not passed and P2 must not begin. Do not start E2–E3 yet.
