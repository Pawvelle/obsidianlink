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

## 本次 v2 refactor 已完成

- `unit_verified`：v2 taxonomy、verification vocabulary、P1 E0–E12 manifest、solver-independent kernel interfaces、catalog quarantine 与文档一致性合同；
- legacy infrastructure 保持原 import，可继续运行离线 regression；
- active catalog 不包含旧 C1–C5 正式 benchmark entries，也没有批量创建空壳 v2 task instances；
- `python -m obsidianlink --check` 与 `scripts/check_environment.py` 使用 v2/P1 语义，不执行 deterministic drivers。

## 本次离线验收

- 2026-08-12：离线测试 1250 项通过；该结果只支持 `unit_verified` 声明，不代表真实 Minecraft 能力；
- 2026-08-13：P0.1 cleanup 后完整离线回归 1259 项通过；仍只支持 `unit_verified` 声明；
- v2 CLI 自检与 P1 环境状态检查脚本通过，均报告 E0–E12 为 `not_run`；
- Python compile check 与 `git diff --check` 通过。
- 标准本地运行时冻结为 `environment.yml` 中的 Conda 环境 `mc-agent`；Python 命令与测试不得使用系统 Python 或其他环境，环境检查会 fail closed 验证该身份。

## 尚未验证

- P1 任一 E0–E12 case 尚未在本次重构中运行；
- 真实 MineRL/Minecraft casting 不是 `integration_verified`；
- E10、portal activation、dimension transition 尚未真实验证；
- L1–L4 正式 end-to-end task 尚未实现；
- Diagnostic task instances、Generalization/Recovery engine 与 Multi-Agent gameplay 尚未实现；
- 没有 `benchmark_evaluated` 结果或正式数据集。

## P1 hard gate

进入 P2 前必须完成真实环境 validation suite，并达到稳定重复成功、`truth_missing=0`、无人工干预。规划建议至少 20 个 fresh episodes；最终次数、seed 和失败处理规则在真实实验合同冻结时确定。每次真实 MineRL/Minecraft 运行与每次 Gradle 构建仍需用户单独授权。

## 下一精确任务

P1 real MineRL environment validation, beginning with the smallest controlled integration case and only running real MineRL after explicit user authorization.
