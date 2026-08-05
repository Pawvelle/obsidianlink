# 当前状态

更新时间：2026-08-05

## 当前唯一目标

任务：`B0-BENCHMARK-SCOPE-FREEZE`（文档范围冻结已完成）

下一任务：`R5-CONTINUOUS-CASTING`（尚未开始）

当前 active implementation：Casting-S-C1 / `casting_c1_fixed`（FakeBackend 离线能力、evaluator 和 deterministic driver）。兼容任务 ID、实例文件、workflow 和 schema 均保持不变。

## B0 目标

把 ObsidianLink 的长期定位冻结为可复现的 Minecraft 单智能体与多智能体 Benchmark，统一评测至少一名指定 Agent 通过当前 episode 内完成建造、修复或激活的传送门进入 Nether。长期范围包含：

- Casting、Ruined Portal、Adaptive Routing 三个任务族；
- Single-Agent、Multi-Agent 两种与任务族正交的模式；
- family/mode/level/layout 命名规则与显式难度参数；
- 通用、Adaptive、Multi-Agent 指标；
- Agent-visible、各 Agent 私有状态与 evaluator-only truth 的严格隔离；
- 统一数据、证据和审计协议。

长期 scope 不等于当前支持范围。B0 只修改文档和设计，不实现后续功能。

## 本轮交付

- [README.md](README.md)：区分 Benchmark Vision 与 Current Implementation；
- [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md)：升级为权威总规范；
- [ROADMAP.md](ROADMAP.md)：加入 B0 和四个 suite/发布路线；
- [DATASET_CARD.md](DATASET_CARD.md)：加入未来统一 run 元数据；
- [PROJECT_STATUS.md](PROJECT_STATUS.md)：把唯一任务切换到 B0；
- [TASK_TAXONOMY.md](docs/benchmark/TASK_TAXONOMY.md)：冻结分类、层级、模式、命名和难度维度；
- [`casting_c1_fixed` 任务页](docs/tasks/casting/casting_c1_fixed.md)：保存单块任务具体合同；
- [FIRST_OBSIDIAN_BLOCK.md](docs/runbooks/FIRST_OBSIDIAN_BLOCK.md)：同步 R2–R4 已完成状态和 B0 边界；
- [AGENTS.md](AGENTS.md)：最小增加 scope/阶段、分类和信息隔离约束。

## 明确不在 B0 范围

- 不实现 R5 连续浇筑；
- 不实现废弃传送门环境或修复逻辑；
- 不实现 Adaptive planner/evaluator；
- 不实现 Multi-Agent observation、通信或协作；
- 不修改 EnvironmentBackend、driver、evaluator、任务 schema 或依赖；
- 不重命名或删除 `casting_c1_fixed`；
- 不生成真实 runs，不调用模型 API；
- 禁止真实 MineRL、Gradle 和模型调用；
- 不启动 Minecraft，不修改 `vendor/minerl`，不 commit，不 push。

## 已完成历史（保留）

### R1 — 任务合同

`casting_c1_fixed` 的 seed、Single-Agent 身份、固定出生点、初始资源、目标 cell、里程碑、预算和禁止世界修改规则已经冻结。

### R2-CAPABILITY-MANIFEST — 后端能力清单

不可变 `BackendCapabilities`、稳定 capability IDs、FakeBackend 正反例、reset 前 fail-closed gate 和 JSON 快照已经完成。真实 backend 缺关键能力时不得开始 casting episode。

### R3 — 单块 evaluator

`CastingEvaluationState`、`CastingEvaluationResult` 和独立 evaluator 已完成。稳定 outcome 包含 `success`、`in_progress`、`wrong_block`、`truth_missing`、step/time budget、invalid initial state、causality missing 和 abnormal termination；evaluator-only truth 不进入 Agent observation。

### R4-DETERMINISTIC-CASTING-DRIVER — 确定性单块 driver

公共动作协议、封闭白名单、有限计划/step/time/wait/recovery、后端异常 fail closed、确定性重放和 driver/evaluator 隔离已经在 FakeBackend 上完成。Driver 的 `completed` 不是 evaluator 的 `success`。

## B0 完成条件

- 总目标、三个 task family、两种 agent mode 已冻结；
- 命名、难度维度、指标、数据划分和证据协议已冻结为文档设计；
- 核心文档一致区分长期愿景与当前实现；
- `casting_c1_fixed` 具体语义已迁移保存并保持兼容；
- 未提前实现任何后续阶段代码；
- 相关离线检查通过，或明确报告本机固定运行时缺失造成的未验证项。

## 当前限制

- 当前只可宣称 `casting_c1_fixed` 的 FakeBackend 离线切片；
- 真实 MineRL 水/熔岩浇筑尚未验证；
- 完整门框、点火、Nether entry、Ruined、Adaptive 和 Multi-Agent 均未实现；
- 当前仓库没有正式真实 Benchmark 数据集；
- B0 设计的未来元数据字段尚未写入 schema。

## 下一任务

`R5-CONTINUOUS-CASTING` 仍是下一工程任务，因为 C1 单块能力之后必须先用 FakeBackend 和确定性 driver 证明多个目标 cell 的连续原版浇筑、独立因果归属、部分完成和有限恢复，才能进入完整门框、其他路线或模型阶段。开始 R5 前必须由新的任务请求更新本文件；不得因 B0 已描述长期 scope 而提前开发。
