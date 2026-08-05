# 当前状态

更新时间：2026-08-05

## 当前唯一目标

任务：`B1-TASK-CATALOG-FOUNDATION`（兼容任务目录与严格验证已完成）

下一任务：`R6-COMPLETE-PORTAL-FRAME`（尚未开始）

当前 active implementation：Casting-S-C2 / fixed 的 `casting_c3_fixed`。旧 ID 中的 `c3` 表示三个 target cell，不表示 B0 taxonomy 的 C3；文档级兼容名称为 `casting_s_c2_fixed`。`casting_c1_fixed` 保留为 Casting-S-C1 回归合同。

## B1 已完成

1. 新增权威 [`benchmark/catalog/tasks.json`](benchmark/catalog/tasks.json)，统一 canonical name、compatibility ID、task instance ID、workflow、taxonomy、实例/实验路径、实现状态、发布可见性和 live-run policy。
2. 新增严格 [`TaskCatalog`](obsidianlink/core/task_catalog.py) loader：frozen 类型、未知字段拒绝、family/level 匹配、唯一性、安全相对路径和 active entry 约束全部 fail closed。
3. `validate_catalog_references()` 校验所有 task/experiment 路径存在，task ID、workflow、taxonomy、compatibility name、实验引用和 live-run policy 相互一致。
4. 现有 `casting_c1_fixed` / `casting_c3_fixed` 保持原 ID、workflow 和路径；正式分类分别是 Casting-S-C1 / C2。
5. `route_a_a0_development` / `route_a_a0_phase3` 保持历史路径，但明确分类为 `calibration`、`benchmark_visible=false`、`legacy_regression`，不得混入正式 Benchmark 指标。
6. CLI `--check` 与 `scripts/check_environment.py` 从 catalog 解析 active task 和 taxonomy，不再各自硬编码分类，并验证 catalog 的所有文件引用。
7. 新增 [TASK_REGISTRY.md](docs/architecture/TASK_REGISTRY.md)，冻结 R6 起的 canonical 目录规则和历史迁移边界。
8. B1 没有移动历史文件，没有修改 evaluator、driver、backend、任务语义或依赖。

## R5 冻结合同

- workflow / task ID：保留历史兼容 ID `casting_c3_fixed`；
- taxonomy：family=`casting`、mode=`single`、level=`C2`、layout=`fixed`；
- 目标：三个冻结、有序 cell `[2,4,3]`、`[3,4,3]`、`[4,4,3]`；
- 初始资源：water_bucket=3、lava_bucket=3、cobblestone=6；
- task 预算：240 environment steps、180 秒 game time、最多 1 次 model call；
- driver：默认 72 步固定 plan、最多 96 wait、per-action recovery≤2、total recovery≤8；
- 动作白名单：`equip_item` / `use_item` / `place_block` / `wait`；
- outcome：`success` / `in_progress` / `partial_completion` / `wrong_block` / `truth_missing` / `step_budget_exceeded` / `time_budget_exceeded` / `invalid_initial_state` / `causality_missing` / `abnormal_termination`；
- 世界变化只能来自 Agent 白名单动作和原版水/熔岩方块更新。

完整任务规则见 [`casting_c3_fixed` 任务页](docs/tasks/casting/casting_c3_fixed.md)。

## R5 已完成

1. `ContinuousCastingCellTruth`、`ContinuousCastingEvaluationState`、`ContinuousCastingEvaluationResult` 和 `ContinuousCastingEvaluator` 均为不可变、类型严格、可序列化的 evaluator-only 表面。
2. Evaluator 严格要求三个冻结、有序 cell；每个 cell 的 relevant actions、水、熔岩和 transition 证据独立归属，拒绝跨 cell 重复 action step。
3. 完成非空严格有序前缀返回 `partial_completion`；中间空洞或零进展不能冒充成功。
4. `run_casting_c3_driver` 使用公共 `MacroAction`、封闭白名单、有限计划和有限预算，不调用模型、不执行代码或命令。
5. 恢复只响应类型受控的 `RecoverableBackendError`，同时受 per-action 和 total recovery 硬上限约束；其他异常 fail closed。
6. Driver 不 import continuous evaluator、不调用 truth set/get 表面、不读取 observation 上的 target/truth/outcome 字段；专项 AST、spy 和 observation guard 测试锁定隔离。
7. FakeBackend truth 注入接口校验 `episode_id`、`step_id`、`agent_id`，并在 reset/step/close 后清空陈旧状态。
8. 任务实例、CLI 和环境检查现已显式报告 Casting-S-C2 taxonomy，同时保留 `casting_c3_fixed` 兼容 ID。
9. C1/C2 任务页、README、BENCHMARK_SPEC、ROADMAP、DATASET_CARD 和 taxonomy 已与实际离线能力统一。

## 已完成历史（保留）

### R1 — `casting_c1_fixed` 任务合同

固定 seed、Single-Agent、出生点、资源、单 target cell、里程碑、预算和禁止 evaluator/命令改世界规则已冻结。

### R2-CAPABILITY-MANIFEST — 后端能力清单

不可变 `BackendCapabilities`、稳定 capability IDs、FakeBackend 正反例、reset 前 fail-closed gate 和 JSON 快照已完成。真实 backend 缺关键能力时不得开始 casting episode。

### R3 — 单块 evaluator

`CastingEvaluationState`、`CastingEvaluationResult`、稳定 outcome、有限因果窗口和 evaluator-only truth 隔离已完成。

### R4-DETERMINISTIC-CASTING-DRIVER — 单块 driver

公共动作协议、封闭白名单、有限 plan/step/time/wait、确定性重放和 driver/evaluator 隔离已在 FakeBackend 完成。

### B0-BENCHMARK-SCOPE-FREEZE — 总范围冻结

三个 task family、两种 agent mode、C/R/A 能力层级、命名规范、难度维度、指标和数据证据协议已冻结。长期 scope 与 active implementation 分开。

### R5-CONTINUOUS-CASTING — 连续浇筑

`casting_c3_fixed` 的三 cell continuous evaluator、deterministic driver、per-cell 因果证据、部分完成和有限恢复已在 FakeBackend 完成，并映射为 Casting-S-C2。

## 当前限制

- R5 只在 FakeBackend 验证，真实 MineRL 水/熔岩浇筑尚未验证；
- 真实 backend 仍未完整接通桶动作、公开 selected item、目标方块 truth 和流体 truth；
- `casting_c3_fixed` 不是完整门框，C2 success 不等于进入 Nether；
- 完整门框、点火、Nether entry、Ruined、Adaptive 和 Multi-Agent 均未实现；
- 当前没有正式真实 Benchmark 数据；
- 禁止真实 MineRL、Gradle 和模型调用，除非用户针对每次操作单独授权；
- `vendor/minerl`、固定依赖和任务兼容 ID 均不得在 R6 前顺手修改。

## 测试要求

Task catalog 解析/路径/分类正反例、R5 evaluator 56 个专项测试、R5 driver 56 个专项测试，以及 capability、benchmark file、CLI、R3/R4 回归和全量离线测试必须保持通过。任何结构整理不得削弱严格解析、预算、因果、兼容性或信息隔离合同。

本轮验证：`python -m obsidianlink --check` 与 `python scripts/check_environment.py` 均通过；全量离线测试 414 个全部通过；`git diff --check` 干净；`vendor/minerl` 无修改。

## 下一任务

`R6-COMPLETE-PORTAL-FRAME` 将从 Casting-S-C2 推进到固定场景的有效门框，并继续覆盖点火与 Nether entry 的阶段化 evaluator 合同。开始前必须先冻结 C3/C4/C5 边界、目标 frame geometry、合法原版更新证据和成功定义；不得直接接模型或真实 MineRL。
