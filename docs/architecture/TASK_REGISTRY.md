# Task Catalog 与兼容路径

本文档定义 ObsidianLink 如何在不破坏历史任务、实验配置和重放证据的前提下扩展任务结构。权威 catalog 位于 [`benchmark/catalog/tasks.json`](../../benchmark/catalog/tasks.json)，严格解析器位于 [`obsidianlink/core/task_catalog.py`](../../obsidianlink/core/task_catalog.py)。

## 目标

- 为正式 Benchmark 任务提供稳定的 family/mode/level/layout taxonomy；
- 区分公开 Benchmark task 与早期 calibration/regression task；
- 保留历史 task ID、workflow、文件路径和实验引用；
- 为 runner、CLI、结果汇总和未来数据发布提供单一任务索引；
- 在 episode 启动前发现重复 ID、错误分类、失效路径和 live-run policy 冲突。

Catalog 只读取本地 JSON，不启动 MineRL/Minecraft、不执行任务、不修改世界，也不包含 evaluator-only truth。

## Entry 类型

### `benchmark`

正式 Benchmark 能力切片。必须声明：

- canonical name；
- compatibility ID；
- task instance ID 和 workflow；
- `task_family`、`agent_mode`、`task_level`、`layout_type`；
- task instance 与 experiment config 路径；
- implementation status、可见性和 live-run policy。

Canonical name 必须由 taxonomy 唯一推导，例如 `casting_s_c2_fixed`。Catalog 不用 canonical name 覆盖旧 ID；`casting_c3_fixed` 仍是 C2 的兼容 ID。

### `calibration`

早期环境、Portal evaluator、脚本或模型链路的校准/回归任务。它们不是 Casting/Ruined/Adaptive 的正式发布实例，因此：

- `taxonomy` 必须为 `null`；
- `benchmark_visible` 必须为 `false`；
- 当前 `implementation_status` 为 `legacy_regression`；
- 可以继续被既有离线测试或需授权的历史 runbook 引用。

`route_a_a0_development` 和 `route_a_a0_phase3` 当前属于这一类。分类不是删除：它只阻止这些早期任务被误计入正式 Benchmark 指标。

## 兼容策略

现有路径保持不变：

```text
benchmark/instances/active/casting_c1_fixed.json
benchmark/instances/active/casting_c3_fixed.json
benchmark/instances/route_a_a0_development.json
benchmark/instances/route_a_a0_phase3.json
```

R6 起的新任务使用 canonical taxonomy 命名，并按 family/mode 建目录，例如：

```text
benchmark/instances/casting/single/casting_s_c3_fixed.json
benchmark/instances/casting/single/casting_s_c4_fixed.json
benchmark/instances/casting/single/casting_s_c5_fixed.json
```

只有在专门迁移阶段、所有实验配置和重放引用都有兼容映射时，才允许移动历史文件。本 B1 阶段不移动任何实例。

## 严格验证

`TaskCatalog` 和 `TaskCatalogEntry` 是 frozen 类型。解析器 fail closed：

- 拒绝未知或缺失字段；
- 拒绝 family 与 level 不匹配；
- 拒绝绝对路径、`..` 和非 JSON 路径；
- 拒绝重复 canonical name、compatibility ID、task instance ID 或 instance path；
- 正式 Benchmark entry 必须包含 taxonomy 且可见；
- calibration entry 必须无 taxonomy 且不可见；
- active entry 必须是可见的 Benchmark task。

`validate_catalog_references()` 进一步校验：

- instance 和 experiment 文件真实存在；
- catalog 的 task instance ID、workflow 与实例一致；
- Benchmark taxonomy 与 `scenario_parameters` 一致；
- experiment config 指向 catalog 声明的实例；
- live-run policy 与实例一致。

CLI `--check` 和 `scripts/check_environment.py` 都必须通过 catalog 解析当前 active task，不能各自硬编码 taxonomy。

## 新任务流程

新增任务时按以下顺序：

1. 在 `TASK_TAXONOMY.md` 已冻结的 family/mode/level/layout 中选择分类；
2. 使用 canonical name 创建新实例，不复用数量型歧义命名；
3. 冻结 task instance、实验配置和 live-run policy；
4. 把 entry 加入 catalog；
5. 增加正反例测试，确保所有实例都被 catalog 覆盖；
6. 先用 FakeBackend 和 deterministic driver 验证；
7. 只有对应阶段和单独授权到位后，才允许真实 MineRL、Gradle 或模型调用。

Catalog 不替代 task schema、evaluator 或运行证据；它只负责身份、分类、路径和发布可见性。
