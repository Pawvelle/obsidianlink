# v2 Task Registry 与 Legacy Quarantine

权威索引是 [`benchmark/catalog/tasks.json`](../../benchmark/catalog/tasks.json)，严格 parser 位于 [`obsidianlink/core/task_catalog.py`](../../obsidianlink/core/task_catalog.py)，v2 facade 位于 [`obsidianlink/benchmark/registry.py`](../../obsidianlink/benchmark/registry.py)。

## Catalog roles

Catalog entry 的 `kind` 使用：

- `benchmark`：未来冻结且可发布的 v2 task；当前为 0 条；
- `legacy`：旧 C1/C2/taxonomy C3–C5 task/contract，保留 traceability 与 regression；
- `calibration`：Route A0 等环境/solver calibration。

Legacy 与 calibration 必须 `benchmark_visible=false`、`live_run_allowed=false`。`verification_level=unit_verified` 只表示离线 contract/regression，不表示真实 Minecraft 能力。

Catalog 顶层 `active_phase` 是 `P1-REAL-MINERL-ENVIRONMENT-VALIDATION`，`active_benchmark_task_id` 为 `null`。P1 environment validation 不是 benchmark task，因此不为它创建空 TaskInstance。

## Compatibility strategy

历史 instance/config/import 路径暂不移动：

```text
benchmark/instances/active/casting_c1_fixed.json
benchmark/instances/active/casting_c3_fixed.json
benchmark/instances/casting/single/casting_s_c3_fixed.json
benchmark/instances/casting/single/casting_s_c4_fixed.json
benchmark/instances/casting/single/casting_s_c5_fixed.json
benchmark/instances/route_a_a0_*.json
```

目录名中的 `active` 是历史路径兼容，不是 v2 active scope。Registry 是发布可见性的唯一真源。

## Validation

Parser fail closed：未知字段、重复 ID、危险路径、taxonomy mismatch、缺文件、experiment 指向错误、live policy 不一致都会失败。Catalog 当前还保证：

- 没有 `benchmark_visible=true` 的旧 entry；
- 没有 active benchmark task；
- active phase 只指向 P1 validation；
- verification vocabulary 使用闭集。

未来新增 v2 task 必须先完成对应阶段的 schema/spec/evaluator freeze，再加入 catalog；不能只靠文件名或 README 宣称支持。

历史 registry 规则归档于 [TASK_REGISTRY_V1.md](../legacy/v1/TASK_REGISTRY_V1.md)。
