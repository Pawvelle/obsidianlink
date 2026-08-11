# C1 Live MineRL Smoke Validation Contract

阶段：`R6-C1-LIVE-MINERL-SMOKE-RUNNER-WIRING`（已完成，offline）

本文档冻结 **C1** 真实 MineRL 烟雾验证合同，并记录 offline runner wiring。合同冻结与 runner wiring **不等于**真实能力已验证，也不得把 `live_run_allowed` 改为 `true`。

## 冻结身份

| 字段 | 冻结值 |
|---|---|
| family | `casting` |
| mode | `single` |
| level | `C1` |
| layout | `fixed` |
| compatibility task | `casting_c1_fixed` |
| designated agent | `agent_1` |
| target cell | `[2, 4, 3]` |

任务页：[casting_c1_fixed](../tasks/casting/casting_c1_fixed.md)。

## 最小目标

- 只验证一个目标 cell；
- 必须使用原版水、熔岩和 Minecraft block update 生成黑曜石；
- 不允许预置或直接放置 `obsidian`；
- evaluator 必须独立验证 target block、water/lava observation、transition step 和合法动作因果；
- driver 完成或文本声称成功不能构成成功；
- truth 缺失、坐标不一致、身份不一致或因果不足时必须 fail closed。

## 授权边界

- 每次真实 MineRL/Minecraft 运行都需用户单独批准；
- 每次 Gradle 构建都需用户单独批准；
- 修复后若要再次真实运行，必须重新批准；
- 不得跳过 C1 smoke 直接进入 C5 live 或 R7。

## 证据要求

结果写入 `runs/`（**仅**用户授权的真实运行），至少包含：

```text
task_instance.json
experiment_config.json
capability_manifest.json
code_version.json
initial.png
final.png
events.jsonl
evaluator_events.jsonl
summary.json
manual_review.md
```

observation、action、evaluation 和 log 必须带 `episode_id`、`step_id`，适用时带 `agent_id`。Agent-visible 与 evaluator-only 数据必须分开；evaluator truth 不得进入 Observation、driver event、prompt 或 memory。

## Offline runner wiring（已完成）

入口：

- 核心：`obsidianlink.runners.casting_c1_live_smoke.run_casting_c1_live_smoke`
- CLI：`scripts/run_c1_live_smoke.py --mode offline_stub --output-dir <绝对路径>`

执行模式闭集：仅 `offline_stub`。必须使用 `build_offline_stub_env_factory()` 返回的受控 `OfflineC1StubEnvFactory`；任意 callable 和外部 backend 注入均被拒绝。拒绝 live 请求；不得写入正式 `runs/`。

编排顺序：preflight → backend open → C1 deterministic driver → mark terminated → `get_casting_evaluation_state((2,4,3))` → 独立 `CastingEvaluator` → evidence bundle → close。

`driver_status`、`evaluator_success` 与 `evidence_complete` 分字段表示；driver completed 不能冒充 success。

Preflight 要求完整 TaskInstance 与冻结 `casting_c1_fixed` 精确一致，包括 step/time/model-call budgets 和精确库存；额外 `obsidian`、修改预算、已存在的输出目录均在环境创建前拒绝。Evidence 只发布到尚不存在的目录，并通过同父目录 staging + 原子 rename 最终化，绝不覆盖已有文件。

## 当前状态

- FakeBackend C1 evaluator/driver：已离线验证；
- MineRL backend typed truth wiring：已在 stub raw observations 上离线验证；
- C1 smoke runner wiring：已在 offline stub 上完成；
- 真实 MineRL/Minecraft 水、熔岩、黑曜石变化：尚未验证；
- `casting_s_c5_fixed`：仍为 `contract_only` / `live_run_allowed=false`。

下一步必须是用户单独授权的一次 C1 真实 MineRL smoke run。若需要 Gradle，仍须另行单独批准。
