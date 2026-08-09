# `casting_s_c3_fixed` 完整门框浇筑任务（C3 合同冻结）

`casting_s_c3_fixed` 是 **Casting-S-C3 / fixed** 的正式 Benchmark 合同，实例 ID 为 `casting_s_c3_fixed_seed_0`。R6-C3 已在 FakeBackend 上完成 frame evaluator、task-origin / truth-grid 数值锚定、严格 public context、capability gate 和 deterministic driver；C4 ignition evaluator/driver、C5 Nether-entry evaluator/driver 与真实 MineRL 接入仍未实现。

C3 要求 Agent 使用水、熔岩、支撑方块和原版方块更新机制，把公开固定方案中的 14 个 cell 全部浇筑为黑曜石。C3 不要求点火或进入 Nether，也不允许直接提供或放置黑曜石。

## 任务文件

- 实例：[`benchmark/instances/casting/single/casting_s_c3_fixed.json`](../../../benchmark/instances/casting/single/casting_s_c3_fixed.json)
- 离线合同：[`configs/experiments/active/casting_s_c3_contract.json`](../../../configs/experiments/active/casting_s_c3_contract.json)
- Catalog：[`benchmark/catalog/tasks.json`](../../../benchmark/catalog/tasks.json)
- 上游任务：[`casting_c3_fixed`（Casting-S-C2 / fixed）](casting_c3_fixed.md)

## 固定合同

- family / mode / level / layout：`casting` / `single` / `C3` / `fixed`
- Agent：`agent_1`
- 维度：`minecraft:overworld`
- world seed：`0`
- 公开任务原点：由场景中的可见 marker 标识；下述坐标均相对该原点
- 初始资源：`water_bucket=14`、`lava_bucket=14`、`cobblestone=28`
- 不提供 `obsidian` 或 `flint_and_steel`
- 预算：640 environment steps、600 秒 game time、最多 1 次 model call
- 状态：`contract_only`

资源数量是固定开发实例的资源上限，不代表未来随机或挑战 split 的标准配置。现有 C3 driver 沿用 R5 的严格动作协议、有限等待和恢复预算。

## Agent-visible 公开门框方案

`scenario_parameters.public_task_spec` 是公开任务规则，必须与 instruction 一致：

| 字段 | 值 |
|---|---|
| `coordinate_space` | `task_origin_relative` |
| `task_origin_marker` | `visible` |
| `orientation` | `plane_z` |
| `min_corner` | `[0,0,1]` |
| `width` / `height` | `4` / `5` |
| `require_full_ring` | `true` |
| `minecraft_minimum_required_block_count` | `10` |
| `benchmark_required_full_ring_block_count` | `14` |
| `required_corner_count` | `4` |

14 个公开 full-ring cell 为：

```text
bottom: (0,0,1) (1,0,1) (2,0,1) (3,0,1)
top:    (0,4,1) (1,4,1) (2,4,1) (3,4,1)
left:   (0,1,1) (0,2,1) (0,3,1)
right:  (3,1,1) (3,2,1) (3,3,1)
```

Minecraft 的最小合法 4×5 门框只要求 10 个非角 cell，四个角可以缺失；现有 `FrameCandidate.required_count` 也使用 10。这个固定 Benchmark 实例有意采用更严格的 **14-block full-ring constraint**。因此：

- `minecraft_minimum_required_block_count=10` 描述原版合法性；
- `benchmark_required_full_ring_block_count=14` 描述本实例完成条件；
- 后续 evaluator 必须先验证原版几何合法，再验证本实例额外要求的四个角；不得把 `FrameCandidate.required_count` 错写成 14。

内框尺寸为 2×3；允许 `air`、`nether_portal`、`fire`，遇到 `dirt`、`bedrock`、`grass`、`grass_block`、`obsidian`、`other` 或 `missing` 时 fail closed。

## C3 evaluator 合同

C3 success 必须同时满足：

1. 14 个 full-ring cell 的 baseline 均明确不是 `obsidian`；仅声明“reset 时没有完整门框”不够；
2. 每个 cell 在本 episode 内通过水、熔岩与原版 block update 转为 `obsidian`；
3. 相关动作来自 `agent_1` 的合法 `use_item`，使用 `water_bucket` / `lava_bucket`，并位于有限因果窗口内；
4. 不允许 `place_block(obsidian)`、Minecraft 命令、driver/evaluator 写世界或外部预存门框取得归因；
5. 内框无阻挡，episode、step 与 agent 身份完整且顺序一致；
6. evaluator truth 缺失时 fail closed，driver 正常退出或模型文本声明不构成成功。

`scenario_parameters.evaluator_contract.frame_attribution` 已机器冻结 baseline policy、合法机制、动作与物品集合、4-step 因果窗口和 fail-closed 规则。

## R6-C3-FRAME-EVALUATOR 子阶段交付

R6-C3-FRAME-EVALUATOR 子阶段在 FakeBackend 上完成了 C3 frame evaluator 和 evaluator-only truth 注入路径；随后的 R6-C3-DETERMINISTIC-DRIVER 子阶段完成了严格 public context、capability gate、固定 14-cell plan、有限预算/恢复与独立测试编排。C4/C5 和真实 MineRL 仍未实现。

### Evaluator 表面

`obsidianlink/evaluation/casting_frame_evaluator.py` 提供：

- `FrozenFrameActionEvidence` / `FrozenFrameCellTruth` / `FrozenFrameInteriorCellTruth` / `FrozenFrameEvaluationState` / `FrozenFrameEvaluationResult` 均为 frozen / 类型严格 / 可序列化的 evaluator-only dataclass；每条相关动作都绑定 `episode_id` / `step_id` / `agent_id` / `target_cell`，且只接受 `use_item(water_bucket | lava_bucket)`；
- 14 个 target cell 顺序与本合同的 `public_task_spec.frame_plan.fixed_offsets` 严格一致；6 个 interior cell 同样在合同中冻结；
- 闭集 outcome：`success` / `in_progress` / `partial_completion` / `wrong_block` / `truth_missing` / `step_budget_exceeded` / `time_budget_exceeded` / `invalid_initial_state` / `causality_missing` / `abnormal_termination` / `interior_blocked`；
- 闭集 outcome 优先级：step / time 预算 → `invalid_initial_state` → `abnormal_termination` → `truth_missing` → `interior_blocked` → `in_progress` → `causality_missing` → `partial_completion` → `wrong_block` → `success`；
- C3 success 要求 14 个 target cell 全部 obsidian、6 个 interior cell 全部在 `INTERIOR_ALLOWED`（`air` / `nether_portal` / `fire`）内、且每 cell 在 `causality_window_steps`（默认 4，<= 32）内通过合法 `use_item(water_bucket | lava_bucket)` 完成 transition；
- `partial_completion` 用于任意 1–13 个 cell 已成功、其余 cell 仍为 `air` / `water` / `lava` 的状态，与公开列表顺序无关；Minecraft 10-cell 最小合法门框 + 4 角仍为空会落入 `partial_completion`，绝不冒充 `success`；`cobblestone` / `stone` 等阻挡目标仍判为 `wrong_block`；
- `evaluate(state)` 是纯函数：重复调用得到完全相同的 `FrozenFrameEvaluationResult` 与 `as_dict()` JSON 快照；不读取 Agent / driver / Agent-visible observation。

### 任务原点 / truth-grid 坐标锚定

- `FrozenFrameOriginAnchor` 是不可变、纯函数、类型严格的转换器，把 `task_origin_relative` 偏移转成 `truth_grid` 偏移；
- 默认 `default_c3_anchor()` 把 task-origin 标记对齐到 grid 原点 `(0, 0, 0)`，grid 范围沿用 `obsidianlink.env.portal_spec.PORTAL_GRID_MIN/MAX` (`(-3,-1,0)`–`(3,5,6)`)；
- 14 个公开 cell 在数值上落在 `x=0..3` / `y=0..4` / `z=1`，完全被现有 truth grid 数值范围覆盖；不为了本轮合同扩展 grid；
- 越界、缺失 origin、bool 混入、grid 边界反向（`grid_min > grid_max`）全部 fail closed；
- 不修改 MineRL `PortalA0EnvSpec`。

### FakeBackend 注入路径

- `FakeEnvironmentBackend` 新增独立 `_frame_evaluation_state` 槽位，与 `_casting_evaluation_state`（C1）和 `_continuous_casting_evaluation_state`（C2）严格隔离；
- `set_frame_evaluation_state` 校验类型 / `episode_id` 与当前 task 一致 / `step_id` 与当前 backend step 一致 / `agent_id` 在 `task.agent_ids` 内，否则 fail closed；
- `get_frame_evaluation_state` / `clear_frame_evaluation_state` 显式可调用；
- `reset` / `step` / `close` 一律清空该槽位，杜绝跨 step 的 truth 泄漏；
- 普通 `Observation`（`step_id=0` 与 `step_id=1`）不携带任何 frame / cell / outcome / 归因 / 截断 truth，公开 schema 字段集严格不变。

### 信息隔离

- 整个 evaluator 源文件 AST 检查锁定：不 import `obsidianlink.agents` / `obsidianlink.workflows` / `obsidianlink.drivers`；不读取 `scenario_parameters` / `evaluator_contract` / `instruction`；
- `evaluate()` 唯一参数是 `state: FrozenFrameEvaluationState`；
- C1 / C2 / portal 旧测试全部回归通过；R6-C3-FRAME-EVALUATOR 103 个专项测试全绿；全量 539 个离线测试通过；
- R6-C3 driver 已通过 AST、Observation guard、backend spy 和独立 test orchestrator 锁定端到端信息隔离；driver 不读取 frame truth，也不调用 truth set/get/clear 表面。

## 信息隔离

公开规则包括任务目标、固定布局、14 个目标 cell、资源、指定 Agent 和预算。Agent 必须知道这些内容才能执行任务。

以下运行时证据属于 evaluator-only，不得进入 prompt、memory、消息或 observation：

- baseline/current grid 的真实方块状态；
- 每个 cell 的水、熔岩和 obsidian transition 证据；
- 相关动作 step、归因候选和外部世界修改；
- `latched_frame_identity`、outcome、success、failure type 和里程碑时间戳。

R6-C3-DETERMINISTIC-DRIVER 子阶段已补齐端到端 public-context 隔离：唯一 context builder 只抽取公开 frame plan、身份、库存和预算并保持原始类型严格校验；driver 不接触原始 `scenario_parameters`、`evaluator_contract` 或 evaluator runtime truth。FakeBackend Observation 仍不携带 frame / cell / outcome / 归因 truth。

## 当前实现状态

R6-C3 evaluator 与 deterministic driver 已完成 FakeBackend 离线证明：14-cell、336-step bounded plan，缺少 backend capability 时 reset 前 fail closed，测试 orchestrator 独立注入 truth 并由 evaluator 给出最终 verdict。C4 ignition evaluator/driver、C5 Nether entry evaluator/driver、真实 MineRL、Gradle和模型 API 仍未实现。

现有 portal truth grid 范围 `(-3,-1,0)–(3,5,6)` 已覆盖本合同的 x=`0..3`、y=`0..4`、z=`1`，因此无需为了该固定方案扩展 grid；真实 backend 的坐标锚定仍需在 driver / backend 接入阶段验证。

下一子任务：`R6-C4-IGNITION-EVALUATOR`；必须先完成点火 evaluator 的离线归因合同，再启动 C4 deterministic driver。
