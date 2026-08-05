# `casting_s_c3_fixed` 完整门框浇筑任务（C3 合同冻结）

`casting_s_c3_fixed` 是 **Casting-S-C3 / fixed** 的正式 Benchmark 合同，实例 ID 为 `casting_s_c3_fixed_seed_0`。本阶段只冻结任务、信息边界和 evaluator 规则；R6 driver、evaluator 与真实 MineRL 接入均未实现。

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

资源数量是固定开发实例的资源上限，不代表未来随机或挑战 split 的标准配置。后续 driver 必须继续沿用 R5 的严格动作协议、有限等待和恢复预算。

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

## 信息隔离

公开规则包括任务目标、固定布局、14 个目标 cell、资源、指定 Agent 和预算。Agent 必须知道这些内容才能执行任务。

以下运行时证据属于 evaluator-only，不得进入 prompt、memory、消息或 observation：

- baseline/current grid 的真实方块状态；
- 每个 cell 的水、熔岩和 obsidian transition 证据；
- 相关动作 step、归因候选和外部世界修改；
- `latched_frame_identity`、outcome、success、failure type 和里程碑时间戳。

当前 R6 尚无 runtime，因此现阶段只能冻结公开/隐藏命名空间并验证现有 Agent 代码不读取 `scenario_parameters`。后续实现 driver 时必须新增显式的 public context 构造器，并测试其输出不包含 `evaluator_contract` 或运行时 truth。

## 当前实现状态

已冻结 C3 合同、catalog、配置、文档和离线一致性测试。未实现 C3 evaluator、deterministic driver、真实 MineRL 或模型接入。

现有 portal truth grid 范围 `(-3,-1,0)–(3,5,6)` 已覆盖本合同的 x=`0..3`、y=`0..4`、z=`1`，因此无需为了该固定方案扩展 grid；真实 backend 的坐标锚定仍需在接入阶段验证。

下一子任务：`R6-C3-FRAME-EVALUATOR`。
