# `casting_c3_fixed` 连续浇筑任务

`casting_c3_fixed` 是 R5 的历史兼容任务 ID。名字中的 `c3` 表示“三个 target cell”，不是 B0 taxonomy 的能力层级 C3；其正式分类是 Casting-S-C2 / fixed，文档级兼容名称为 `casting_s_c2_fixed`。任务文件、workflow、driver 和 evaluator API 均保留原 ID，避免破坏重放与回归。

## 任务文件

- 实例：[casting_c3_fixed.json](../../../benchmark/instances/active/casting_c3_fixed.json)
- 实验合同：[casting_c3_contract.json](../../../configs/experiments/active/casting_c3_contract.json)
- 基础任务：[`casting_c1_fixed`](casting_c1_fixed.md)

## 固定合同

- family：`casting`
- mode：`single`
- level：`C2`
- layout：`fixed`
- Agent：`agent_1`
- 维度：`minecraft:overworld`
- world seed：`0`
- 三个有序 target cell：`[2,4,3]`、`[3,4,3]`、`[4,4,3]`
- 初始资源：3 个水桶、3 个熔岩桶、6 个圆石
- 预算：240 environment steps、180 秒 game time、最多 1 次 model call
- 当前 planner：不调用模型的 deterministic `casting_c3` driver
- 当前 backend：仅 FakeBackend 离线验证

Driver 还执行更严格的计划约束：默认 72 步、最多 96 个 wait step、每个动作最多 2 次恢复、总恢复预算最多 8（冻结实例默认每个动作 1 次、总计 3 次）。调用方只能收紧任务预算，不能放宽。

## 成功与部分完成

局部 `success` 要求三个冻结 cell 按固定顺序全部由非黑曜石变为黑曜石，并且每个 cell 都有独占的相关 Agent 动作、水/熔岩 truth 和有限窗口内的 obsidian transition。Episode 还必须在 step/time 预算内正常终止。

完成非空严格有序前缀时返回 `partial_completion`；中间出现空洞、错误方块或跨 cell 重复声明 action step 不能伪装为成功。Driver 的 `completed` 只表示有限计划执行完毕，最终 verdict 必须来自 evaluator。

R5 的局部 `success` 只证明 C2 连续浇筑切片完成，不代表完成门框、点火或进入 Nether。

## 稳定 outcome

- `success`
- `in_progress`
- `partial_completion`
- `wrong_block`
- `truth_missing`
- `step_budget_exceeded`
- `time_budget_exceeded`
- `invalid_initial_state`
- `causality_missing`
- `abnormal_termination`

## 信息隔离

每个 cell 的 target、initial/current block、water/lava truth、transition 和 relevant action steps 都是 evaluator-only。Driver 只能读取 Agent-visible observation 和公开的类型受控 `RecoverableBackendError`，不得调用 continuous evaluator truth 接口，也不得读取 outcome、completed cells 或 failure type。

Observation、driver event 和 evaluator state 均校验 `episode_id`、`step_id` 和 `agent_id`。测试锁定 evaluator 不依赖 driver/Agent/VLM 模块，driver 不依赖 evaluator 类型，并阻止 truth 出现在 observation frame、inventory、messages 或 workflow stage。

## 当前实现状态

R5 已在 FakeBackend 上完成 deterministic driver、continuous evaluator、有限恢复、重放稳定性、部分完成、预算失败、因果失败和信息隔离验证。真实 MineRL 尚未接通所需桶动作、目标方块和流体 truth；未运行 Minecraft、MineRL、Gradle 或模型 API，也没有正式真实 episode 数据。
