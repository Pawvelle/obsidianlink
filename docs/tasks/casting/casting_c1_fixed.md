# `casting_c1_fixed` 单块浇筑任务

本文档保存历史兼容任务 `casting_c1_fixed` 的具体合同。其文档级分类名称是 `casting_s_c1_fixed`：Casting family、Single-Agent、C1、fixed layout。兼容 ID、任务文件、workflow 和 schema 均不改名。

## 任务文件

- 实例：[benchmark/instances/active/casting_c1_fixed.json](../../../benchmark/instances/active/casting_c1_fixed.json)
- 离线实验合同：[configs/experiments/active/casting_c1_contract.json](../../../configs/experiments/active/casting_c1_contract.json)
- 操作说明：[FIRST_OBSIDIAN_BLOCK.md](../../runbooks/FIRST_OBSIDIAN_BLOCK.md)

实例和实验 JSON 中的 `contract_only` / `not_implemented` 文本保留了 R1 冻结合同时的历史状态。B0 按要求不修改 schema 或配置；当前 R2–R4 离线实现状态以 [PROJECT_STATUS.md](../../../PROJECT_STATUS.md) 为准。

## 初始条件与固定布局

- 维度：`minecraft:overworld`
- world seed：`0`
- Agent：`agent_1`
- 出生点：`[0, 4, 0]`
- 目标 cell：`[2, 4, 3]`
- 目标初始方块：`air`
- 初始资源：1 个水桶、1 个熔岩桶、8 个圆石
- 布局：固定受控场景，固定资源布局
- 所需机制：原版水、熔岩和方块更新

任务只要求让这一个冻结目标 cell 从非黑曜石变成黑曜石；它是低层能力测试，不要求完成门框、点火或进入 Nether。

## 预算

- 最多 160 个 environment step
- 最多 120 秒 game time
- 合同最多 1 次 model call；当前 deterministic driver 不调用模型
- 所有等待、动作计划和重试还必须服从 driver 的更严格有限上限

调用方只能收紧任务预算，不能放宽。预算数值来自已冻结任务实例，本文档不改变其语义。

## C1 成功条件

局部 evaluator 的 `success` 只有在以下条件全部成立时产生：

1. reset 后目标 cell 不是黑曜石；
2. Agent 在预算内执行经过解析、白名单、类型与数值检查的动作；
3. evaluator truth 明确证明水和熔岩参与；
4. 目标 cell 通过原版方块更新变为黑曜石；
5. transition 位于该 cell 的相关 Agent 动作之后的有限因果窗口内；
6. episode 以允许的正常原因结束，证据身份和 step 顺序一致。

Driver 正常结束、模型文本、画面印象或 evaluator 之外的世界修改都不能证明成功。C1 的 `success` 只表示单块切片完成，不等同于 ObsidianLink 端到端“进入 Nether”的完整成功。

## Evaluator-only truth

Evaluator 可以读取冻结目标坐标、initial/current target block、water/lava truth、方块 transition 证据、相关动作 step、终止原因和预算计数。Planner/driver 只能读取正常画面、公开库存、手持物品和公开 workflow 状态；上述 evaluator-only 字段不得进入 observation frame、prompt、memory 或消息。

Evaluator 不得修改世界。Minecraft 命令不得用于制造结果。

## 稳定 outcome

| Outcome | 含义 |
|---|---|
| `success` | C1 的全部局部成功条件成立 |
| `in_progress` | 尚未到允许判定终局的 step |
| `wrong_block` | 目标未形成黑曜石或形成错误方块 |
| `truth_missing` | 必需 evaluator truth 不完整，fail closed |
| `step_budget_exceeded` | 超出 environment step 预算 |
| `time_budget_exceeded` | 超出 game time 预算 |
| `invalid_initial_state` | reset 时目标已经是黑曜石 |
| `causality_missing` | transition 与相关 Agent 动作没有有限因果关联 |
| `abnormal_termination` | episode 以不允许的原因结束 |

## 当前实现状态

`casting_c1_fixed` 已在 FakeBackend 上完成：

- R2 的不可变 `BackendCapabilities` 和 reset 前 fail-closed 能力门禁；
- R3 的独立 `CastingEvaluationState` / evaluator 与 outcome 分类；
- R4 的确定性、有限动作、有限等待和有限恢复 driver；
- observation 与 evaluator truth 隔离、结构化身份字段和离线测试。

MineRL backend 的 typed target-block / fluid truth 入口已在 stub raw observations 上离线接通；C1 smoke runner wiring 已在 offline stub 上完成。这不等于真实环境已验证。当前没有正式真实 episode 数据。真实 MineRL 中的水、熔岩、黑曜石变化仍未验证。C1 live smoke 合同与 runner 见 [C1_LIVE_MINERL_SMOKE.md](../../runbooks/C1_LIVE_MINERL_SMOKE.md)。任何真实 MineRL/Minecraft 运行及 Gradle 构建都必须另行获得用户批准；不得把 `live_run_allowed` 改为 `true`。
