# Benchmark 说明

## 目标

评估一个 Minecraft Agent 是否能用原版机制构建并使用下界传送门。当前评估 R5 最小连续浇筑切片 `casting_c3_fixed`；R4 的 `casting_c1_fixed` 作为回归合同保留。

## 当前任务

- 维度：Overworld
- Agent：1 个
- 布局：固定受控场景
- 资源：3 个水桶、3 个熔岩桶和 6 个支撑方块
- 目标：让 3 个冻结、有序的目标 cell 从非黑曜石变成黑曜石
- 禁止：Minecraft 命令和 evaluator 修改世界

任务实例：[`benchmark/instances/active/casting_c3_fixed.json`](benchmark/instances/active/casting_c3_fixed.json)

## 成功条件

以下条件全部满足才算成功：

1. reset 后 3 个目标 cell 都不是黑曜石；
2. Agent 在预算内执行白名单动作；
3. 3 个目标 cell 都通过水与熔岩更新变成黑曜石；
4. 每个变化都发生在该 cell 独占的相关动作后的有限时间窗口；
5. episode 正常结束，自动评估和人工复核一致。

少于或多于 3 个 cell、目标顺序错误、跨 cell 重用动作 step、真值缺失、因果不清、形成圆石、超预算或进程残留都不能算成功。完成非空有序前缀但未完成全部 cell 时只能得到 `partial_completion`。

## 信息边界

Agent 只能看到正常 observation，例如画面、公开库存和手持物品。目标 cell、流体状态、Portal 结构和评分结果属于 evaluator-only 信息，不得进入 prompt、memory 或策略输入。

## 安全边界

- 动作必须经过 schema、类型、白名单和数值限制；
- 每个事件带 `episode_id`、`step_id`，适用时带 `agent_id`；
- 所有循环、等待、重试和模型调用都有上限；
- 模型不得生成并执行代码、shell 或 Minecraft 命令。

## 证据

正式运行至少保存任务配置、代码版本、初始与最终画面、动作事件、evaluator 事件、summary 和 manual review。`accepted=true` 或 driver 结束都不等于任务成功。

## 当前状态

该任务仍处于离线准备阶段。必须按 [ROADMAP.md](ROADMAP.md) 先完成能力清单、evaluator 和确定性 driver，之后才可申请真实 MineRL 运行。
