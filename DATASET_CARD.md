# 运行数据说明

ObsidianLink 的数据是每次 benchmark episode 产生的证据，不是训练数据集。

## 每次运行保存

- 任务实例和实验配置；
- 代码与依赖版本；
- Agent 可见的初始、关键和最终画面；
- observation、action、message 和 evaluation 事件；
- 自动评分结果；
- `manual_review.md`。

所有记录使用 `episode_id` 和 `step_id`，适用时使用 `agent_id`。

## 信息隔离

Agent-visible 数据与 evaluator-only 真值分开保存。目标方块、流体真值和 Portal 结构不能进入 Agent prompt 或 memory。

## 不保存

- API key 或其他密钥；
- 本地模型权重；
- 隐藏推理；
- 与当前 episode 无关的个人数据。

## 当前状态

`casting_c1_fixed` 尚未进行正式真实运行。当前阶段只产生离线测试结果，不产生 benchmark episode 数据。
