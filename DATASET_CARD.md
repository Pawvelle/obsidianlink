# ObsidianLink 运行数据说明

ObsidianLink 的数据是每次 benchmark episode 产生的可审计证据，不是当前提供的训练数据集。本卡同时冻结未来正式 Benchmark run 的文档级元数据设计；它不修改当前 schema。

## 每次正式运行保存

- 任务实例、实验配置、capability manifest；
- 代码、依赖和 evaluator 版本；
- Agent 可见的初始、关键和最终画面；
- observation、action、message、evaluation 和 log 事件；
- 自动评分、summary 和 `manual_review.md`。

所有事件使用 `episode_id` 和 `step_id`，适用时使用 `agent_id`。多 Agent 事件必须能区分发送者、接收者和共享状态更新；Adaptive 路线事件必须能审计选择和切换，但不能把 evaluator-only 参考答案写入 Agent-visible 流。

## 未来统一 run 元数据

以下字段是未来正式 Benchmark 数据格式设计。字段类型、枚举和可空规则将在对应 schema 阶段冻结；B0 只冻结语义。

| Field | 含义 |
|---|---|
| `benchmark_version` | Benchmark 规范/发布版本 |
| `task_family` | `casting` / `ruined` / `adaptive` |
| `agent_mode` | `single` / `multi` |
| `task_level` | family 对应的 C1–C5、R1–R5 或 A1–A5 |
| `task_instance_id` | 唯一任务实例 ID；不以分类名称替代 |
| `layout_type` | `fixed` / `randomized` / `hidden` / `challenge` |
| `difficulty_parameters` | 显式难度参数对象，不只保存单一 difficulty 数字 |
| `world_seed` | 可复现世界 seed；test 发布可按协议隐藏 |
| `agent_ids` | 冻结的 Agent ID 列表及指定可计分 Agent |
| `route_options` | 场景候选路线；可能为 evaluator-only |
| `selected_route` | 从 Agent 事件推导的初始选择 |
| `final_route` | 最终执行/完成路线 |
| `route_switches` | 有界、结构化的切换记录与原因 |
| `success` | 对该任务层级冻结成功合同的 evaluator verdict |
| `completion_rate` | 已验证里程碑完成比例及所用权重版本 |
| `environment_steps` | episode 消耗的环境 step |
| `game_time_seconds` | 冻结口径下的游戏时间 |
| `model_calls` | 模型调用计数，区分有效、过期和失败 |
| `communication_messages` | 协议接受的 Agent 间消息数 |
| `communication_tokens` | 按冻结计数规则得到的消息 token 数 |
| `failure_type` | 稳定、可审计的失败分类 |
| `code_version` | 代码版本和 dirty worktree 状态 |
| `evaluator_version` | 产生 verdict 的 evaluator 版本 |

Single-Agent 的通信字段应为 0 或按未来 schema 明确设为不适用；非 Adaptive 任务的路线字段也必须按 schema 明确处理，不能伪造路线值。端到端结果还必须能证明指定 Agent 的 Nether entry。低层 `casting_c1_fixed` 的局部 `success` 必须与 `task_level=C1` 一起解释，不能冒充端到端成功。

## 难度参数

`difficulty_parameters` 至少支持记录：`layout_variation`、`resource_availability`、`exploration_distance`、`observation_uncertainty`、`execution_noise`、`recovery_required`、`route_ambiguity`、`communication_constraint` 和 `agent_count`。D1–D4 等档案标签不能替代具体参数。

## 数据划分

未来发布使用 `train` / `dev` / `test`，记录 split、生成器版本和 benchmark version。同源模板、近重复场景和 seed 不得跨 split 泄漏。Test 的隐藏布局、可行路线、参考成本和 evaluator truth 不进入 Agent 输入；正式汇总按 family、mode、level、layout 和难度参数分层。

## 信息隔离

Agent-visible 数据与 evaluator-only truth 分开存储、授权和回放：

- 目标方块、流体 truth、隐藏 Portal 结构、可行路线、参考路线/成本和评分结果不能进入 prompt、memory 或消息；
- Multi-Agent 中每个 Agent 的私有 observation/memory 不能未经协议出现在其他 Agent 的数据流；
- evaluator 事件可以引用受控 truth，但不能被 Planner 消费；
- 人工复核材料不能反向污染同一 episode 的动作决策。

## Evidence Completeness

正式 run 必须检查所需文件、身份字段、连续 step、版本信息和 evaluator truth 是否完整。缺关键证据时 fail closed，不能只因模型文本、driver 退出或 summary 声称成功而通过。Evidence Completeness 是辅助指标，不替代 Success Rate。

## 不保存

- API key、访问令牌或其他密钥；
- 本地模型权重；
- 隐藏推理或 chain-of-thought；
- 与当前 episode 无关的个人数据；
- 未经任务合同允许的 evaluator truth 副本或跨 Agent 私有数据。

## 当前状态

当前仓库仍未产生真实正式数据集。`casting_c1_fixed` 只有离线测试和受控 FakeBackend 结果，真实 MineRL episode 尚未运行。Ruined、Adaptive、Multi-Agent、统一未来字段和正式 train/dev/test 发布都只是规划，不能声称已有数据支持。
