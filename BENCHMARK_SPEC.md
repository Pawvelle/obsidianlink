# ObsidianLink Benchmark 总规范

本文档是 ObsidianLink Benchmark 的权威总规范。任务分类和稳定命名见 [TASK_TAXONOMY.md](docs/benchmark/TASK_TAXONOMY.md)；当前能力切片合同见 [`casting_c1_fixed`](docs/tasks/casting/casting_c1_fixed.md) 和 [`casting_c3_fixed`](docs/tasks/casting/casting_c3_fixed.md)。

正式任务身份、canonical taxonomy、历史兼容 ID 和 calibration 可见性以 [`benchmark/catalog/tasks.json`](benchmark/catalog/tasks.json) 为统一索引；解析与迁移规则见 [TASK_REGISTRY.md](docs/architecture/TASK_REGISTRY.md)。Catalog 不包含 evaluator truth，也不替代 task instance 合同。

## 1. Benchmark 总目标

ObsidianLink 是一个可复现的 Minecraft 单智能体与多智能体 Benchmark，用于评测 Agent 在“进入下界”任务中的环境感知、路线可行性判断、长程规划、具身动作执行、错误恢复、路线切换、分工、通信与协作，并要求自动评估和可审核运行证据。

端到端目标是：至少一名任务合同指定的 Agent，通过当前 episode 内完成建造、修复或激活的下界传送门进入 Nether。

## 2. 任务族

### Casting

Agent 使用原版水、熔岩、支撑结构和方块更新机制构造传送门。中间里程碑为：单块黑曜石、连续目标黑曜石、有效门框、点火、进入 Nether。

### Ruined Portal

Agent 探索废弃传送门，理解缺失结构，收集或利用资源完成修复，再点火并进入 Nether。识别、定位、到达、判断缺口和完成修复都是中间里程碑；仅找到结构不是完整成功。

### Adaptive

场景同时或潜在地提供 Casting 与 Ruined Portal 路线。Agent 需要根据距离、资源、结构完整程度、风险和执行成本选择路线，并在失败或条件变化时切换。早期场景优先采用“只有一条路线实际可行”的可审计设计；后续才引入两条路线均可行但成本不同的比较。

## 3. Agent 模式

- Single-Agent：一个 Agent 独立观察、规划、执行和恢复。
- Multi-Agent：多个具有独立身份和私有 observation/memory 的 Agent，在同一任务族内通过受控消息或共享状态合作。

模式与任务族正交，形成 Casting-S/M、Ruined-S/M 和 Adaptive-S/M 六个矩阵单元。Multi-Agent 不是独立的第四任务族。

## 4. 统一成功定义

端到端 Benchmark success 必须同时满足：

1. 任务实例、指定 Agent 集合、初始世界和预算在 reset 时冻结；
2. 门框在当前 episode 中被建造、修复或从非完整状态激活；
3. 世界变化只来自允许的 Agent 动作和原版 Minecraft 机制；
4. 至少一名指定 Agent 在预算内实际进入 Nether；
5. evaluator-only 真值和证据链独立验证以上事实；
6. episode 正常终止，证据身份、step 顺序和版本信息完整。

仅发现传送门、仅完成门框、仅点火、使用与本 episode 无关的预先完整传送门、文本声称成功、driver 正常退出，均不构成端到端成功。Evaluator 或 Minecraft 命令不得直接修改世界。

C1/R1 等能力切片可以使用局部 evaluator outcome `success` 表示该切片完成，但汇总时必须标注 `task_level`，不得把它冒充进入 Nether 的端到端成功。

## 5. 中间里程碑与 Completion Rate

每个实例必须冻结有序、可判定的里程碑：

- Casting：流体与材料就绪、黑曜石 cell、连续区段、有效门框、点火、Nether entry；
- Ruined：识别、定位、到达、缺口判断、材料就绪、修复、点火、Nether entry；
- Adaptive：候选信息获取、初始路线选择、路线执行、必要时切换、最终路线完成、Nether entry；
- Multi-Agent：在所属 family 的里程碑之外记录每个 Agent 的可归因贡献。

`completion_rate` 是已验证里程碑权重或数量占比；权重必须在任务合同中预先冻结。它是诊断指标，不替代 `success`，也不能掩盖关键末端里程碑缺失。

## 6. 信息边界

### Agent-visible

只包含正常游戏画面、公开库存、手持物品、允许的状态字段、该 Agent 接收的消息，以及任务明确公开的信息。Planner 只使用这一侧的数据。

### Evaluator-only

包括目标 cell 和方块 truth、流体 truth、隐藏 Portal 结构、可行路线集合、参考路线与参考成本、评分结果、其他 Agent 的私有 observation，以及任务未公开的场景参数。

两侧必须使用独立类型、存储和日志通道。Evaluator-only 数据不能进入 prompt、memory、消息、共享任务板或策略输入。Multi-Agent 中，一个 Agent 的私有 observation/memory 也不能未经协议直接泄漏给另一 Agent。

## 7. 动作安全与世界修改

- 模型输出必须经过严格结构解析、封闭动作白名单、类型检查和数值限制；
- 不执行模型生成的代码、shell、Minecraft 命令或无限输入；
- 所有环境 step、等待、重试、恢复、消息和模型调用都有硬上限；
- Planner I/O 不得阻塞环境 step loop，过期决策必须丢弃；
- evaluator、runner 和 reviewer 不得为了达成结果而修改世界；
- 只允许任务合同声明的原版机制和 Agent 动作产生计分世界变化。

## 8. 自动 evaluator 要求

Evaluator 必须独立于 Agent 文本和 driver 状态，使用 evaluator-only truth 与结构化事件：

- 输入、版本、outcome 集合、判定优先级和预算应可序列化并冻结；
- truth 缺失、身份不一致、step 乱序或因果证据不足时 fail closed；
- 每个判定关联 `episode_id`、`step_id`，适用时关联 `agent_id`；
- 成功必须能追溯到合法 Agent 动作后的有限因果窗口；
- 自动结果与人工复核不一致时不得计为已确认成功；
- evaluator 版本随结果保存，支持确定性重放和审计。

## 9. 通用指标

Success Rate 是主要指标。效率和可靠性指标作为辅助报告，不设计未经验证的单一综合分数。

| Metric | 定义 |
|---|---|
| Success Rate | 满足该任务层级冻结成功合同的 episode 比例；端到端榜单必须要求进入 Nether |
| Completion Rate | 已验证里程碑完成比例 |
| Environment Steps | episode 消耗的环境 step |
| Game Time | 冻结口径下的游戏内时长 |
| Model Calls | 有效和被拒绝/过期的模型调用计数，分别保存 |
| Invalid Action Rate | 被 parser、安全层或 backend 拒绝的动作占提交动作比例 |
| Recovery Rate | 触发恢复的 episode 比例，并同时报告恢复次数 |
| Evidence Completeness | 必需证据字段和文件完整的比例；缺关键真值时 fail closed |

所有比率必须同时报告分子、分母、任务层级、split 和置信区间或重复 seed 范围，避免跨不同难度直接混合。

## 10. Adaptive 专属指标

| Metric | 定义 |
|---|---|
| Route Selection Accuracy | 初始所选路线属于 evaluator 冻结可行/目标路线集合的比例 |
| Decision Steps | reset 到可审计初始路线承诺之间的 environment steps |
| Route Switch Count | 已验证路线切换次数 |
| Successful Route Switch Rate | 切换后最终成功的切换 episode 比例 |
| Route Abandonment Cost | 被放弃路线在切换前消耗的 step、时间和资源 |
| Final Route | 最终执行完成的 Casting 或 Ruined 路线 |
| Normalized Execution Cost | 实际执行成本相对冻结参考成本的比值或差值，口径由实例预先声明 |

还必须记录路线切换原因。可行路线和参考成本只供 evaluator 使用；Agent 的选择从动作/计划事件推导，不能由 evaluator truth 回填。

## 11. Multi-Agent 专属指标

| Metric | 定义 |
|---|---|
| Team Success Rate | 团队满足任务层级成功合同的 episode 比例 |
| Makespan | reset 到团队终局所经历的 environment steps 或 game time |
| Communication Message Count | 协议接受的 Agent 间消息数 |
| Communication Token Count | 按冻结 tokenizer/计数规则计算的消息 token 数 |
| Idle Step Ratio | 各 Agent 可行动但没有有效任务推进的 step 占比 |
| Duplicate Work Rate | 被 evaluator 归类为重复且无新增里程碑贡献的工作占比 |
| Coordination Failure Count | 冲突动作、错误交接、状态不一致等可判定失败次数 |
| Per-Agent Milestone Contribution | 每个 Agent 对已验证里程碑的可归因贡献 |

贡献均衡度只作分析，不规定越平均越好；有效分工可能天然不均衡。团队 success 仍要求至少一名任务指定 Agent 进入 Nether（对端到端任务）。

## 12. 数据划分

正式 Benchmark 使用 `train` / `dev` / `test`：

- train 可公开生成规则、场景和必要标注，用于开发 baseline；
- dev 用于调试和选择公开配置，但不得反复据其隐藏真值手工定制；
- test 的 seed、隐藏布局、可行路线、参考成本和 evaluator truth 与 Agent 隔离。

同源模板、近重复世界和 seed 不得跨 split 泄漏。Split 分配、生成器版本和 benchmark version 必须保存；最终报告按 family、mode、level、layout 和难度参数分层。

## 13. 运行证据协议

正式 episode 至少保存：

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

Observation、action、message、evaluation 和 log 都包含 `episode_id`、`step_id`，适用时包含 `agent_id`。`summary.json` 使用 [DATASET_CARD.md](DATASET_CARD.md) 规定的统一元数据。Agent-visible 与 evaluator-only 证据分开保存；不得保存密钥、模型权重、隐藏推理或与 episode 无关的个人数据。

## 14. 当前实现范围

当前 active benchmark implementation 包含：

- Casting-S-C1 / `casting_c1_fixed`：FakeBackend 单块能力清单、独立 evaluator 和 deterministic driver；
- Casting-S-C2 / `casting_c3_fixed`：三个有序 cell 的 continuous evaluator、deterministic driver、部分完成、有限恢复和 per-cell 因果证据。

`casting_c3_fixed` 是旧的数量型兼容 ID，其中 `c3` 表示三个 cell；它不属于 taxonomy C3（完整门框）。两个任务目前都只完成 FakeBackend 离线验证。

R6 阶段已在 [catalog](benchmark/catalog/tasks.json) 中新增 3 个 Benchmark 任务条目，分别对应 B0 taxonomy 的 **Casting-S-C3 / Casting-S-C4 / Casting-S-C5 / fixed**。三个任务使用水、熔岩和原版 block update 继续 Casting 主线，并把 `public_task_spec`（门框方案、精确点火目标、指定 Agent/源/目标维度）与 `evaluator_contract`（baseline、因果窗口、frame identity、Nether entry 归因）分开冻结（`implementation_status="contract_only"`、`benchmark_visible=true`、`live_run_allowed=false`）。R6-C3 与 R6-C4 已分别完成 frame/ignition evaluator 和 deterministic driver 的 FakeBackend 离线证明；R6-C5 已完成 `FrozenNetherEntryEvaluator`、typed transition evidence、指定 Agent/维度/transition step/切换前位置/episode portal/frame identity 归因、独立 FakeBackend truth 槽和 347-step C5 deterministic driver 的离线证明。正式 experiment runner 接线、真实 MineRL、Gradle、模型 API 仍未实现。详见 [C3](docs/tasks/casting/casting_s_c3_fixed.md) / [C4](docs/tasks/casting/casting_s_c4_fixed.md) / [C5](docs/tasks/casting/casting_s_c5_fixed.md) 任务页。MineRL backend typed truth wiring 已在 stub raw observations 上离线完成；真实水/熔岩/黑曜石变化与 portal transition 尚未验证。`R6-C1-LIVE-MINERL-SMOKE-VALIDATION-CONTRACT-FREEZE` 与 `R6-C1-LIVE-MINERL-SMOKE-RUNNER-WIRING` 已完成（offline stub）；下一步必须是用户单独授权的一次 C1 真实 MineRL smoke run。

`active_compatibility_id` 保持 `casting_c3_fixed`（C2），即 C3 / C4 / C5 仍不冒充正式 active/live implementation；任何引用都必须同时说明 `implementation_status="contract_only"`。R6-C3/C4/C5 的离线证明仅代表 FakeBackend 行为，不代表真实 MineRL 门框建造、点火、进入 Nether 或正式 Benchmark episode 已验证。

当前不得声称：

- 真实 MineRL 浇筑、门框建造、点火或进入 Nether 已验证；
- 真实 MineRL 维度切换证据已采集；
- 正式 benchmark episode 数据集已发布；
- Casting-S-C5 已在真实 MineRL 端到端运行；
- Ruined Portal 环境或修复任务已实现；
- Adaptive planner/evaluator 已实现；
- Multi-Agent observation、通信或协作已实现。

这些内容属于 [ROADMAP.md](ROADMAP.md) 的后续阶段，而不是 B0 / R6 合同冻结与 C3/C4/C5 离线证明的交付。
