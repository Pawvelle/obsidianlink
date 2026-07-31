# ObsidianLink-Bench Specification

本文档定义 ObsidianLink 第一版 Benchmark 的任务、信息边界、成功条件、对照组、
指标和数据格式。它是实现和实验的共同契约；模型、Prompt 和后端不得各自解释
成功条件。

## 1. 统一任务

角色或团队必须在当前回合中制造一个有效地狱门，激活它，并让至少一名角色进入
下界。

完整成功定义为：

```text
portal_built_by_episode
AND valid_portal_frame
AND portal_activated
AND any(agent_entered_nether)
```

以下情况不算完整成功：

- 找到已有或自然生成的传送门；
- 只建造门框但未激活；
- 激活传送门但无人进入；
- 进入一个不能证明由当前回合建造或浇筑的门；
- 仅由模型文本声称任务完成。

## 2. 任务路线

### 2.1 Route A - Obsidian Mining

标准工作流：

1. 检查初始资源；
2. 定位可采集黑曜石；
3. 采集足够数量；
4. 获得或确认点火工具；
5. 选择建造地点与朝向；
6. 建造有效门框；
7. 激活地狱门；
8. 至少一名角色进入下界。

标准里程碑：

| 阶段 | 里程碑 |
|---|---|
| 初始化 | `task_reset` |
| 资源检查 | `critical_resources_checked` |
| 定位黑曜石 | `obsidian_located` |
| 首次采集 | `first_obsidian_mined` |
| 数量充足 | `obsidian_sufficient` |
| 点火资源 | `ignition_tool_ready` |
| 建造选址 | `build_site_selected` |
| 门框完成 | `valid_portal_frame` |
| 激活 | `portal_activated` |
| 进入下界 | `agent_entered_nether` |

### 2.2 Route B - Lava Casting

标准工作流：

1. 检查水、岩浆、桶、辅助方块和点火资源；
2. 选择门框位置与朝向；
3. 放置辅助结构；
4. 放置岩浆；
5. 使用水生成目标黑曜石；
6. 验证生成结果；
7. 重复直到门框完成；
8. 清理必要的辅助结构；
9. 激活并进入下界。

标准里程碑：

| 阶段 | 里程碑 |
|---|---|
| 初始化 | `task_reset` |
| 资源确认 | `liquid_resources_ready` |
| 浇筑选址 | `casting_site_selected` |
| 辅助结构 | `support_structure_ready` |
| 首块浇筑 | `first_obsidian_cast` |
| 局部完成 | `portal_segment_completed` |
| 门框完成 | `valid_portal_frame` |
| 点火准备 | `ignition_tool_ready` |
| 激活 | `portal_activated` |
| 进入下界 | `agent_entered_nether` |

## 3. 难度

| 难度 | 场景特征 | 主要能力 |
|---|---|---|
| L1 | 关键资源齐全、地形平坦、目标明确 | 基础视觉动作对齐与建造 |
| L2 | 缺少少量资源、朝向或槽位随机 | 状态检查与短程补全 |
| L3 | 缺少关键资源或目标更隐蔽 | 探索、记忆与路线执行 |
| L4 | 放置错误、地形障碍或动作失败 | 检测、恢复和重规划 |

每个实例必须显式声明改变了哪些参数。难度标签不能替代具体场景配置。

## 4. 信息边界

### Agent 可以获得

- 自己的第一人称 RGB 画面；
- 任务文本和当前语义子目标；
- 允许公开的自身物品栏；
- 自己的历史摘要；
- 队友主动发送的消息；
- 工作流阶段和允许公开的共享任务板。

### Agent 不可直接获得

- 完整地图或全部方块坐标；
- evaluator 的有效门框内部标记；
- 未公开的附近实体列表；
- 队友未发送的私有观察、库存或记忆；
- 隐藏成功函数和失败注入标签。

代码必须使用不同类型和调用路径承载 `Observation` 与 `EvaluationState`，不能只靠
Prompt 约定隔离。

## 5. 动作协议

模型只能输出一个 JSON 对象，字段为：

```json
{
  "action_type": "place_block",
  "target": "obsidian",
  "duration_ticks": 1,
  "parameters": {
    "yaw": 0.0,
    "pitch": 0.0
  }
}
```

第一版允许动作：

- `wait`
- `look`
- `move`
- `equip_item`
- `mine_target`
- `place_block`
- `use_item`
- `craft_item`

协议规则：

1. 未知字段和未知动作拒绝；
2. 类型错误拒绝；
3. `duration_ticks` 限制为 1-40；
4. 相对 `yaw/pitch` 限制为 -30 至 30 度；
5. 移动轴限制为 -1 至 1；
6. 需要目标的动作缺失 `target` 时拒绝；
7. 解析失败产生一个单 tick `wait`；
8. 工作流和本地安全层可以进一步拒绝已经解析成功的动作；
9. 模型不得输出代码、命令、端口、文件路径或低层无限循环。

## 6. 评测器

Evaluator 必须使用环境真值，而不是模型理由文本。最低要求：

- 跟踪当前回合造成的关键方块变化；
- 检测门框几何与朝向；
- 检测传送门方块生成；
- 记录激活 tick；
- 记录每个角色的维度变化；
- 输出里程碑证据；
- 输出失败类型；
- 防止已有结构误判。

`portal_built_by_episode` 是防止“偶然发现已有门”的必要条件。只有在完整成功时，
`success=true`。

### 6.1 门框几何规则（vanilla 1.16.5）

- 外宽 W、外高 H：4 ≤ W ≤ 23，5 ≤ H ≤ 23。
- 最小可激活门框：4×5。不接受 4×4 或更小。
- 朝向：`plane_z`（门框在 X-Y 平面，常数 Z）或 `plane_x`（门框在 Y-Z 平面，
  常数 X）。两种朝向尺寸约束完全相同。
- 边框：每个边框 cell 必须是 `obsidian`。
- 缺角规则：四个角 cell 可缺。其余边框（`2W + 2H - 8` 个 cell）必须
  全部为 `obsidian`。
  - 完整外周（包含四角）共 `2W + 2H - 4` 个 cell。
- 内部空间：W-2 × H-2。内部 cell 仅允许 `air`、`nether_portal`、`fire`。
  任何其他块（`dirt` / `bedrock` / `obsidian` / `grass` / `grass_block`
  / `other` / `missing`）都视为阻挡，并使该候选失效。
- 激活：某 frame 候选内部出现至少 1 个 `nether_portal` cell 视为激活。
  激活状态被锁存，激活 step 被锁存。
- 维度切换：agent 的 `dimension` 变为 `minecraft:the_nether` 时永久加入
  `agents_in_nether`，对应 `first_nether_step_by_agent` 锁存。

### 6.2 门框由本回合建造（portal_built_by_episode）

- 候选 frame 的所有**非角** cell（共 `2W + 2H - 8` 个，缺角规则下）必须满足：
  - 当前 grid 中是 `obsidian`；
  - baseline grid 中**不是** `obsidian`。
- 4 个角 cell 在缺角规则下不计入“本回合新增黑曜石”。
- baseline 中已是 obsidian 但当前 grid 仍是 obsidian 的 cell，**不计入**
  “本回合新增黑曜石”。
- 该规则防止：门框在 reset 时已存在、被外部因素激活、门框在别处被点亮等
  情况被误算为成功。
- 门框的 latched frame identity 在首次锁存后保持，即使后续 grid 被 Nether
  grid 替换，verdict 仍然有效。

### 6.2.1 Obsidian attribution（关键补充）

- backend 维护一个 pending 队列：每次收到 `place_block(obsidian)` 动作
  自增 1。
- 每 step 结束后把 obsidian delta（current - baseline - 已 attributed）
  按 FIFO 匹配到 pending；匹配的进入 `attributed_obsidian_offsets`，
  剩下的进入 `external_obsidian_offsets`。
- `selected.required_frame_blocks` 必须全部属于
  `attributed_obsidian_offsets`，frame 才算 `is_episode_built=True`。
- 任何从外部写入的 obsidian（即不匹配 pending 队列的 delta）即使构
  成几何合法 frame，也属于 `frame_not_built_by_episode`。
- 单角色受控 A0 归因：credit 只在动作的 post-step observation 有效；
  fresh obsidian delta 数必须与 accepted action credit 数完全相等，否则
  fresh cells 全部归为 external。credit 不跨 observation 存活，已分类
  external 的 cell 永不重新归因。如未来 bridge 能提供"动作 → cell 坐标"
  映射，应替换此保守 heuristic。

### 6.3 激活绑定

- 激活只对锁存的 episode-built frame 内部 cell 计算。
- 预存的、attribution 失败的门框内的 `nether_portal` 不得触发本回合
  `portal_activated`。
- `latched_activation_offsets` 保存精确的 `nether_portal` 坐标，激活 step
  与 latched frame step 一起锁存。

### 6.3.1 进入下界与 episode portal 的关联（关键补充）

完整成功条件 `success=true` 要求所有四项都满足，**并且**
`entered_via_episode_portal=True`（至少一个 Nether-entering agent 被
证实从 latched frame 内部或紧邻区域进入下界）。

- backend 在 dimension 从 overworld 变 nether 的 step 锁存该 agent 的
  `pre_transition_position_by_agent[agent_id]`（最后一次 overworld
  position）。
- atSpawn grid offset 先通过 reset 时的实际 world-position anchor 转换为
  world bounds；不得把 grid-relative y 直接当成 world y。
- `_position_matches_latched_frame(position, tolerance=1.5)` 仅作为
  sanity check，不是因果证据。
- True 必须同时满足：server-side portal change guard 确认该 dimension change
  来自 `Entity.updatePortal()`、bridge typed
  `portal_transition.entered_via_portal` 为 true、bridge 返回的 source portal
  block world position 位于 latched frame interior world bounds 内、
  pre-transition position 通过邻域检查。
- bridge 明确返回 false 或 identity 不匹配时为 False；证据缺失/无效时
  保持 unset，evaluator 输出 unknown。False/unknown 均 `success=False`。

### 6.4 里程碑序列

每个事件都是 `StructuredEvent` 顶层对象，包含 `episode_id` / `step_id` /
`event_type` / `timestamp` / `agent_id` / `payload` 字段。`timestamp`
锁存于首次观察；如果 `milestone_step` 被设置但 `latched_timestamps` 缺
对应 key，`EvaluationState.__post_init__` 会抛 `ValueError`。多 agent
Nether 转换用 `agent_entered_nether:<agent_id>` 区分 timestamp。

| 里程碑 | 触发条件 |
|---|---|
| `task_reset` | `reset` 成功 |
| `first_obsidian_placed` | 当前 grid 出现 baseline 中不是 obsidian 的 obsidian |
| `build_site_selected` | 首次检测到 partial frame candidate（结构连续性：同一边 ≥ 3 块 episode 增 obsidian，或共享 corner 的 L 形） |
| `valid_portal_frame` | 首次出现 episode-built 合法 frame candidate；锁存 `latched_frame_identity` |
| `portal_activated` | 锁存 frame interior 内首次出现 `nether_portal` |
| `agent_entered_nether` | 某 agent 的 dimension 首次变为 `minecraft:the_nether` |

每个里程碑只触发一次；锁存 step 与 timestamp 写入 evidence。

### 6.5 终止与失败分类

`EvaluationState.episode_terminated` 是由环境、driver 或显式
`mark_terminated()` 设置的强制信号。**未终止**的 episode 一律为
`in_progress`（`failure_type=None`、`failure_step=None`），即便已经看
到终局条件。

终止后按以下优先级输出 `failure_type`：

1. `frame_not_built_by_episode`：观察到 attribution 失败候选（门框几何合法
   但其边框 cell 在 baseline 中已是 obsidian，或其 required cells 全部来
   自 `external_obsidian_offsets`）。
2. `frame_never_valid`：未观察到任何 episode-built frame candidate。
3. `portal_never_activated`：episode-built frame 已锁存但未激活。
4. `no_agent_entered_nether`：已激活但无 agent 进入下界。

`failure_step` 是 episode 终止 step（`terminated_step`），即不可恢复的首
次确认点；不是“当前 step”。

## 7. 单智能体基线

| 方法 | 说明 | 研究用途 |
|---|---|---|
| Single-Direct | 当前观察直接生成下一宏动作 | 原始闭环能力 |
| Single-Workflow | 提供工作流与当前阶段 | 工作流收益 |
| Single-Reflection | 阶段后有限自检、重试和重规划 | 错误恢复 |
| Single-Knowledge | 可选的完整路线知识或依赖图 | 知识增强 |

基线必须共享任务、动作、预算和 evaluator。Prompt 变化作为独立配置记录。

## 8. 多智能体基线

| 方法 | 角色数 | 通信 | 研究用途 |
|---|---:|---|---|
| Single | 1 | 无 | 单角色基线 |
| Dual-NoComm | 2 | 禁止 | 纯并行收益 |
| Dual-Chat | 2 | 自然语言消息 | 自由通信收益 |
| Dual-Workflow | 2 | 共享结构化任务板 | 结构化协作收益 |

多智能体角色不能访问对方未发送的私有观察。必须报告总模型调用和总推理成本，
不能只比较环境时间。

## 9. 指标

主指标分别报告，不构造模糊综合分数：

- Success Rate；
- Milestone Completion Rate；
- Time to Portal；
- 环境步数与宏动作数；
- 重复或无进展动作比例；
- Recovery Rate；
- 模型调用、输入/输出 Token、费用、P50/P95 延迟；
- 多角色消息数、通信 Token 和无效消息比例；
- Handoff Success；
- Coordination Conflict；
- Idle Rate。

多智能体收益分解：

```text
Parallelism Gain = SR(Dual-NoComm) - SR(Single)
Communication Gain = SR(Dual-Chat) - SR(Dual-NoComm)
Structured Collaboration Gain = SR(Dual-Workflow) - SR(Dual-NoComm)
Total Multi-Agent Gain = SR(Dual-Workflow) - SR(Single)
```

## 10. 失败分类

失败记录至少包含最后完成的里程碑、失败类型、失败 tick、责任角色（如适用）和
证据路径。标准类型见 `ROADMAP.md`。

模型超时、环境崩溃和 API 错误属于系统或推理基础设施问题，不能伪装成任务能力
失败。

## 11. TaskInstance

任务实例必须包含：

- `task_id`
- `route`
- `difficulty`
- `agent_ids`
- `world_seed`
- `instruction`
- `spawn_positions`
- `initial_inventories`
- `workflow`
- `milestones`
- `limits`
- `split`
- `schema_version`

实例需要通过 `benchmark/schemas/task_instance.schema.json` 验证。

## 12. 轨迹事件

每行 JSONL 是独立事件，至少包含：

- `episode_id`
- `step_id`
- `event_type`
- `timestamp`
- `agent_id`（角色事件必需）
- `payload`

模型事件可以保存原始公开响应和决策摘要，但不得记录 API key、认证头或模型隐藏
推理链。

## 13. 实验可复现性

每次运行保存：

- TaskInstance 和全部配置快照；
- 代码提交标识和 dirty 状态；
- 环境、模型和 Prompt 版本；
- 随机种子和预算；
- 结构化事件与 summary；
- 人工审查结论（正式验收需要时）。

不同任务集、Prompt、动作协议、预算或 evaluator 版本的运行不得合并为同一公平
对照。
