# ObsidianLink Roadmap

> 完整研究与开发计划见 `docs/plans/`。本文是**唯一动态状态源**：Current Phase、Current Task、Completed、Next、Blocked 只在这里维护。不要把进度写回冻结计划、README 或 AGENTS.md。

## Current Phase

**General Minecraft Agent Plan — Phase 1: Core Framework**

用户于 2026-08-21 明确调整当前优先级：暂停 Benchmark / Nether Portal 工作，先开发 Agent 本身。保留既有 benchmark 代码与历史证据，但当前不重构、不扩展。

当前通用入口架构：

```text
Natural-language Task → GeneralAgent → Planner
                                      ├─ Subgoal decomposition
                                      ├─ Wiki Knowledge → Memory → Planner
                                      └─ Primitive Skill → Controller
                                                              ↓
              Memory ← Reflection ← Observation ← Fake or live Environment
                                      ↓
                                 Next Decision
```

约束：单 Agent；Planner 只调用 Wiki knowledge tool、primitive skill 或 finish；不引入 LangChain、复杂 Agent framework、Multi-Agent、Vision 或 RAG pipeline。

## Current Task

**Agent Reasoning Foundation**

2026-08-22 Cognitive Retrieval + Long-Horizon Planning：

* 新增本地、确定性 Memory Retrieval：按当前 goal/subgoal 对 semantic / episodic / spatial memory 做词法相关性、失败优先级、recency 与 spatial confidence 排序；结果有界进入 `retrieved_memory`，不引入 embedding、vector DB 或 RAG
* `GeneralAgent` 每轮自动准备相关记忆；Planner 也可显式输出有预算的 `memory` decision，指定 query、memory types 与 limit，再根据返回记忆调整计划；外部未知 Minecraft 规则仍走 Wiki
* Wiki structured knowledge 增加 `spatial` 类型，抽取 generation / biome / dimension / Y-level 等位置规则；同时保存到 Semantic Memory，并以 `source=wiki`、较低 confidence 写入 Spatial Memory，与 agent-observed location 区分
* Hierarchical plan 增加可审计 revision history；subgoal 记录 attempts、failures、status 与 last outcome，失败会显式标记节点，后续 Planner 可恢复同一节点或修改 downstream plan；completed/skipped 状态保持单调，避免长任务回退循环
* `GeneralAgentResult` 增加 explicit `memory_queries` evidence；旧 Planner JSON、GeneralAgent、Autonomous prototype 与 primitive-only skill surface 保持兼容
* 新增 retrieval ranking/type filter、active Planner retrieval、spatial Wiki integration、failure recovery state 与 plan revision 测试；完整 offline regression：**210 passed**；未启动 MineRL/Minecraft，未改 benchmark task，未增加 workflow skill、Vision 或 Multi-Agent

2026-08-22 Cognitive Layer 重构：

* Wiki 从单一搜索摘要升级为有界的 `search → article parse → structured knowledge`：解析正文段落、章节与表格，输出 `recipe` / `item` / `mechanic` 类型及可审计 attributes；正文请求失败时安全回退搜索摘要
* `WikiKnowledge` 优先查询 `AgentMemory.semantic_memory`；相同规范化 query 跨 task 复用，不重复访问网络，也不消耗 live Wiki call budget
* `AgentMemory` 明确支持四层：working（当前 task/plan/observation）、episodic（成功与失败经历）、semantic（Wiki 结构化知识）、spatial（Agent 可见来源的位置/资源）；默认只重置 working memory，长期记忆可影响后续 Planner prompt
* Planner 增加稳定 subgoal ID、parent、dependency、status、attempt/outcome 与 plan revision；兼容旧 `subgoal` / `pending_subgoals` JSON，同时优先使用结构化 `Goal → Subgoal → Primitive Skill` plan
* Planner prompt 同时读取 working / episodic / semantic / spatial memory，根据 expected-vs-observed mismatch 和历史失败修改 downstream plan；默认 skill surface 仍为 primitive-only
* 新增 Knowledge / Memory / Planner / cache compatibility 测试；完整 offline regression：**204 passed**；未启动 MineRL/Minecraft，未改 `obsidianlink/benchmark/`，未增加 workflow skill、Vision 或 Multi-Agent

2026-08-21 当前实现：

* 新增 `FakeMinecraftEnv`：不启动 Minecraft/MineRL，模拟 inventory、可采矿资源、hotbar、放置与 GUI crafting；primitive skill 仍发同一套 `Action`，Observation 只有 `frame` / `inventory` / `selected_item`
* Planner 从单步选择提升为任务分解：`pending_subgoals` + `current subgoal` + 可选 `expected` 观察结果；prompt 要求根据 subgoal progress、last_reflection、failure history、knowledge usage 调整计划
* Memory 增加 `subgoal_progress`（current / completed / pending）、`knowledge_usage`、`failure_history` 与 `last_reflection`
* 轻量 Reflection：skill 后比较 expected vs observed；不匹配则写入 Memory，影响下一轮 Planner。不是额外 LLM、也不是 reflection framework
* Offline reasoning 闭环已验证：Wiki → 过早 attack 失败 → Memory 记录 mismatch → 改为 move → 再 attack → inventory 验证成功
* 完整 offline regression：**198 passed**；`obsidianlink/benchmark/` 无变更；未恢复 `collect_wood` / `mine_iron` / `build_portal`

**GeneralAgent Planner / Memory Foundation**

2026-08-21 先前实现：

* Planner 决策循环明确为 `Task → Subgoal → Primitive Skill/Wiki → Observation → Memory → Next Decision`
* `PlannerDecision` 增加可选 `subgoal`；`LLMSkillPlanner` prompt 使用结构化 Memory 与 agent-visible Observation 摘要，不再把 skill metadata 当日志塞进上下文
* `AgentMemory` 从步骤日志改为决策状态：`task_status`、`current_subgoal`、`completed_subgoals`、`wiki_knowledge`、`recent_failures`、`inventory_delta`、当前 inventory / selected item
* skill 失败、被拒绝的 `finish`、Wiki 错误写入 `failed_attempts`，供下一轮 Planner 改策略而不是盲目重试
* `Observation` 契约不变（仍只有 `frame` / `inventory` / `selected_item`）；新增 `agent_view()` 给 Planner，不含 RGB 像素与 evaluator-only pose
* 完整 offline regression：**189 passed**；`obsidianlink/benchmark/` 无变更

**GeneralAgent Wiki Knowledge + Primitive Skill Foundation**

2026-08-21 先前实现：

* `GeneralAgent` 正式处理 Planner 的 `wiki` decision；查询经 `WikiKnowledge` / `MinecraftWikiTool` 访问 live Minecraft Wiki，成功结果写入 episode-local `AgentMemory.known_knowledge`，下一 planning cycle 自动进入 prompt context
* Wiki 调用有独立 `max_wiki_calls` 边界；网络/协议错误写入 `memory.last_error`，不引入缓存语料、embedding、向量数据库或 RAG pipeline
* 默认 `SkillLibrary` 已改为 primitive-only：`move`、`look`、`attack`、`interact`、`select_hotbar`、`inspect_inventory`、`place_block`、`crafting_action`、`wait`
* 默认 Planner surface 不再暴露 `collect_wood`、`explore_area`、完整木镐 `craft_item` 或 `build_structure`；复杂任务由 Planner 组合 primitives
* 历史 workflow 类未删除，集中通过 `legacy_workflow_skill_library()` 显式注入，供旧 `AutonomousMinecraftAgent` 与 live smoke runner 兼容使用
* 本阶段只做基础能力与 offline contract tests，不启动 MineRL 任务实验；当时完整 offline regression：**182 passed**；`obsidianlink/benchmark/` 无变更

* 新增 `obsidianlink.agents.GeneralAgent`，以自然语言任务作为统一 `run(task)` 入口，不包含 Portal 专用目标
* 核心闭环为 `Task → Planner → primitive skill Action → Environment → Observation → AgentMemory`
* 复用现有 `TaskPlanner`、`AgentMemory`、`MinecraftController`、`SkillLibrary` 与 LLM model client 层；保留 `LLMAgent`、`AutonomousMinecraftAgent`、Portal/RuleBased/Random baselines
* 支持注入只读取 agent-visible Observation/Memory 的 `GoalVerifier`；有 verifier 时拒绝未经验证的 `finish`，无 verifier 时允许 planner 声明完成
* skill 失败或异常会记录为 `StepRecord` 并返回 Planner 重规划；planner、environment reset 与 verifier 边界返回结构化失败结果
* `LLMSkillPlanner` 可生成 `skill` / `wiki` / `finish`，也可用 `allow_wiki=False` 显式关闭知识查询；始终不包含 RAG、Vision、Multi-Agent 或复杂 Skill System
* 新增 4 个通用闭环测试；完整离线回归 **163 passed**
* 新增 `run_general_agent.py` 与 agent-local smoke task（不进入 benchmark）：默认任务 `Mine 1 obsidian block`
* 新增 `GeneralBlockSmokeEnv`：真实 MineRL/Minecraft survival mode，固定出生点、diamond pickaxe 与单个 MCP-Reborn 允许的 obsidian 方块；成功只读 agent-visible inventory
* **Live success（2026-08-21）**：`general_live_phase1_obsidian3`，1 次 Planner → `mine_block` → 240 ATTACK + 12 MOVE + 2 WAIT → inventory `obsidian=1`；254 steps；`success=True`
* `MineBlockSkill` 支持目标 inventory feedback、独立 pickup/settle 阶段；`MoveForwardSkill` 提供 bounded basic movement；Controller 记录 action counts
* 修复 GeneralAgent 在最后 planning cycle 执行 skill 后不再次验证目标的问题
* 自然森林 `collect_wood` 已加入树干/岩浆诊断、连续 attack burst 与稀疏 POV trace；真实方块破坏曾观察到，但随机地形下尚未稳定完成 inventory pickup
* 完整 offline regression：**175 passed**；`obsidianlink/benchmark/` 无变更

既有木镐 autonomous prototype 保留，当前不继续扩展 Portal 专用 Agent：

* 新增 `AutonomousMinecraftAgent`：`observe → plan → skill/wiki → memory update`；Planner 只接受 `skill` / `wiki` / `finish`，拒绝 `move` / camera / mouse 等低级动作
* 新增 episode-local `AgentMemory`：目标、完成步骤、Wiki 知识、inventory / selected item、last error
* 新增高层 skill library：`collect_wood` / `mine_block` / `craft_item` / `explore_area` / `build_structure`
* 新增 `MinecraftController` 与 `WoodPickaxeEnv`；旧 benchmark 路径未重构
* 复用 live Minecraft Wiki，并把查询结果写入 memory；无 embedding / vector DB / RAG pipeline
* MCP-Reborn live 发现：结构化 `CraftAction` 会把 `craft none` 送入 `constructKeyboardState` 并触发 `NumberFormatException`，因此不使用该 handler；`craft_item` 改走真实 inventory / crafting-table GUI 控制
* Live environment smoke：reset / WAIT / inventory open / inventory close 全部成功；POV `(360, 640, 3)`，真实初始 inventory `iron_axe=1`
* 实现当时 Offline tests：159 passed
* **尚未获得 live task success**：`collect_wood(quantity=3)` 在 3 次有界 run（300 / 500 / 500 steps）均为 `0/3 logs`；已加入 RGB tree servo、dense forest generator 与 jump movement，但目标接近仍未解决。因此 GUI 木镐制作只通过 fake-environment 闭环，尚未 live 验证

当前第一个 controlled live task 已完成。本轮按用户要求不启动任务实验；下一步优先验证 primitive 的 live 可靠性与 Planner 多步组合，再恢复自然森林任务测试，不恢复 Nether Portal 专用开发。

## Prior Benchmark Context

**L1 LLMAgent Vision Baseline v3** 已完成（见下）。RGB POV 已接到 MiniMax-M3；视觉 grounding 明显提高 world interaction，但 276/500 步因 `agent_exception` 中断，任务仍未成功。不要加 memory / RAG / planner。

**2026-08-21 L1 LLMAgent Vision Baseline v3**（`run_l1_llm_agent.py --vision`，`l1_llm_20260821_071744Z`）：

* RGB 来源：`MineRLEnvironment` `pov` → `Observation.frame`，live shape `(360, 640, 3) uint8`（spec `RESOLUTION=(640, 360)`）
* MiniMax-M3 支持 `image_url`；JPEG data-URL 随 prompt 发出。`LLMAgent(use_vision=True)`，Agent interface 未改
* 276/500 steps 后 `agent_exception` 中断（wall ≈ 1856s，≈6.7s/step）。全程 `vision_calls=276`，`last_used_vision=true`，无 text fallback。parser 276/276
* Evaluator：`success=False`，`portal_activated=False`，`nether_entered=False`
* 动作：camera 166（60.1%），**move 48（17.4%）**，hotbar 37（13.4%），**use 16（5.8%）**，wait 7（2.5%），**attack 2（0.7%）**
* vs v2（500 步 text）：world interaction（move+use+attack）4.0% → **23.9%**；camera 仍最多但不再几乎只有转视角
* `valid_for_l1_capability_conclusion: false`（未跑满 500，且未激活 portal）。未改 env / evaluator / oracle / Task schema
* Offline tests: 151 passed（2026-08-21）

**2026-08-21 L1 LLMAgent Prompt Baseline v2**（`obsidianlink/experiments/run_l1_llm_agent.py`，`l1_llm_20260821_064713Z`）：

* Prompt：明确 Nether Portal 四步目标 + 行为约束；**不**写 lava / water / bucket casting / recipe。JSON 输出格式不变
* Horizon：500 steps（官方 task 仍是 4000）。`L1Evaluator` / success 定义未改
* MiniMax-M3 500/500 调用，parser 498/500（`invalid_actions=2`），Minecraft `env.step` 500/500，wall ≈ 1013s
* Evaluator：`success=False`，`portal_activated=False`，`nether_entered=False`，`failure_reason=nether_entry_not_confirmed`
* 动作分布：camera 328，hotbar 77，wait 75，move 14，**use 5**，**attack 1**
* vs Baseline v1（64 步，hotbar 45 / wait 9 / camera 8 / move 2 / use 0 / attack 0）：hotbar 占比 70% → 15%；首次出现环境交互，但 camera 占 66%，USE/ATTACK 合计 1.2%
* `valid_for_l1_capability_conclusion: false`。未改 env / evaluator / oracle / Task schema / Agent interface
* Offline tests: 147 passed（2026-08-21）

**2026-08-21 First L1 LLMAgent portal episode**（`obsidianlink/experiments/run_l1_llm_agent.py`，`l1_llm_20260821_063413Z`）：

* 正式 `MineRLL1Controlled-v0` + `L1_PORTAL_TASK` + `L1Evaluator` + `LLMAgent(MiniMax-M3)`
* Prompt 使用 portal goal（不是 smoke goal）。无 planner / memory / RAG / hard-coded solver
* `--max-steps 64`（官方 task 预算 4000）。MiniMax 64/64，parser 64/64，Minecraft `env.step` 64/64，wall ≈ 133s
* Evaluator：`success=False`，`portal_activated=False`，`nether_entered=False`，`failure_reason=nether_entry_not_confirmed`（`portal_activation_not_confirmed`）
* 动作：hotbar 45，wait 9，camera 8，move 2；**0 USE / 0 ATTACK**。模型在选物品和转视角，没有对岩浆/水 `use`
* `valid_for_l1_capability_conclusion: false`（预算低于 4000，且未发生液体交互）。未改 env / evaluator / oracle / task / Agent interface
* Offline tests: 146 passed（2026-08-21）

**2026-08-21 MiniMax default = China + 4-step live smoke**（`obsidianlink/experiments/run_llm_smoke.py`，`llm_smoke_20260821_050510Z`）：

* 默认 endpoint 从 `api.minimax.io` 改为 `api.minimaxi.com`；key 仍只读 `MINIMAX_API_KEY`
* 不是 Nether Portal：`--max-steps 4`。MiniMax-M3 4/4 API 成功，parser 4/4，Minecraft `env.step` 4/4，wall ≈ 29s
* 本 episode 仍只输出 `camera`（yaw 30 / 20 / -30 / 30，最后一步 pitch=-10）。frame mean 66.2 → 64.4 → 60.8 → 65.9
* `valid_for_l1_agent_conclusion: false`。未改 env / evaluator / oracle / task
* Offline tests: 144 passed（2026-08-21）

**2026-08-20 First LLM embodied control experiment**（`obsidianlink/experiments/run_llm_smoke.py`）：

* 目标不是 Nether Portal：prompt 只要求 camera / move / hotbar / USE-or-WAIT
* `llm_smoke_20260820_150552Z`：Minecraft reset 成功；`api.minimax.io` HTTP 401 `invalid api key (2049)`（国际站拒国内 key）
* `llm_smoke_20260820_151507Z`：**成功**。改用 `https://api.minimaxi.com/v1/chat/completions` 后 MiniMax-M3 8/8 调用成功，parser 8/8，Minecraft `env.step` 8/8，wall ≈ 33s
* 模型本 episode 只输出 `camera`（yaw ±30，最后一步 pitch=-10）。frame mean 69.4 → 53.2，说明相机确实转了。未做 move / hotbar / USE
* `valid_for_l1_agent_conclusion: false`。未改 env / evaluator / oracle / task
* MiniMax client 现对 401 会回退国际站 ↔ 中国站

**2026-08-20 Agent Interface Layer**（`obsidianlink/agents/base_agent.py`）：

* `BaseAgent.reset` / `act(observation)` — 最小统一 Agent API，无 framework
* `RandomAgent`：在 L1 合法动作面（MOVE / CAMERA / USE / ATTACK / HOTBAR / WAIT）上随机采样；不发 EQUIP / PLACE
* 规则 `ReactiveAgent`（`reactive_agent.py`）：inventory FSM 浇灌 baseline，**不是** LLM。现有视觉/Wiki agent 仍在 `agents.reactive`
* `experiments/run_agent.py`：`reset → act(obs) → env.step(action)`；reward/done 只由 runner 从 `hidden_state` 读出并打印，从不传入 `agent.act`
* 这是 interface smoke / rule baseline，不是 L1 能力结论

**2026-08-20 LLM Agent Adapter**（`obsidianlink/agents/llm_agent.py`）：

* `LLMAgent(BaseAgent)`：`Observation → prompt → BaseLLMClient.generate → parse_action → Action`
* `obsidianlink/models/`：`BaseLLMClient.generate(prompt) -> str`；`MiniMaxClient` 读 `MINIMAX_API_KEY`，默认 `api.minimaxi.com`，不把 key 写入代码
* `agents/prompt.py`：任务目标 + Observation + 合法 action space + JSON 输出格式；非法 JSON / EQUIP / PLACE fallback 为 WAIT
* `run_agent.py --agent llm`；offline tests 不启动 Minecraft、不打真实 API
* 无 memory / RAG / planner / LangChain / multi-agent / tool calling。这是 adapter，不是模型评测结论

**Formal L1 Controlled Construction**（背景，本阶段不扩展）

端到端目标保持 method-agnostic：

```text
Construct / complete → activate → enter Nether
```

Bucket Casting 是第一版受控评测的主要 reference strategy，而非强制 solver。Agent 可使用 live Minecraft Wiki tool 查询任务相关的原版知识；Evaluator 仍只从 evaluator-only world truth 判断 portal activation 与 Nether transition。

不要把 scene 预建 portal frame 当作正式 L1。旧 L1 已移出 active path。

**2026-08-19 technical feasibility spike**（`obsidianlink/experiments/spike_l1_feasibility.py`，live `l1_spike_20260819_124538Z`）：

* bucket casting **可行**：真实 `use` lava/water 生成新的 obsidian，不是 DrawBlock 预建 frame
* 物品准备：**可行**，`InventoryAgentStart`（不改 L1 语义）
* 选物品：必须用 `hotbar.1-9`。`EquipAction` 会把 `equip none` 送给 MCP-Reborn 并 crash
* Oracle 走到：inventory → hotbar → pour lava → water convert → pickup → 至少 1 次 extra cast
* 未完成：10-block frame / flint ignition / Nether entry（extra cast 期间 connection timeout）
* Evaluator 不要用 `ObservationFromGrid`。可用：inventory delta、`location_stats`（gym info）、POV 帧、可选 `RewardForTouchingBlockType(nether_portal)`

该 spike 是 scripted/oracle mechanics feasibility，不是正式 L1 Agent 或 live L1 benchmark evidence。正式 L1 不用预建 frame，也不使用已知不可靠的 EquipAction。

**2026-08-19 L1 Controlled Environment v0.1**（`obsidianlink/env/l1_scene.py`，live `l1_env_smoke_20260819_175245Z`）：

* 施工面是超平坦 **草地**（`FLAT_WORLD`），不是黑曜石地板。DrawBlock 只画 4×4 lava pool
* 原因：Malmo DrawingDecorator 只能画 `lava` / `obsidian`；黑曜石地板会让 Agent 把地面当 portal 材料
* spawn `(0.5, 4.0, 0.5)`；inventory 与 `hotbar.1-9` 与上一版相同
* live `C2_grass_floor` 通过（`grass_frac≈0.76`）；无预建 portal，无 EquipAction
* 这是 environment smoke，不是 Oracle、Evaluator 或 ReactiveAgent L1 实验

**2026-08-20 L1 Mechanical Interaction Test**（`obsidianlink/experiments/run_l1_mechanics.py`，live `l1_mechanics_20260820_033330Z`）：

* 在正式 `MineRLL1Controlled-v0` 上，用未来 Agent 同款动作（MOVE / CAMERA / USE / ATTACK / HOTBAR / WAIT，外加 `sneak` 修饰）完成浇灌法关键 mechanics
* empty bucket 从 lava source 舀岩浆 → `lava_bucket`；`use` 放岩浆；`use` 放水；lava+water 生成 **至少 1 块新 obsidian**（非 DrawBlock / 非预建）
* cobblestone 经 hotbar + 原生 `use` 放置；iron pickaxe `attack` 拆除后 inventory `63 → 64`
* 未使用 EquipAction、ObservationFromGrid、PlaceBlock
* **NEW OBSIDIAN = TRUE**。这是机械可行性验证，不是 Oracle、Evaluator 或 Agent 能力结论

**2026-08-20 L1 Evaluator**（`obsidianlink/benchmark/l1_evaluator.py`，live `run_l1_evaluator_smoke.py`）：

* evaluator-only truth channels，live-verified on this MineRL 1.0.2 / MCP-Reborn / Malmo 0.37.0 stack：
  * `portal_activated` / `portal_contacted` ← gym step `reward` from `RewardForTouchingBlockType(nether_portal)`（新增到 `L1ControlledEnv.create_rewardables`；reward 只经 `MineRLEnvironment.hidden_state["reward"]`，从不写入 `Observation`）
  * `nether_entered` ← `biome_id == 8` (Nether) from `ObservationFromCurrentLocation`/`location_stats`，且必须发生在 `portal_activated` **之后**（防止 biome 噪声单独判定成功）
  * `portal_constructed` 保持 `"unknown"`：这台 stack 上没有便宜可靠、非 `ObservationFromGrid` 的 frame-complete 真值，不为它开发 block parser
* `success = nether_entered`，`nether_entered` 必须同时满足 activation 证据与 strict biome match；只有弱 biome 变化（未命中 8）不算 success
* live smoke 确认：`hidden_state` 含 `reward`/`biome_id`/`can_see_sky`/`light_level` 等字段，Agent-visible `Observation`（`frame`/`inventory`/`selected_item`）不变；空场景下 `evaluator.evaluate(...)` 正确 fail-closed（`success=False`, `reason=nether_entry_not_confirmed`）

**2026-08-20 Water Recovery Isolation**（`obsidianlink/experiments/run_water_recovery_isolation.py`，live `water_recovery_iso_20260820_105237Z` + `105355Z`）：

* 最小场景：fresh reset → 低头放 1 格 water source → fluid wait → 单次 `USE` 回收 → **20 tick 纯 WAIT**（无 USE / ATTACK / MOVE / HOTBAR / CAMERA；每 tick 记录 mapped `minerl.use`）
* 两次独立 episode 时间线相同：放水前 `bucket=1, water_bucket=1`；pour `USE` 当 tick（tick 3）稳定为 `bucket=2, water_bucket=0`；recover `USE` 当 tick（tick 13）出现 `water_bucket=1, bucket=1`；随后 20 WAIT 全程保持，`minerl.use=0`
* **没有 inventory rollback**。`water_bucket=1` 在这个 WAIT-only 窗口里不是 transient observation，也不是 delayed sync 后被权威状态打回 empty bucket
* 因此 **不要** 为这个 primitive 加 `N=3` consecutive confirmation，也不要把单帧 `water_bucket>=1` 一律当成假读数
* 这不是 Gate 1、不是完整地狱门。Gate 1 里曾经看到的 “CAMERA/WAIT 后 water_bucket 消失” **不能**用 “MineRL inventory 天生会回滚” 解释；剩余嫌疑是 recover 后的额外交互（多 tick `USE` burst、随后 CAMERA 瞄到残留流动水）或岩浆模具场景，本次 isolation 按协议没有测 CAMERA
* POV：回收后准星处水源消失，hotbar 变为 empty bucket + water bucket，与 inventory 一致

**2026-08-20 Gate 1 — one obsidian**（`run_l1_oracle.py`，live `l1_oracle_20260820_113909Z` + `114030Z`）：

* 最短序列：scoop lava → place lava → 邻格放水 → wait until obsidian。无模具、无 2 格底边、无 portal frame、无点火
* 桶交互全部 **单次 `USE`**（3 次 USE / episode）。约 65 steps，wall ≈ 28s，远低于 240s socket 超时
* 成功标准：`observed_new_obsidian`（inventory 链 + POV `obsidian_visual_rose`）。`L1Evaluator.success` 仍是 Nether entry，Gate 1 不为 True
* Run `113730Z` 是假阳性：对准岩浆格放水会把 lava source **替换成水**，`lava_frac` 下降但 `obsidian_frac` 几乎不变。已要求 `obsidian_visual_rose`，并改为邻格放水 + lava settle wait
* 修正后 2/2：`obsidian_frac` 0.0009 → 0.041 / 0.023，水保持放下（WAIT 时不再被空桶舀回）
* 限制：开阔地面岩浆仍会蔓延，这不是精确 portal 格；`exact_block_truth` 不可用

**2026-08-20 Full Scripted Oracle — 卡在 Gate 1**（`obsidianlink/experiments/run_l1_oracle.py` / `l1_oracle.py`，`obsidianlink/tasks/portal.py`）：

* Portal 参考几何：经典 cornerless 10-block frame（省略四角），`base_x=-1, base_y=4, z=3`，底边落在已验证的 y=3 grass 施工面正上方；offline tests 覆盖 frame/interior 形状、method-agnostic Task goal
* Live 发现 1 — **不加模具的浇灌不可控**：在开阔草地上倒岩浆会向四周多格蔓延，不会停留在单一目标格（有截图证据），之前 `l1_mechanics` 的“NEW OBSIDIAN=TRUE”只是启发式视觉判定，不是几何证明。修复：Oracle 在浇筑前先用 cobblestone 在目标两侧砌矮墙（mold）约束岩浆
* Live 发现 2（更严重，未解决）— **长 episode 会话可能挂起并触发 240s socket 超时**：`minerl/env/_multiagent.py` 硬编码 `SOCKTIME = 240s`。纯 `WAIT` 循环验证 93,200 步 / 340s 无异常（排除固定 wall-clock episode 上限）；但一旦 episode 内出现较多真实液体/方块放置动作（造好模具 + 舀岩浆 + 往返移动），会在约 270–280s 处遇到 `TimeoutError` → `RuntimeError: Attempted to step an environment server with done=True`，两次独立复现，失败点不完全固定在同一动作类型上（一次卡在放置模具的 `use`，一次卡在返回途中的 `move`），符合“服务器端因流体模拟负载累积而逐步变卡、最终某一步响应超过 240s”的特征，而不是我们代码的死循环（步数预算已大幅收紧后仍复现）
* Gate 1（浇筑 2 块底边 obsidian）**未在单次稳定 episode 内端到端确认**：模具搭建与至少一次舀岩浆已经真实成功，但受上述挂起风险影响，尚未拿到一次完整跑通 pour→water→obsidian 视觉确认的干净 run
* 未使用 EquipAction、PlaceBlock、ObservationFromGrid、DrawBlock portal、teleport、command、预建 frame、inventory 注入
* 结论：**Oracle SUCCESS = False**，停在 Gate 1（bottom row casting），不是几何/瞄准逻辑问题，而是 MCP-Reborn/Malmo 在长时间高频液体交互下的服务器端稳定性限制。下一步需要先定位/缓解这个挂起（例如减少单 episode 内的液体方块更新总量、拆分动作节奏、或确认是否有官方已知 issue），而不是继续堆 Gate 2-8 的建造逻辑

L1 背景约束仍然有效：不要自动继续完整 10 格 frame / 点火 / 入 Nether。精确几何仍需要模具，长 episode 240s 挂起风险仍在。不要提前开发 Planner / Reflection / L2。

## Completed

* Research direction frozen
* Research-First Master Plan frozen
* Development Plan frozen
* **2026-08-19 architecture reset**：删除被 Malmo workaround 污染的 L1 与过期 diagnostic 实现，重建最小 Environment / Agent / Runner
* **Phase 1 — Minimal Minecraft Agent Loop**
  * `MineRLEnvironment`: reset / observe / step / close
  * RGB + inventory + selected_item
  * bounded actions
  * Live 2026-08-19：`MineRLTreechop-v0` 16 steps，frame mean 117.5 → 115.3
* **Phase 2 — Benchmark MVP**
  * Task / Evaluator / BenchmarkRunner / Result
  * Agent-visible Observation 与 evaluator-only hidden_state 隔离
  * D1 以 env `target_truths` 为 ground truth；与 Task 标签冲突则为 evaluation_error
  * Runner 将 env / agent / evaluator 异常写成结构化 Result，不中断实验
  * `Task.allowed_actions` 外的动作被夹成 WAIT 并计入 invalid_actions
  * 代表性 diagnostic：D1 Lava Presence
  * Vision dispatch 必须把 `Observation.frame` 交给 vision model；fallback 写入 Result.evidence
  * Live 2026-08-19：Qwen3-VL-2B，`vision_completions=1`，`success=True`，GT 只在 hidden_state
* Live Minecraft Wiki Tool
* Tool-enabled ReactiveAgent
* **2026-08-21 Autonomous Agent prototype architecture**（LLM high-level planner + memory + skill library + controller + dedicated env；fake-environment wood→wooden_pickaxe 闭环通过；live task success 尚未通过）
* **2026-08-21 GeneralAgent Phase 1 core**：自然语言 `run(task)` 统一入口、Task→Planner→Skill→Environment→Observation→Memory 有界闭环、可注入 GoalVerifier、失败重规划；保留全部 baseline 与 benchmark；offline 163 passed
* **2026-08-21 GeneralAgent first live task**：controlled survival-mode MineRL 中 `Mine 1 obsidian block` 成功；Planner 1 call，254 environment steps，inventory Observation 验证 `obsidian=1`；未使用 benchmark/Wiki/RAG/Vision/Multi-Agent
* **2026-08-21 GeneralAgent Planner/Memory foundation**：Task→Subgoal→Primitive/Wiki→Observation→Memory 决策状态；offline 189 passed；未改 benchmark
* **2026-08-21 Agent Reasoning Foundation**：FakeMinecraftEnv + Planner 分解/进度 + Memory 知识使用 + 轻量 expected/observed reflection；offline 198 passed；未改 benchmark；未启动真实 Minecraft 任务
* **L1 Controlled Environment v0.1**（env + inventory + hotbar smoke；无 Oracle / 无 Agent）
* **L1 Mechanical Interaction Test**（正式 L1 上 scripted 浇灌 mechanics；NEW OBSIDIAN = TRUE；无 Oracle / 无 Evaluator / 无 Agent）
* **L1 Evaluator**（evaluator-only `reward` + `biome_id` truth；live-verified fail-closed；无 ObservationFromGrid；无 Observation 泄漏）
* **Formal L1 Portal Task**（method-agnostic goal，`obsidianlink/tasks/portal.py`）与 cornerless 10-block 参考几何（offline-tested）
* **Water Recovery Isolation**（单次 `USE` 回收 + 20 WAIT；2/2 fresh episodes 无 rollback；非 Gate 1）
* **Gate 1 one obsidian**（短 scripted 浇灌；修正后 2/2 `observed_new_obsidian=True`；非 portal frame）
* **Agent interface layer**（`BaseAgent` + `RandomAgent` + 规则 `reactive_agent.ReactiveAgent` + `run_agent.py`；Agent 只看 Observation；未改 evaluator / oracle / task）
* **LLM Agent Adapter**（`LLMAgent` + `BaseLLMClient` / `MiniMaxClient` + prompt/parser；`run_agent.py --agent llm`；未改 Environment / Evaluator / Oracle / task）
* **First LLM embodied control experiment**（`run_llm_smoke.py`：`150552Z` 国际站 401；`151507Z` 中国站 8/8 MiniMax + 8/8 Minecraft step，全是 camera；非 L1 能力结论）
* **2026-08-21 MiniMax default China + 4-step smoke**（`llm_smoke_20260821_050510Z`：默认 `api.minimaxi.com`，4/4 API + 4/4 parse + 4/4 `env.step`；非 L1 能力结论）
* **2026-08-21 First L1 LLMAgent portal episode**（`run_l1_llm_agent.py`，`l1_llm_20260821_063413Z`：正式 L1 env/task/evaluator；64/64 MiniMax+parse+step；`success=False`，无 USE；非 L1 能力结论）
* **2026-08-21 L1 LLMAgent Prompt Baseline v2**（`l1_llm_20260821_064713Z`：task-aware prompt + 500 steps；camera 328 / hotbar 77 / wait 75 / move 14 / use 5 / attack 1；`success=False`；非 L1 能力结论）
* **2026-08-21 L1 LLMAgent Vision Baseline v3**（`l1_llm_20260821_071744Z`：RGB→MiniMax-M3；276/500 后 agent_exception；move+use+attack 23.9% vs v2 4.0%；`success=False`；非 L1 能力结论）

## Next

1. 继续用 `FakeMinecraftEnv` 测更长的 primitive 组合（例如 log → planks），仍不启动真实 Minecraft 任务。
2. Reasoning 闭环稳定后再验证 primitive 的 live 可靠性（`move` / `look` / `attack` / `interact`）；不要恢复 `collect_wood` 等高层 workflow skill。
3. 不要提前开发 Multi-Agent、Vision pipeline 或 RAG。

不恢复 Nether Portal，不增加多 Agent、reflection framework 或通用 RAG pipeline。

## Blocked

* **Autonomous prototype live blocker（2026-08-21）**：`collect_wood` 在自然 Treechop world 中 300 / 500 / 500 steps 均为 0 logs；dense forest + RGB tree servo + jump 仍未解决目标接近。完整 wooden-pickaxe live success 因此前置阻塞。
* **GeneralAgent natural collect blocker（2026-08-21）**：live trace 已确认 movement、树干接近与真实 log block 破坏，但多树候选、泥土/树皮误判、岩浆避障和掉落拾取仍使 fresh random forest 的 `Collect 1 log` 不稳定；controlled obsidian smoke 不受此 blocker 影响并已成功。
* **Structured crafting blocker（2026-08-21）**：MCP-Reborn EnvServer 对 `craft none` 执行 `Integer.parseInt("none")` 并终止 episode；原型必须使用真实 inventory GUI，不能用 `CraftAction` handler。
* `MineRLNavigate-v0` 在本机 Malmo 0.37.0 上有 `NullPointerException`。Phase 1 使用 `MineRLTreechop-v0`。
* Malmo 0.37.0 已知限制（记录，不在本次用 workaround 改 Benchmark 定义）：
  * `<Placement>` / `/teleport` 可能被忽略
  * `<ChatCommands> /give` 可能不执行
  * `PlaceBlock` handler 可能 crash server
  * `ObservationFromGrid` 可能返回全 air，不能作为 evaluator truth
* DrawingDecorator：仅 `DrawBlock`，仅 `lava` / `obsidian`
* `EquipAction`：MineRL 1.0.2 MCP-Reborn `constructKeyboardState` 对 `equip none` / `equip <item>` 做 `Integer.parseInt`，episode 直接结束。L1 应使用 hotbar keys
* 流动水（flowing water）不能用空桶回收，只会推玩家。scripted 放置/挖掘圆石时准星必须打在方块上，不能打在水面。这不是 Benchmark 定义问题，也不要为此启用 PlaceBlock
* 在开阔地面（无模具）浇岩浆会自由蔓延到多个相邻格，不会停在单一目标格；构造精确几何（如 portal frame）前必须先用 cobblestone 砌矮墙围住目标格
* **2026-08-20**：长 episode（约 270-280s 内出现较多真实液体/方块放置动作）可能触发 `minerl/env/_multiagent.py` 硬编码的 240s socket 超时（`RuntimeError: Attempted to step an environment server with done=True`）。纯 `WAIT` 循环 93,200 步 / 340s 验证无此问题，说明不是固定 episode 时长上限，而更像服务器端因液体模拟负载累积导致某一步响应变慢。两次复现的具体失败动作类型不同（一次 `use`、一次 `move`），指向服务器端渐进变卡而非单一确定性触发点。这是 Full Scripted Oracle 卡在 Gate 1 的直接原因之一，需要专项调查（不要通过修改 L1 语义规避）
* **2026-08-20 MiniMax live smoke**：同一把国内 key 在 `api.minimax.io` 返回 HTTP 401，在 `api.minimaxi.com` 成功。不要为此修改 Environment。client 默认中国站，401 时回退国际站

## Historical L1

`obsidianlink/experiments/runs/l1_*` 是 debugging record，**invalid for L1 capability conclusion**。原因：

1. L1 semantics changed（Casting/Frame 被移入 scene）
2. evaluator world truth unreliable
3. Reactive run did not actually use vision
