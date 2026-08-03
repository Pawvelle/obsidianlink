# ObsidianLink Development Roadmap

本路线图把 ObsidianLink 的总体规划转化为可执行的工程阶段。阶段编号描述依赖关系，
不是允许并行堆功能的清单。只有当前阶段的退出条件全部满足，才进入下一阶段。

## 总体原则

- 先任务与评测，后模型能力。
- 先小规模、可人工验证的任务，后批量实例。
- 先单角色稳定闭环，后双角色运行时。
- 先确定性驱动验证环境，后接入视觉语言模型。
- Agent 观察与 Evaluator 真值严格隔离。
- 所有模型动作必须结构化、受限、可中断和可回放。
- 阶段完成需要真实证据，不能只依据代码或单元测试。

## 固定技术栈

- Python 3.10.20
- OpenJDK 8.0.472
- MineRL 1.0.2 / Minecraft 1.16.5
- Gym 0.23.1 / NumPy 1.23.5
- Qwen3-VL-2B-Instruct 本地视觉基线
- `vendor/minerl` 为独立嵌套 Git 仓库

普通功能开发不得顺便升级依赖。需要升级时必须单独规划兼容性、回滚和复现验证。

---

## Phase 0 - Clean Core

**状态：完成。**

### 目标

移除旧洞穴任务的专用结构，建立不依赖具体 MineRL 任务的最小核心，使任务、
多角色观察、语义动作、评测真值和日志从第一天起具有稳定边界。

### 实现内容

1. 建立 `obsidianlink` Python 包和 `python -m obsidianlink` 入口。
2. 定义 `TaskInstance`、`Observation`、`MacroAction`、`BackendStep`。
3. 定义 `EnvironmentBackend` 协议，接口始终使用 `agent_id` 映射。
4. 实现不依赖 MineRL 的 `FakeEnvironmentBackend`。
5. 实现严格 JSON 动作解析、动作白名单、类型检查、数值限制和安全 `wait`。
6. 定义 `EvaluationState`、`EvaluationResult` 和初始 PortalEvaluator 规则。
7. 实现带身份字段的 JSONL 结构化事件记录器。
8. 定义语义工作流阶段和依赖检查。
9. 建立任务 JSON Schema 和 A0 示例实例。
10. 重写 README、路线图、Benchmark 规范、数据集卡和开发约束。

### 测试

- 核心数据对象拒绝空 ID、负 step 和非法库存数量。
- 动作解析拒绝未知字段、未知动作、错误类型和非法参数。
- 数值参数被限定在协议范围内。
- FakeBackend 可以 reset、step、close，并保持多角色映射形状。
- PortalEvaluator 不把已有地狱门误判为本回合成功。
- 工作流只在依赖满足时推进。
- `python -m obsidianlink --check` 输出可解析的检查结果。

### 退出条件

- `python -m unittest discover -s tests -v` 全部通过。
- `python -m obsidianlink --check` 成功。
- 新包中不存在 `cave_visible`、FindCave 环境 ID 或洞穴完成状态机。
- 项目文档清楚标记真实 MineRL 地狱门任务尚未实现。

### 2026-07-30 验证记录

- 21 个标准库单元测试通过；
- Phase 0 CLI 契约检查通过；
- 源码和测试语法编译通过；
- Benchmark/配置 JSON 文件解析通过；
- `git diff --check` 通过；
- 固定解释器 `/opt/anaconda3/envs/mc-agent/bin/python` 的版本为 Python
  3.10.20；
- 固定运行时版本核对通过：Gym 0.23.1、NumPy 1.23.5、Torch 2.13.0、
  Transformers 4.57.6。

### 本阶段不做

- 不启动真实 Minecraft。
- 不运行 Gradle。
- 不加载 Qwen 或调用远程 API。
- 不创建双角色客户端。
- 不声称任何地狱门任务已成功。

---

## Phase 1 - Portal Environment Feasibility

**状态：完成。Java 桥接、完整门框、激活和维度切换均有真实证据。**

### 目标

证明当前固定 MineRL/Malmo 技术栈能够承载地狱门任务所需的环境、资源、动作和
真值采集。在本阶段结束前，不假设现有 BASALT FindCave 环境可直接复用。

### 实现内容

1. 设计 `PortalEnvSpec`，不修改 `vendor/minerl` 历史。
2. 建立 A0 固定平地场景：
   - 单角色；
   - 固定出生点和朝向；
   - 预置足够黑曜石和打火石；
   - 禁止自然生成的附近地狱门干扰评分。
3. 验证以下 MineRL 能力：
   - 第一人称 POV；
   - 完整或允许公开的物品栏观察；
   - 装备物品；
   - 放置黑曜石；
   - 使用打火石；
   - 挖掘/破坏用于负例和修复；
   - 当前坐标与维度真值；
   - 方块状态或可用于评测的替代真值。
4. 实现 `MineRLEnvironmentBackend`：
   - `open/reset/step/get_evaluation_state/close`；
   - 单一环境所有者；
   - 返回 `dict[agent_id, Observation]`；
   - 环境 ID、种子、预算和初始资源来自配置。
5. 为低层 MineRL action space 建立显式能力报告。
6. 记录环境启动、重置和关闭的结构化事件。

### 验证顺序

1. 先生成或检查任务 XML，不启动 Minecraft。
2. 用 fake action space 验证动作翻译。
3. 获得用户明确批准后，运行一次短真实 reset/step/close。
4. 再运行人工 A0 能力检查，不接入模型。

### 退出条件

- 同一种子和配置可重复重置到等价 A0 状态。
- 人工可以装备、放置、点火并进入传送门。
- 环境能在异常和正常结束后自动关闭，无残留进程。
- 能明确读取或推导 PortalEvaluator 所需真值。
- 如果某项能力不可用，形成后端替代方案决策记录，而不是在上层写补丁绕过。

### 风险决策

若固定 MineRL 后端无法可靠提供方块/维度真值，应比较：

1. 自定义 MineRL EnvSpec/handler；
2. Malmo 任务 XML 与自定义监视器；
3. 独立多人服务器后端。

此决策只改变后端实现，不改变上层接口和 Benchmark 任务定义。

### 2026-07-30 实现与验证记录

已完成：

- `PortalA0EnvSpec`、`MineRLEnvironmentBackend` 和 MineRL action translator；
- A0 初始资源由 `TaskInstance.initial_inventories` 注入；
- XML 生成、动作翻译、生命周期、随机回退帧拒绝和 evaluator 隔离测试；
- 固定 `mc-agent` 环境中 38 个单元测试通过；
- 真实 MineRL 运行完成 14/14 tick，正常 reset、step、close；
- 640x360 POV 和公开物品栏正常；
- 黑曜石放置后数量从 10 变为 9；
- `use_item.obsidian=1`、`use_item.flint_and_steel=1`；
- 所有测试动作均通过 action space 校验，未执行模型代码或命令。

初始真实运行暴露了以下桥接缺口：

- 请求出生点 `(0, 64, 0)` 未生效，实际为
  `(-893.5, 63.0, -501.5)`；
- `FlatWorldGenerator` 未被当前 MineRL `EnvServer` 执行；
- `ObservationFromGrid` 未进入 info JSON；
- 当前 info JSON 没有 dimension。

根因是 MineRL 1.0.2 Java `EnvServer` 只转发有限观察字段，并固定创建默认世界。
这些缺口已由最小 MineRL 桥接扩展解决；详细范围和回退方案见
[`docs/decisions/0001-portal-environment-backend.md`](docs/decisions/0001-portal-environment-backend.md)。

补充验证结果：

- Gradle `compileJava` 和 `shadowJar` 均通过；
- 固定位置、25x25 平整平台、343-cell 方块 grid 和 overworld dimension 已回传；
- 真实放置后 `obsidian_added=1`，grid 出现一个黑曜石；
- 打火石使用后 grid 出现一个 fire；
- 38 个 Python 测试通过；
- 可复现 Java 补丁位于
  `patches/minerl/obsidianlink-envserver.patch`。

确定性 Scripted-A0 最终真实运行：

- 运行目录：
  `runs/history/phase1-scripted-a0/20260730-214356/`；
- 14 块黑曜石构成完整 4x5 外框，2 块泥土用于生存模式原地垫高；
- `max_obsidian_added=14`；
- `portal_activated_latched=true`；
- 打火石使用 1 次；
- 门内等待 84 tick 后，dimension 变为 `minecraft:the_nether`；
- 共完成 251 environment step，未提前终止；
- 正常关闭后无 Minecraft 或 Gradle 残留进程。

Phase 1 退出条件已满足。Scripted-A0 driver 作为后续 Phase 3 的预备实现保留，但
Phase 2 必须先完成独立门框几何评测、负例和可追溯里程碑，不能因为确定性运行通过
而跳过 Evaluator。

---

## Phase 2 - Portal Evaluator

**状态：完成。离线正负例、真实 attribution、typed portal-entry
correlation、自动评测和人工复核均已通过。**

### 目标

在任何 VLM Agent 参与前，建立可信、可测试、与决策逻辑隔离的自动评测器。

### 冻结规则

门框规则、尺寸、朝向、缺角、激活、attribution、portal-entry correlation
全部冻结在 [`docs/decisions/0002-portal-frame-rules.md`](docs/decisions/0002-portal-frame-rules.md)；
关键摘要：

- 外宽 W、外高 H：4 ≤ W ≤ 23，5 ≤ H ≤ 23。
- 两种水平朝向 `plane_z` / `plane_x`。
- 缺角允许：边框非角 cell 数为 `2W + 2H - 8`（vanilla 1.16.5 合法形态）。
- 内部必须只包含 `air` / `nether_portal` / `fire`；
  `dirt` / `bedrock` / `obsidian` / `grass` / `grass_block` / `other` /
  `missing` 都是阻挡。
- 激活：某 frame 候选内部出现至少 1 个 `nether_portal` cell，且该 frame 必须
  是本回合建造的。
- Attribution：每个 obsidian 偏移必须能匹配到一次允许 agent 的
  `place_block(obsidian)` 动作，否则归为 `external`。
- Portal-entry correlation：Nether 进入必须有显式 bridge transition
  evidence 指向 exact latched frame identity；interior 邻域检查仅是额外
  sanity check，不能单独证明因果。
- Termination：评测器只在 `episode_terminated=True` 时分类失败。

### 已落地的实现

1. 纯几何检测器 [`obsidianlink/evaluation/frame_geometry.py`](obsidianlink/evaluation/frame_geometry.py)：
   无 MineRL 依赖，输入 `(x, y, z)` 3D grid + baseline，返回
   `FrameDetectionResult`（五个互斥桶 + `has_missing_truth` 等真实聚合）。
2. **Observation-bound attribution**：backend 只为翻译成功的
   `place_block(obsidian)` 在当前 environment step 发放 credit。只有当前
   post-step observation 首次出现的 obsidian cell 数与 credit 数完全相等
   才归因；不相等时 fresh delta 全部 fail closed 为 external。未匹配 credit
   在 observation boundary 失效，且 external cell 永远不能重新进入
   attributed。`is_episode_built` 只有在
   `selected.required_frame_blocks ⊆ attributed_obsidian_offsets` 时为
   True。私有 `_credit_pending_place_block_for_test` 仅供直接改 grid 的
   离线 fixture 使用，生产 driver 不得调用。
3. **Partial 连续性**：`is_partial` 必须是"同一条边 ≥ 3 块 obsidian"或
   "共享 corner 的 L 形"（每条 incident 边各 ≥ 1 块 obsidian）。预存
   obsidian 不计入。所有缺角 obsidian 块必须先扣除 baseline 才是
   episode 新增。
4. `EvaluationState` 锁存 frame identity：一旦 `is_episode_built=True`
   触发，`latched_frame_identity` 保存完整几何证据。Nether grid
   替换后仍能输出正确 verdict。
5. 激活绑定到 latched frame：只有 latched frame identity 的 interior
   出现 `nether_portal` 才记 `portal_activated`。
6. **Portal-entry correlation**：`pre_transition_position_by_agent` 在
   dimension 从 overworld 变 nether 之前一刻锁存；只有 bridge 的 typed
   `portal_transition` 明确为 true、其 interior offsets 精确匹配 latched
   frame identity、且 pre-transition position 通过 world-anchor 邻域检查时
   才记录 True。证据明确拒绝时为 False；证据缺失/不完整时字段保持 unset，
   evaluator 输出 unknown。False/unknown 均不能成功。
7. 终止信号：MineRL `done=True` 或 `mark_terminated(step_id, reason)`
   设置 `episode_terminated=True`；只有此时 `failure_type` 才会输出。
8. 失败分类优先级：attribution_failed > frame_never_valid >
   portal_never_activated > no_agent_entered_nether >
   nether_entry_not_via_episode_portal / nether_entry_portal_unknown。
9. 里程碑事件使用 `StructuredEvent` 顶层契约，timestamp 锁存于首次
   观察；`__post_init__` 强制每个 milestone step 必须有对应
   `latched_timestamps` key。多 agent 用
   `agent_entered_nether:<agent_id>` 避免 timestamp 冲突。
10. `has_missing_truth` 是 detector 构造时基于 grid 整体 missing cell
    数计算的属性，不是 placeholder。
11. **外部结构检测（Round 3 新增）**：backend 计算
    `external_structure_candidate_count`：
    - 完全外部：candidate 的所有 required cells 全部在
      `external_obsidian_offsets`；
    - 混合：candidate 的 required cells 部分在 attributed、部分在 external
      （即 episode 没有独立完成门框）。
    两类都会被 `_derive_failure` 提升为
    `FAILURE_FRAME_NOT_BUILT_BY_EPISODE`，不再误判为
    `frame_never_valid`。`attribution_failed_candidate_count`（pre-existing
    frame）也走同一通路。

### 必测正例（已覆盖）

- 标准 4x5 `plane_z` 含/缺角；
- 标准 4x5 `plane_x`；
- 合法的较大 6x7 门框；
- `test_required_count_matches_documented_formula` 验证 2W+2H-8 公式；
- 完整路径：`test_full_path_with_termination_succeeds` 与
  `test_latched_frame_identity_survives_nether_grid_loss`；
- `test_entered_via_episode_portal_true_with_explicit_transition` 验证
  exact frame transition evidence + 邻近位置时 correlation 为 True。
- `test_at_spawn_grid_bounds_include_world_anchor` 验证 y=64 spawn 不会被
  错误映射到 y=0。

### 必测负例（已覆盖）

- 外部生成的完整门框：
  `test_external_full_frame_is_not_attributed`（attribution 队列空
  → 14 块全部 external → `portal_built_by_episode=False` →
  `frame_not_built_by_episode`）。
- 外部 dimension 切换：
  `test_external_dimension_switch_is_not_success`（agent 站远处
  → pre_trans 远离 frame → `entered_via_episode_portal=False` →
  `success=False`）。
- 走其他 portal 进入：`test_other_portal_entry_is_not_success`
  （已建 B 激活 B 但 agent 站 C 位置 → `success=False`）。
- 零散 obsidian 不触发 partial：
  `test_isolated_obsidian_blocks_do_not_form_a_frame` /
  `test_three_obsidian_on_different_edges_is_not_partial` /
  `test_l_shape_partial` / `test_pre_existing_frame_does_not_trigger_build_site_selected`。
- 内部阻挡：`test_dirt_inside_interior_blocks_frame` /
  `test_bedrock_inside_interior_blocks_frame` /
  `test_other_inside_interior_blocks_frame` /
  `test_missing_inside_interior_blocks_frame`。
- 缺角 + 错几何 + 缺必需 cell + 预存 frame / 预存已激活 portal。
- `has_missing_truth` 真实聚合：全 missing grid → True；
  frame/interior cell missing → True；正常 air grid → False。
- Timestamp 一致性：milestone step 缺 timestamp → `EvaluationState`
  构造失败；同一 state 多次 emission → timestamps 完全相同；
  多 agent nether 各自 timestamp 不串号。

### 审计后新增的回归测试（Round 3）

`tests/test_minerl_backend.py` 新增两个回归测试类，所有测试名以
`test_regression_*` 前缀方便 code review 时检索：

- `AttributionRegressionTests`：
  - `test_regression_external_full_frame_not_attributed` — 外部生成完整
    门框 → `external_structure_candidate_count >= 1` →
    `frame_not_built_by_episode`。
  - `test_regression_single_place_block_is_one_obsidian` — 单次
    `place_block(obsidian)` 只能归因一块，禁止把一次动作泛化为多块。
  - `test_regression_external_dimension_switch` — 远点站立 + 维度切换
    → `entered_via_episode_portal=False` → `success=False`。
  - `test_regression_other_portal_entry` — 建 B 激活 B 但 agent 站 C
    位置 → `success=False`。
  - `test_regression_three_non_contiguous_obsidian_not_partial` — 三块
    分布在不同边的 obsidian 不触发 `build_site_selected`。
  - `test_regression_missing_timestamp_rejected` — 缺失
    `latched_timestamps` 键时 `EvaluationState` 构造失败。
  - `test_regression_full_missing_grid_has_missing_truth` — 全 missing
    grid 暴露 `has_missing_truth_latched=True`。
  - `test_regression_external_cell_is_never_reattributed` — 已分类 external
    的旧 cell 永不重新归因。
  - `test_regression_unmatched_credit_expires_at_step_boundary` — no-op
    placement credit 不得污染后续 delta。
  - `test_regression_nearby_external_dimension_flip_is_unknown` — 门旁外部
    切维但无 typed transition evidence 时为 unknown、终止失败有明确类型。
- `FrameGeometryRegressionTests`：用纯几何 detector 重做上述 6 个审计
  场景，每条都对应 audit 编号。

### 退出条件

- 真实 MineRL 受控集成轨迹的回放证据（含每步 grid + per-action
  obsidian attribution + 精确维度切换 step + pre-transition position
  + explicit termination signal）：由
  `runs/phase2-scripted-a0/20260731-173302/` 满足。历史 Phase 1 replay
  仍保持 `status=insufficient_evidence`，未伪造升级。
- 单元测试套件必须全通过；`python -m obsidianlink --check` 通过；
  Python 语法、JSON、`git diff --check` 全部通过。
- 离线代码契约（attribution, portal-entry correlation, terminal
  failure, StructuredEvent, external structure）必须通过全部回归测试。

### 2026-07-31 验证记录

- Round 4 修复 old-external re-attribution、stale no-op credit、
  atSpawn world-anchor 和 nearby external dimension flip；
- 单元测试套件 121 / 121 通过；
- 经用户单次批准执行 `./gradlew compileJava`；ForgeGradle 在项目配置阶段
  从默认 PATH 检测到 Java 25.0.3，而固定工具链要求 Java 8，故构建在
  Java 源码编译前失败。未启动 Minecraft，也没有产生真实 MineRL 证据；
- 随后使用 `/opt/anaconda3/envs/mc-agent` 提供的固定 OpenJDK 8.0.472
  重新执行，`compileJava` 成功（5 个任务：4 executed、1 up-to-date）；
- `shadowJar` 使用相同固定 JDK 8 构建成功；
- canonical 真实运行 `runs/phase2-scripted-a0/20260731-173302/`：
  251 step、84 个 portal wait、14 个 attributed obsidian、0 external；
  valid frame step 148、activation step 162、typed transition 和 Nether
  entry step 251；
- 正式 `PortalEvaluator` 返回 `success=true`、
  `entered_via_episode_portal=true`、blocking conditions 为空；driver 在
  step 251 以 `scripted_a0_driver_complete` 显式终止；
- 251 条 action JSONL 全部包含 `episode_id`、`agent_id`、`step_id`；
  六个结构化里程碑齐全并按 step 排序；
- `manual_review.md` 接受该 run；`final.png` 可见 Nether 场景和
  “We Need to Go Deeper” 成就。视觉证据仅作一致性检查，attribution 与
  portal correlation 以 evaluator-only typed evidence 为准；
- 真实运行后无 Minecraft、MineRL 或 Gradle 残留进程。Phase 2 退出条件
  全部满足。

### 2026-07-31 bridge source 准备

- `patch_obsidianlink_phase2.py` 以幂等方式持久化生成 MCP tree 的改动，
  `patch_mcp.sh` 在基础 patch 后自动应用；
- `ObservationFromGrid` 同时返回 grid payload 与真实 world origin；
- `ServerPlayerEntity.changeDimension()` 仅在
  `Entity.updatePortal()` guard 生效且 source block 是
  `Blocks.NETHER_PORTAL` 时递增 transition sequence；
- EnvServer 输出 typed `portal_transition`：sequence、from/to dimension、
  source portal block world position；Python handler 缺失或类型错误时
  fail closed；
- Java 源码已完成静态/幂等 patch 检查，并使用固定 JDK 8 通过
  `compileJava` 与 `shadowJar`；canonical 真实运行已验证 origin、
  dimension 和 typed `portal_transition`。

### 2026-07-30 验证记录

- 30 个 `test_frame_geometry` 测试 + 21 个 `test_evaluation` 测试 +
  32 个 `test_minerl_backend` 测试（含 13 个新增 regression tests）+
  32 个其他；
- 单元测试套件 115 / 115 通过；
- `scripts/replay_run_for_evaluator.py` 报告 `status=insufficient_evidence`：
  历史 `runs/history/phase1-scripted-a0/20260730-214356/` 的
  `events.jsonl` + `summary.json` 不足以重建 per-step grid、obsidian
  attribution、pre-transition position 或 termination signal，新
  evaluator 不会伪造成功状态。
- 此记录形成时仍缺受控真实集成证据；该缺口已由 2026-07-31 canonical
  Phase 2 run 补齐。
- `vendor/minerl` 嵌套仓库状态：当前非 clean（多个 modified / untracked
  文件），与本轮 Phase 2 无关，外层 diff 未触及该路径。

---

## Phase 3 - Route A0 Vertical Slice

**状态：完成。**

### 目标

跑通第一条完整研究链路：材料齐全、固定平地、固定朝向，只负责建门、点火和进入。

### 实现内容

1. 冻结 A0 TaskInstance。
2. 实现低层动作执行器：
   - 有限角度看向；
   - 有限步数移动；
   - 装备指定物品；
   - 对当前目标方块放置；
   - 使用点火工具；
   - 每个宏动作有超时和可中断边界。
3. 先实现确定性 A0 driver，用于证明环境和评测器闭环。
4. 保存完整运行目录：
   - 配置快照；
   - 代码提交标识；
   - 初始/最终帧；
   - 决策帧；
   - 动作和环境事件 JSONL；
   - evaluator 事件；
   - summary。
5. 加入失败注入：
   - 放置失败；
   - 视角偏移；
   - 目标位置被占用；
   - 点火未生效。
6. 错误必须有限重试或明确终止，不允许无限循环。

### 2026-07-31 启动记录

- 冻结 `benchmark/instances/route_a_a0_phase3.json` 与
  `configs/experiments/phase3_scripted_a0.json`：单角色、seed 0、固定出生点、
  14 块黑曜石、1 个打火石、2 块泥土，以及 320 step 上限；
- Scripted-A0 增加四个一次性、确定性、可记录的负例入口：
  `placement_failure`、`view_offset`、`target_occupied`、`ignition_no_effect`。
  注入只会替换为 allowlist 内宏动作或改变受限的 look delta，绝不读取
  evaluator-only truth，也不生成低层输入；
- 基线配置的自动 placement retry 固定为 0：真实 MineRL 物品栏可能在放置后的
  同一观察尚未反映扣减，不能据此推断放置失败。placement retry 只允许作为显式
  负例实验参数；点火未生效会在 portal wait 预算耗尽后明确返回 `blocked`，不允许
  无限循环。
- Scripted-A0 为每一个 `backend.step` 设定 30 秒主线程 deadline；超时抛出并
  记录 `TimeoutError`，以便把 EnvServer 通信卡住与动作/评测失败区分开。
- 后端 reset 对握手/transport 的 `OSError`、`RuntimeError`、`TypeError` 最多
  重建一次 MineRL 环境（共 2 次尝试），随后带原始异常链失败；它不修改
  `vendor/minerl`，也不会无限重启 Minecraft。
- 每次 Scripted-A0 运行会在独立目录写入冻结的任务和实验配置、commit/启动参数、
  初始帧、所有非 wait 动作后的 agent-visible 决策帧、最终帧、动作 JSONL、
  evaluator JSONL 与 summary；帧归档接口只接收 `Observation`，不会接收
  `EvaluationState`。
- `WorkflowA0Policy` 与 `DirectA0Policy` 已定义为纯 agent-visible prompt
  适配器，二者都经严格 JSON action parser fail closed 到 `wait`；相应 Qwen
  配置已冻结，但遵守 evaluator-first 顺序，必须等 Scripted-A0 真实基线稳定后
  才能加载或运行模型。
- `AsyncA0PolicyWorker` 使用容量为 1 的 request/decision mailbox；环境 owner
  只 submit/poll，永不等待模型推理，过期或满队列结果会被丢弃而非复用到其他
  observation。
- `LocalQwenResponder` 使用 `model.lock.json` 中已锁定的本地 Qwen3-VL 模型，
  延迟加载到 policy worker；禁止联网下载、远程 API 与模型生成代码。它最多生成
  64 token，输出仍需经过 JSON action parser。运行时自动优先 MPS，MPS 不可用时
  才回退 CPU；不会假设编译进 PyTorch 的 MPS 后端等同于设备可用。
- `scripts/run_vlm_a0.py` 固定使用一条 Phase 3 VLM 实验配置和最多一次模型调用。
  环境 owner 在推理期间按固定节奏执行安全 `wait`，只接受相同 episode/agent 且在
  明确 step-age 上限内的决策；过期、拒绝和 worker 异常均单独计数或带 traceback
  归档，不能影响环境 step。
- `MiniMaxM3Responder` 通过官方 Anthropic-compatible Messages API 提供远程视觉
  planner 选项：仅上传 agent-visible JPEG 和公开提示，禁用 thinking、工具和自动
  重试；`MINIMAX_API_KEY` 仅从进程环境读取，响应摘要只记录 request ID、token 用量、
  延迟和解析结果。`phase3_minimax_m3_workflow_a0` 冻结为 standard 服务档、96 输出
  token 和最多 1 次调用，先验证单帧契约后才允许完整受控运行。
- `scripts/probe_minimax_m3.py` 使用已经验收的 Scripted-A0 初始帧，且必须显式传入
  `--allow-live-request` 才会发送唯一一笔 API 请求。它将 HTTP 成功与动作 JSON
  合规分别报告；请求错误、解析错误均不是 portal 任务失败，也不会启动 Minecraft。
  Token Plan 的 `sk-cp-...` Subscription Key 在 Anthropic-compatible endpoint 使用
  `Authorization: Bearer` 认证；不可误用标准 Anthropic 的 `x-api-key` 头。
- Canonical Scripted-A0 真实运行
  `runs/phase3-scripted-a0/20260731-210140/` 已通过：251 step、14 块
  attributed obsidian、0 external structure、valid frame step 148、activation
  step 162、typed Nether entry step 251、正式 evaluator `success=true`；
  `manual_review.md` 已接受该确定性基线。此前的 socket/timeout 运行均仍为
  非 canonical 诊断产物。
- `scripts/run_scripted_a0.py` 在导入 MineRL 前固定子进程的 `JAVA_HOME` 与
  `PATH` 为锁定的 `/opt/anaconda3/envs/mc-agent` OpenJDK 8，避免 PATH 中的
  非兼容 Java 启动 EnvServer 并缺失 JAXB。
- 本阶段的真实闭环尚未验收：`runs/phase3-vlm-a0/20260731-210927/` 在受限执行
  环境中因本机端口绑定被拒而正常写出失败摘要；获得本机端口权限后的运行可成功
  reset 并归档 `initial.png`，但在模型与 MineRL 同时常驻时由底层进程提前终止，未
  进入 Python 的失败处理，故没有模型动作、里程碑评分或可接受的负例结论。
  隔离预检 `runs/phase3-vlm-a0-preflight/20260731-211357/` 已以 exit code 0 验证
  锁定 Qwen 可加载到 MPS；使用预加载路径的真实受控重跑仍提前结束。由此可排除
  模型缺失、MPS 不可用与端口权限，当前证据指向模型和 MineRL 组合进程的资源或
  运行时冲突。这些仅是设备运行时诊断，不作为 Phase 3 的 VLM 通过或失败证据。
  下一步是把模型推理隔离为可监督的子进程（父进程记录子进程 exit code、峰值资源
  和请求/响应时间），再以该边界执行一次受控 VLM 运行。

### 2026-08-03 Qwen 负 stride 修复 + 受控 VLM A0 真实运行

- 修复 `obsidianlink/agents/local_qwen.py` 的 numpy 负 stride bug：
  抽出 `_prepare_frame` 静态方法，在 PIL 衍生视图（flip / crop 等
  会带负 stride）上调用 `np.ascontiguousarray` 归一化为 C-contiguous
  数组，再传给 `apply_chat_template` 和 processor；之前
  Qwen `_process_image` 直接 `torch.from_numpy(image).contiguous()` 会
  抛 `ValueError: tensors with negative strides are not currently
  supported`。
- 新增 `tests/test_local_qwen.py`（4 个回归测试）：
  flip 视图、transpose 视图、已 contiguous 帧恒等返回、非 ndarray 拒绝；
  测试名带 `test_regression_` 前缀方便 review 检索。
- 单元测试套件 140 / 140 通过（之前 121 + 4 新增 + 既有累计），0 回归；
  `python -m obsidianlink --check` 仍然通过。
- 受控 VLM A0 真实运行 `runs/phase3-vlm-a0/20260803-215738/`（**先于本轮
  instrumentation；该 run 的 `code_version.json` 仍指向修复前的提交
  40e84e8，未记录 dirty 标记、未记录推理延迟、未记录 model_requests.jsonl，
  不应作为 Phase 3 最终证据**）：
  - workflow mode + local_qwen + MPS；预算 320 step、
    `--min-step-interval-seconds 0.25`、`--max-decision-age-steps 160`；
  - Qwen 预加载成功、MineRL reset 成功，进程共驻稳定；episode 墙钟约 81s；
    跑完后无 Minecraft / MineRL / Gradle 残留进程；
  - 320 step 全跑完，`events.jsonl` 320 条结构化事件齐全；evaluator
    写出 `formal_evaluation`；终止信号 `vlm_a0_budget_complete` @ step 320；
  - `model_calls_submitted=1`、`decisions_applied=0`、
    `decisions_dropped_stale=1`：Qwen 决策到达 owner 时已超过
    `max-decision-age-steps=160` 窗口，被丢弃为 stale；mailbox 容量 1
    期间没有新提交；所有 320 个 action 都是 `wait`（`action_source_step:
    null`）；
  - **该 run 没有 owner 端推理延迟埋点**（instrumentation 是 2026-08-03
    本轮新加的），所以"延迟约 80s"是先前的非直接表述；本轮后
    `model_requests.jsonl` 才有真正的 `responder_started_at_monotonic` /
    `responder_completed_at_monotonic` / `responder_latency_seconds` 可供
    复核；
  - `failure_type=frame_never_valid`、`last_successful_milestone=task_reset`、
    `success=false`；脚本语义下 `status=blocked`、exit code 2（不是
    transport 不可用，是 VLM 未推动 episode 进展）。
- 本轮新增的 VLM runner instrumentation（`scripts/run_vlm_a0.py` +
  `obsidianlink/agents/local_qwen.py`）：
  - `LocalQwenResponder` 暴露 `QwenRequestRecord`（`started_at_monotonic`、
    `completed_at_monotonic`、`latency_seconds`、`device`），不记录 prompt
    文本、模型输出、API key、evaluator 状态；
  - `run_vlm_a0.py` 在 owner 端每次 poll 写出
    `runs/<dir>/model_requests.jsonl`，每行一条
    `{source_step, return_step, decision_age_steps,
    max_decision_age_steps, drop_reason, decision_accepted,
    decision_error, responder_started_at_monotonic,
    responder_completed_at_monotonic, responder_latency_seconds,
    responder_device}`，并在 budget 结束后再做一次 final flush 避免
    残留决策被静默丢弃；
  - `code_version.json` 新增 `working_tree_dirty`（bool）与
    `dirty_paths`（list）；`summary.json` 同步暴露
    `working_tree_dirty`、`working_tree_dirty_paths`、`code_commit`、
    `reproducible_from_clean_commit`，dirty 时一律不声称完全可复现。
- 本次 fix 解决了 2026-07-31 报告中"模型与 MineRL 同时常驻由底层进程
  提前终止"的部分根因（之前根本到不了负 stride 这一层），但跑通后
  暴露新的真实诊断：在 0.25s step 节奏和 capacity-1 mailbox 之下，
  Qwen3-VL-2B 在 MPS 上的单次推理延迟 ≫ episode 预算，单帧单调用
  契约无法影响 episode 走向；这属于受控运行的设计边界，不是新 bug。
  本轮无法从 20260803-215738 跑里直接给出推理延迟数字，**实测推理
  延迟以干净提交下的新一轮 VLM run 的 `model_requests.jsonl` 为准**。
### 2026-08-03 Phase 3 VLM close-out (干净提交)

- 本地提交 `280ec920df963522355335137a57f0e2083c6fcd`（branch
  `main`，未 push）；工作区在跑前为 clean（`git status --porcelain`
  输出空）；`code_version.json.working_tree_dirty = false`、
  `summary.json.reproducible_from_clean_commit = true`。
- 受控 VLM A0 真实运行 `runs/phase3-vlm-a0/20260803-222729/`：
  - 同一 workflow / local_qwen / MPS 配置，`min-step-interval 0.25s`、
    `max-decision-age-steps 160`、预算 320 step；
  - Qwen3-VL-2B-Instruct 在 MPS 上的**实测推理延迟**
    `responder_latency_seconds = 41.47003112499806`（owner 端
    `started_at_monotonic = 34646.38780725` →
    `completed_at_monotonic = 34687.857838375`），设备 `mps`；
  - 模型决策到达 owner 时 `source_step = 0`、`return_step = 164`、
    `decision_age_steps = 164 > 160`，
    `drop_reason = "stale_age_exceeded"`；额外附带
    `decision_error = "Expecting value: line 1 column 1 (char 0)"`，
    即模型输出为空文本，即便没有超龄也会被 parser fail-closed 到
    `wait`；
  - `decisions_applied = 0`、`decisions_dropped_stale = 1`、
    `decisions_rejected = 0`；所有 320 个 action 都是
    `wait`（`action_source_step: null`）；
  - 终止信号 `vlm_a0_budget_complete` @ step 320；
    `formal_evaluation.failure_type = "frame_never_valid"`、
    `last_successful_milestone = "task_reset"`、`success = false`；
  - 跑完无 Minecraft / MineRL / Gradle / run_vlm 残留进程；
  - 人工 `manual_review.md`（同目录）记录运行 / 安全 / 模型应用 /
    任务完成 / failure 原因四类结论，并接受此 run 作为 Phase 3
    VLM close-out 证据。

### 退出条件（全部满足）

- ✅ Scripted-A0 在固定配置稳定完成
  （`runs/phase3-scripted-a0/20260731-210140/`，`success=true`、
  251 step、evaluator 闭环、人工 review 已接受）；
- ✅ 至少一个 VLM 配置产生可诊断的里程碑失败
  （`runs/phase3-vlm-a0/20260803-222729/`，来自干净提交
  `280ec92`，`failure_type=frame_never_valid`、
  `last_successful_milestone=task_reset`、`model_requests.jsonl`
  提供 `responder_latency_seconds=41.47` 真实测量）；
- ✅ Agent 不读 evaluator-only 状态（单元测试 + 代码契约守护）；
- ✅ 失败有明确类型和最后有效里程碑（`formal_evaluation` 完整
  记录，`blocking_conditions` 与 `failure_type` 一致）；
- ✅ 可从配置与代码版本复现
  （`code_version.json.working_tree_dirty=false`、
  `summary.json.reproducible_from_clean_commit=true`）。

### Phase 3 关闭记录

- 收尾证据见
  `runs/phase3-vlm-a0/20260803-222729/{summary,events,evaluator_events,
  code_version,model_requests}.jsonl` + `initial.png` + `final.png` +
  `manual_review.md`。
- 旧 run `runs/phase3-vlm-a0/20260803-215738/` 保留为
  pre-instrumentation 诊断产物；其 `manual_review.md` 也已落地，
  明确它**不是** Phase 3 close-out 证据。
- 用户 2026-08-03 决定暂不开展：
  - 模型推理隔离为可监督子进程（exit code / 峰值资源 / 请求响应
    时间）；
  - mailbox 调优（提高 `max-decision-age-steps`、允许多发）；
  - 切换到 MiniMax-M3 远程 planner
    （`scripts/probe_minimax_m3.py --allow-live-request` + 已冻结
    的 `phase3_minimax_m3_workflow_a0` 实验配置）；
- Phase 4 在 Phase 3 正式关闭后，随用户后续推进要求启动。

### 模型接入顺序

1. `Scripted-A0`：确定性环境基线。
2. `Single-Workflow-A0`：模型接收当前语义阶段。
3. `Single-Direct-A0`：模型只接收任务与观察。

模型接入必须保留相同任务、动作限制和 evaluator。

### 退出条件

- Scripted-A0 能在固定配置上稳定完成。
- 至少一个 VLM 配置能完成完整闭环，或产生可诊断的里程碑失败。
- Agent 不能读取 evaluator-only 状态。
- 所有失败都有明确类型和最后有效里程碑。
- 运行可以从配置和代码版本复现。

---

## Phase 4 - Route A Single-Agent

**状态：进行中。Phase 3 前置条件已满足。**

### 目标

从“材料齐全只建门”逐步扩展到附近黑曜石采集和有限资源补全，形成第一个稳定的
单智能体长程基线。

### 2026-08-03 启动记录

- 冻结 `benchmark/instances/route_a_a1_phase4.json` 作为 A1 第一个开发实例：
  单角色、seed 0、固定出生点和朝向、初始无黑曜石、已有钻石镐与打火石；
- `scenario_parameters` 明确附近矿源至少 14 块、最大距离 8 blocks、固定且可达，
  建造点保持固定平地，避免在第一条 A1 纵向切片中同时引入资源、朝向和地形变化；
- A1 新增 `obsidian_source_located`、`first_obsidian_mined`、
  `obsidian_quota_collected` 里程碑，再复用既有建造、激活和下界进入里程碑；
- 当前只完成任务契约与离线守护，不声称真实环境已有矿源、`mine_target` 已贯通或
  A1 可完成。下一步按 evaluator-first 顺序实现固定矿源环境真值、确定性采集 driver
  和自动里程碑，再申请一次真实 MineRL 验证。

### 子阶段

#### A0 - Build Only

- 已有黑曜石与打火石；
- 固定平地、固定朝向；
- 建造、激活、进入。

#### A1 - Nearby Obsidian

- 已有钻石镐和打火石；
- 黑曜石位于限定、可到达区域；
- 定位、采集足量、返回建造。

#### A2 - Ignition Resource

- 黑曜石与钻石镐可用；
- 缺少或需要确认点火工具；
- 只允许有限、预定义的点火资源补全。

#### A3 - Controlled Variations

- 随机初始朝向；
- 随机物品栏槽位；
- 小范围建造位置变化；
- 轻微平坦地形变化。

#### A4 - Recoverable Error

- 注入一次已知、可恢复的建造错误；
- 测量检测、修复和重规划能力。

### 关键模块

- 黑曜石目标识别和可达性描述；
- 数量和物品栏状态管理；
- 受限 `mine_target`；
- 建造场地选择；
- 门框空间工作流；
- 阶段记忆和有限反思；
- 防重复动作和无进展终止。

### 基线

- `Single-Direct`
- `Single-Workflow`
- `Single-Reflection`
- `Single-Knowledge`（仅在前三者稳定后可选）

### 退出条件

- A0-A4 每层至少有经过人工验证的任务实例。
- 每个基线报告成功率、里程碑完成率、时间、动作、调用和失败类型。
- 增加难度时只改变已登记参数，不暗中改变 Prompt、动作或评分。
- 不把额外模型调用直接解释为方法改进。

---

## Phase 5 - Route B Single-Agent

**状态：计划中。依赖 Phase 4 的核心运行与评测稳定。**

### 目标

在强语义工作流下完成水火浇筑，研究高精度空间操作、逐步验证和错误恢复。

### 子阶段

#### B0 - Fixed Casting

- 固定平地、完整水/岩浆/桶/辅助方块；
- 固定位置和朝向；
- 强工作流逐块浇筑。

#### B1 - Random Orientation

- 随机初始朝向；
- 工作流保持不变。

#### B2 - Random Inventory Slots

- 随机物品栏位置；
- 验证装备和资源追踪。

#### B3 - Nearby Sources

- 水源和岩浆源位于附近受控位置；
- 加入有限资源往返。

#### B4 - Terrain Variation

- 轻微地形和放置面变化；
- 不加入开放世界搜索。

#### B5 - Recoverable Casting Error

- 注入一次错误水流、岩浆位置或辅助方块错误；
- 要求检测、局部修复或安全终止。

### 工作流里程碑

- `liquid_resources_ready`
- `casting_site_selected`
- `support_structure_ready`
- `first_obsidian_cast`
- `portal_segment_completed`
- `valid_portal_frame`
- `ignition_tool_ready`
- `portal_activated`
- `agent_entered_nether`

### 退出条件

- 每次浇筑都有目标位置、动作、前后证据和环境结果。
- 流体错误不会进入无限循环。
- 失败能定位到具体区段和工作流阶段。
- Route A/B 在相同指标定义下可比较。

---

## Phase 6 - Benchmark Alpha

**状态：计划中。依赖 Route A/B 核心场景。**

### 目标

冻结第一版任务定义、实例格式、数据划分、评测器和实验运行器。

### 开发策略

不直接生成规划上限的 60 个实例。先完成一个小规模 Alpha：

- Route A：3 个单智能体模板；
- Route B：3 个单智能体模板；
- 每个模板 1-2 个经人工验证的实例。

Alpha 全部通过可完成性与评分检查后，再扩展到：

- Route A：6 个单智能体 + 4 个多智能体模板；
- Route B：6 个单智能体 + 4 个多智能体模板；
- 每个模板 3 个环境变体；
- 合计 60 个实例。

### 实现内容

- TaskInstance JSON Schema；
- workflow 与依赖图格式；
- train/dev/test 或 development/test 划分策略；
- 批量运行器和失败恢复；
- 配置、Prompt、模型、动作协议版本快照；
- 数据集版本和变更日志；
- 人工可完成性审查表；
- 指标汇总和置信区间脚本。

### 退出条件

- Alpha 每个实例都有人工或确定性可完成证明。
- 任务定义、预算和 evaluator 在正式比较前冻结。
- 失败运行不会污染或覆盖其他结果。
- 相同实例可由多个基线使用，不包含方法专用捷径。

---

## Phase 7 - Multi-Agent Core

**状态：计划中。依赖 Phase 6。**

### 目标

证明两个角色可以在同一世界中稳定运行，各自拥有独立观察和动作，并完成通信、
汇合与物品交接。先用确定性策略，不立即接入两个大模型。

### 实现内容

1. 验证 MineRL `_MultiAgentEnv` 与自定义 EnvSpec 的实际可用性。
2. 两个 Minecraft 客户端连接同一世界。
3. 每个角色拥有：
   - 独立 POV；
   - 自身物品栏；
   - 独立动作通道；
   - 私有记忆；
   - 独立模型调用生命周期。
4. 建立统一调度器，单一世界 tick 接收 `dict[agent_id, MacroAction]`。
5. 建立 Message Bus：
   - 消息发送者、接收者、step、Token；
   - 禁止共享未发送的私有观察。
6. 建立 Shared Blackboard：
   - 已分配子目标；
   - 已完成里程碑；
   - 公开资源与集合点；
   - 冲突和等待状态。
7. 实现物品交接事件与自动评分。
8. 记录每个角色贡献和空闲时间。

### 资源门槛

- 评估同时运行两个客户端和本地模型的内存/GPU需求。
- 必要时使用共享模型服务或轮流推理，但环境 tick 不得等待推理。
- 任何后端替换必须保持 `EnvironmentBackend` 契约。

### 退出条件

- 两个确定性角色连续完成多次 reset/step/close。
- 双方观察、库存和动作不会串号。
- 物品交接有环境真值证明。
- 一个角色异常不会造成不可清理的残留世界。

---

## Phase 8 - Route A Multi-Agent

**状态：计划中。依赖 Phase 7。**

### 目标

比较单角色、纯并行、自由通信和结构化协作在黑曜石采集路线上的收益。

### 对照组

- `Single`
- `Dual-NoComm`
- `Dual-Chat`
- `Dual-Workflow`

### 固定分工

- Agent 1：定位和采集黑曜石。
- Agent 2：获取/确认点火工具，准备建造区域。
- 共同：发送坐标、汇合、交接资源、建造、激活和进入。

### 公平性

- 相同任务实例、种子、动作空间和成功条件；
- 同时报告总调用、Token、费用和端到端延迟；
- 加入匹配推理预算的单智能体反思基线；
- Dual-NoComm 不得读取对方私有状态。

### 核心分析

- `Parallelism Gain = SR(Dual-NoComm) - SR(Single)`
- `Communication Gain = SR(Dual-Chat) - SR(Dual-NoComm)`
- `Structured Collaboration Gain = SR(Dual-Workflow) - SR(Dual-NoComm)`
- 资源交接成功率、冲突率、空闲率和重复劳动。

### 退出条件

- 能区分并行收益和通信收益。
- 能把每个里程碑归因到角色或团队事件。
- 协作方法的额外计算成本被完整报告。

---

## Phase 9 - Route B Multi-Agent

**状态：计划中。依赖 Phase 8 与稳定 Route B。**

### 目标

研究水火浇筑中的固定职责、精确同步、相互干扰和局部错误恢复。

### 固定分工

- Agent 1：管理岩浆并执行岩浆放置。
- Agent 2：管理水、辅助方块和结构检查。
- 共同：同步区段状态、修复、点火和进入。

### 关键问题

- 消息延迟是否导致错误浇筑；
- 两个角色是否互相阻挡或破坏结构；
- 结构化任务板是否减少重复检查；
- 一个角色失败时是否能有限接管；
- 通信成本是否超过节省的环境时间。

### 退出条件

- Single 与三种 Dual 方法使用相同 Route B 实例和 evaluator。
- 每个同步点、冲突和恢复都可从日志重放。
- 报告成功率和里程碑，而不是只报告最终成功。

---

## Phase 10 - Formal Experiments and Release

**状态：计划中。依赖所有基线冻结。**

### 目标

完成可复现的正式实验、数据集、代码发布和论文。

### 实验冻结

- 固定代码提交；
- 固定 MineRL/Minecraft/JDK/Python；
- 固定模型标识、模型提交、API 版本和区域；
- 固定 Prompt、动作协议和图像预处理；
- 固定任务实例、划分、预算和 evaluator；
- 预登记主要对照、重复次数和统计方法。

### 正式指标

- Success Rate；
- Milestone Completion Rate；
- Time to Portal；
- Action Efficiency；
- Recovery Rate；
- Inference Cost；
- Communication Cost；
- Handoff Success；
- Coordination Conflict；
- Idle Rate。

不把所有指标压缩成一个难以解释的综合分数。使用分项表、置信区间、
性能-成本散点图和 Pareto 分析。

### 发布内容

- 任务模板、实例和数据划分；
- 工作流、依赖图和里程碑；
- 自动评测器与复现实验脚本；
- 观察帧、结构化动作、消息和环境变化；
- 决策摘要，不公开隐藏推理链；
- 数据集卡、许可、已知偏差和版本变更；
- 论文表格、图和失败分析所需脚本。

### 最终退出条件

- 发布产物可以从干净环境按文档复现。
- 报告所有排除项、失败运行和人工审查规则。
- 不使用不同 Prompt、任务集或安全策略的运行声称“最佳模型”。

---

## 跨阶段失败分类

- `PERCEPTION_ERROR`
- `RESOURCE_SEARCH_FAILURE`
- `RESOURCE_SHORTAGE`
- `PLANNING_ERROR`
- `WORKFLOW_STAGE_ERROR`
- `NAVIGATION_FAILURE`
- `BLOCK_PLACEMENT_ERROR`
- `PORTAL_STRUCTURE_ERROR`
- `IGNITION_FAILURE`
- `STATE_MEMORY_ERROR`
- `REPEATED_ACTION_LOOP`
- `COMMUNICATION_FAILURE`
- `RESOURCE_HANDOFF_FAILURE`
- `COORDINATION_CONFLICT`
- `TIMEOUT`
- `SYSTEM_ERROR`

新增失败类型必须有明确判断条件，不得把未知错误随意归入感知或规划。

## 阶段验收模板

每个阶段交付时必须回答：

1. 实现了什么，未实现什么？
2. 哪些测试通过？
3. 运行了哪些真实环境验证？
4. 自动评分与人工审查是否一致？
5. 证据目录和配置快照在哪里？
6. 是否修改了固定依赖、动作边界或 evaluator？
7. 剩余风险会阻止下一阶段吗？

只有七项都有明确答案，阶段状态才可以标记为完成。
