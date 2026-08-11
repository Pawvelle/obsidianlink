# ObsidianLink 路线图

路线图把长期 Benchmark scope 与当前工程阶段分开。每个阶段先冻结合同，再用 FakeBackend 和确定性流程验证；真实 MineRL、模型和更复杂场景必须在对应阶段获得授权后推进。

## Foundation

### R1：冻结真实任务契约（完成）

活动任务 `casting_c1_fixed`：固定场景、Single-Agent、用水和熔岩生成一块黑曜石。任务 seed、资源、目标 cell、里程碑和预算已冻结。

### R2：后端能力清单（完成）

离线定义不可变 `BackendCapabilities`，覆盖水桶/熔岩桶选择与使用、公开库存/手持物品、目标方块 truth 和流体 truth。缺关键能力时 reset 前 fail closed；FakeBackend 正反例、序列化与离线测试已完成。

### R3：单块黑曜石 evaluator（完成）

为 `casting_c1_fixed` 实现独立 `CastingEvaluationState` 和 evaluator：区分 `success`、`wrong_block`、`truth_missing`、预算、初态、因果和终止问题。Evaluator 只读 evaluator-only truth，不读取 Agent 文本、图像或 Planner 输入，也不修改世界。

### R4：确定性单块 driver（完成）

使用公共 `MacroAction` 协议和封闭白名单，在 FakeBackend 上以有限动作、有限等待和有限恢复完成单块浇筑。Driver 不读取 evaluator-only truth；离线重放、隔离和失败路径测试已完成。

### B0：Benchmark Scope Freeze（完成）

B0 冻结：

- 统一“进入 Nether”的端到端 Benchmark 目标；
- Casting、Ruined Portal、Adaptive Routing 三个任务族；
- Single-Agent、Multi-Agent 两种正交模式；
- family/mode/level/layout 文档命名规范；
- 通用、Adaptive 和 Multi-Agent 指标体系；
- README、BENCHMARK_SPEC、ROADMAP、DATASET_CARD、PROJECT_STATUS 与 taxonomy 的一致范围；
- B0 完成时 active implementation 保持 `casting_c1_fixed`，不改文件 ID 或 schema。

B0 不实现新环境、evaluator、driver、planner 或通信逻辑。完成后下一工程任务仍是 `R5-CONTINUOUS-CASTING`。

## Suite A — Casting

### R5：连续浇筑（完成）

`casting_c3_fixed` 已把 C1 扩展为三个有序目标黑曜石，并完成多次流体操作、per-cell 因果证据、部分完成、有限恢复和 FakeBackend 确定性证明。按 B0 taxonomy，该旧兼容 ID 属于 Casting-S-C2 / fixed；`c3` 表示三个 cell，不是能力层级 C3。

### B1：Task Catalog Foundation（完成）

建立严格、只读的任务 catalog，统一 canonical taxonomy、历史兼容 ID、实例/实验路径、实现状态、发布可见性和 live-run policy。现有 C1/C2 保持原路径；`route_a_a0` 明确归为 calibration/regression；CLI 和环境检查从 catalog 解析当前任务。B1 不移动历史文件、不修改 evaluator/driver/backend，也不实现 R6。

### R6-COMPLETE-PORTAL-FRAME — CONTRACT FREEZE（合同冻结完成；C3/C4 与 C5 evaluator 已离线验证）

R6 阶段按 B0 taxonomy 冻结 3 个递进合同，但本轮**只冻结合同**：

- Casting-S-C3：使用水、熔岩和原版 block update 浇筑公开固定 4×5 full ring；原版最小合法 10-block 规则与本实例额外要求的 14-block full-ring constraint 分开记录；
- Casting-S-C4：在 C3 之上通过合法 `use_item(flint_and_steel)` 在唯一公开目标 `[1,1,1]` 点火；
- Casting-S-C5：在 C4 之上由公开指定的 `agent_1` 通过本 episode 门框进入 Nether，且机器合同要求完整 frame-identity 与 transition 归因。

合同冻结阶段交付：3 个 `benchmark/instances/casting/single/*.json` 实例、3 个 contract-only 实验配置、catalog 3 个新条目（`implementation_status=contract_only`、`benchmark_visible=true`、`live_run_allowed=false`）、3 个任务文档与离线合同测试。R6 合同冻结阶段**没有**实现任何 evaluator、driver、真实 MineRL 接入、Gradle 构建或模型 API 调用；`active_compatibility_id` 保持 `casting_c3_fixed`（C2）。

#### R6-C3-FRAME-EVALUATOR（C3 离线 frame evaluator + task-origin / truth-grid 坐标锚定）

本子阶段在 R6 合同冻结基础上只前进 C3 frame evaluator：

- `obsidianlink/evaluation/casting_frame_evaluator.py` 提供不可变、类型严格、可序列化的 `FrozenFrameCellTruth` / `FrozenFrameInteriorCellTruth` / `FrozenFrameEvaluationState` / `FrozenFrameEvaluationResult` / `FrozenFrameEvaluator`；
- 闭集 outcome 与 R3 / R5 共享并新增 `interior_blocked`；
- `FrozenFrameOriginAnchor` 是纯函数、不可变、类型严格的 task-origin → truth-grid 转换器；`default_c3_anchor()` 把任务原点对齐到 `obsidianlink.env.portal_spec.PORTAL_GRID_MIN/MAX` 范围内的 grid 原点；
- `FakeEnvironmentBackend` 新增 `_frame_evaluation_state` 槽位 + `set_frame_evaluation_state` / `get_frame_evaluation_state` / `clear_frame_evaluation_state`，严格身份校验，`reset` / `step` / `close` 清空陈旧 truth；
- 信息隔离：evaluator 源文件 AST 检查不 import `agents` / `workflows` / `drivers` 也不读取 `scenario_parameters` / `evaluator_contract` / `instruction`；FakeBackend `Observation` 不携带任何 frame / cell / outcome / 归因 truth；
- C3 frame evaluator 103 个专项测试通过；C1 / C2 / portal 旧测试全部回归通过；全量 539 个离线测试通过。

R6-C3-FRAME-EVALUATOR 子阶段本身没有实现 driver；随后的 `R6-C3-DETERMINISTIC-DRIVER` 已完成严格 public context、capability gate、336-step bounded plan、有限恢复、driver/evaluator 隔离和 FakeBackend 离线编排。`active_compatibility_id` 仍为 `casting_c3_fixed`（C2），C3 仍未接入正式 experiment runner 或真实 MineRL。

#### R6-C3-DETERMINISTIC-DRIVER（完成，FakeBackend 离线证明）

- `obsidianlink/drivers/casting_s_c3_frame.py` 实现固定 14-cell、336-step deterministic plan，只使用 `equip_item` / `use_item` / `place_block` / `wait`；
- `obsidianlink/core/casting_s_c3_frame_context.py` 从公开任务合同构造严格、不可变 context，原始类型错误不被强制转换掩盖；
- `casting_s_c3_fixed` 纳入 reset 前 capability gate，缺少桶动作、公开库存或 evaluator truth 能力时 fail closed；
- driver 不读取 evaluator truth，测试 orchestrator 独立注入 `FrozenFrameEvaluationState`，最终 success 只由 evaluator 判定；
- 下一子任务是 `R6-C4-IGNITION-EVALUATOR`；C4 driver 必须等 evaluator 离线完成后再启动。

#### R6-C4-IGNITION-EVALUATOR（完成，FakeBackend 离线证明）

- `obsidianlink/evaluation/casting_ignition_evaluator.py` 实现 C4 ignition evaluator + typed `FrozenFrameIdentity` + 分层构造合同 + 4-step 因果窗口；
- `FrozenFrameIdentity` 是 frozen dataclass，13 个显式字段（`orientation` / `min_corner` / `max_corner` / `width` / `height` / `target_offsets` / `interior_offsets` / `required_corner_count` / `required_full_ring_count` / `activation_offsets` / `episode_id` / `step_id` / `agent_id`），`as_dict()` detached、JSON-serializable；target/interior offsets 精确同序且无重复，activation offsets 是非空、无重复、canonical-order 的内部子集并包含实际观测激活点；任意 mapping 或矛盾 activation snapshot 不再能冒充成功；
- 单一权威构造器 `build_c4_c3_frame_identity(episode_id, step_id, agent_id="agent_1", activation_offsets=())`：从 C3 固定门框与 C4 公开几何常量化拼出唯一可被 evaluator 接受的 episode-built 身份；
- 闭集 outcome `IGNITION_OUTCOMES` 19 个；闭集 per-event verdict 包括 ignition / activation / frame_identity / agent verdict 共 5+4+2+1=12 个；
- 优先级 19 层：step_budget → time_budget → abnormal_termination → state-level frame identity geometry check（priority 4，**在任何 C3 success 检查之前** fail closed）→ C3 in_progress 透传 → frame_not_built → ignition_action_missing → wrong_ignition_agent → wrong_ignition_action → wrong_ignition_item → wrong_ignition_target → portal_activation_missing → activation agent 错（priority 13）→ activation_before_ignition → activation_outside_window → external_activation → frame_identity_mismatch → in_progress → success；
- **分层构造合同**：`IgnitionActionEvidence` / `PortalActivationEvidence` / `FrozenFrameIdentity` / `FrozenIgnitionEvaluationState` 构造期只做结构/类型校验；语义白名单（agent / action_type / item / target_cell / interior set 成员 / agent_id 一致 / identity geometry）由 evaluator 产出 `OUTCOME_WRONG_IGNITION_*` / `OUTCOME_EXTERNAL_ACTIVATION` / `OUTCOME_FRAME_IDENTITY_MISMATCH`；
- `PortalActivationEvidence.agent_id` 必填非空标识符；activation agent 与 ignition agent 不一致时 priority 13 产出 `OUTCOME_WRONG_IGNITION_AGENT`；
- `causality_window_steps` 默认 4，inclusive on both ends（delta ∈ [0, 4]）；activation delta > 4 → `activation_outside_window`，delta < 0 → `activation_before_ignition`；
- 闭集 `failure_type` 与 outcome 对齐；`as_dict()` 稳定快照；相同 state 重复 `evaluate()` 产生完全相同 result；
- 复用 `FrozenFrameEvaluator` 重新验证 C3 14-cell 浇筑条件，**不**重复实现 frame 评估逻辑；
- `obsidianlink/env/fake.py` 新增独立 `_ignition_evaluation_state` 槽位与 `set_ignition_evaluation_state` / `get_ignition_evaluation_state` / `clear_ignition_evaluation_state`；与 C1 / C2 / C3 槽位**互不污染**；`reset` / `step` / `close` 清空 C4 truth；`Observation` schema 字段集保持 8 个公开字段不变；
- evaluator 源文件 AST 检查确认：未 import `obsidianlink.agents` / `obsidianlink.workflows` / `obsidianlink.drivers`；未读取 `scenario_parameters` / `evaluator_contract` / `instruction`；`evaluate()` 第二参数严格注解为 `FrozenIgnitionEvaluationState`；
- 144 个专项测试通过：覆盖公开构造 API 下的 wrong agent / wrong action / wrong item / wrong target / external activation / activation 早于 ignition / 超出 4 步窗口 / identity missing / identity geometry mismatch / identity state-vs-activation mismatch / target/interior 重排与重复 / activation snapshot 缺失、重复及与观测点矛盾 / C3 non-success / truth missing / budget / abnormal termination / 确定性重放 / 稳定 `as_dict()` / FakeBackend C1/C2/C3/C4 槽位互不污染 / reset/step/close 清空 / Observation 不泄漏 / evaluator AST 隔离 / immutability 专项测试仅依赖 `object.__setattr__` 验证 `FrozenInstanceError` / C1/C2/C3/portal 回归；
- 全量 779 个离线测试通过；`python -m obsidianlink --check` 与 `python scripts/check_environment.py` 通过；`git diff --check` 干净；
- 下一子任务是 `R6-C4-DETERMINISTIC-DRIVER`；C4 driver 必须等 ignition evaluator 离线完成后再启动。**没有**提前实现 C4 driver、C5 evaluator/driver、真实 MineRL、Gradle 或模型 API。

#### R6-C4-DETERMINISTIC-DRIVER（完成，FakeBackend 离线证明）

- `obsidianlink/drivers/casting_s_c4_ignition.py` 实现 C4 deterministic driver：14-cell × 24 step C3 浇筑子计划 + 4 step C4 ignition 子计划 = **340 step** default plan（落在 700 step 任务预算内）；
- 公开 `ignition_plan` 验证：`action=use_item` / `item=flint_and_steel` / `target_offset=[1, 1, 1]` / `target_policy=exact` 闭集白名单；plan builder 与 `PublicC4IgnitionDriverContext` 构造期只校验 `use_item(flint_and_steel)` 与公开 `[1, 1, 1]`；
- 动作白名单严格闭合：`equip_item` / `use_item` / `place_block` / `wait`；物品白名单 `water_bucket` / `lava_bucket` / `cobblestone` / `flint_and_steel`；永不放 `obsidian`、永不下 Nether；duration_ticks 1..40；
- 预算硬上限：environment step 700、game time 640 秒、plan wait 320、plan length 700、per-action recovery 2、total recovery 32；
- 恢复只响应 typed `RecoverableBackendError`，受 per-action 与 total 双重预算；其他异常 fail closed；
- driver status 闭集 `completed` / `blocked` / `failed`；永不返回 `success` / `passed`（这些 verdict 仍由 `FrozenIgnitionEvaluator` 独立判定）；
- `CastingC4IgnitionDriverResult` 不可变、类型严格、可序列化、暴露 `as_dict()`；新增 `ignition_relevant_action_step` / `ignition_target_offset` / `ignition_equip_step` 让 orchestrator 完全不读 evaluator truth 即可独立构造 `IgnitionActionEvidence`；`per_cell_relevant_action_records` 复用 C3 表面给 `FrozenFrameEvaluator`；
- 5 个新 PHASE 常量（`ignition_equip` / `ignition_use` / `ignition_portal_settle` 等）与 4 个 ROLE 常量（`cast` / `ignition_equip` / `ignition_use` / `ignition_settle`）严格分离 C3 浇筑与 C4 点火子计划；
- `obsidianlink/core/casting_s_c4_ignition_context.py` 是 driver 家族中**唯一**允许读取 task `scenario_parameters` 的函数；`build_public_c4_ignition_driver_context_from_task(task)` 只读取 `public_task_spec.frame_plan.fixed_offsets` 与 `public_task_spec.ignition_plan`，**忽略** `evaluator_contract`；
- `ignition_plan.required` 必须显式存在且原始类型严格为 bool；运行入口只接受与受控 builder 结果完全一致的完整计划，仅点火、截断、重排或重复点火计划在 reset 前 fail closed；
- `obsidianlink/env/capabilities.py` 的 `_GATED_WORKFLOWS` 新增 `casting_s_c4_fixed`，确保 reset 前的 capability 检查覆盖 C4 合同；缺桶/选物品/selected_item/target_block_truth/fluid_truth 任一能力时 fail closed；
- driver 源文件 AST + 源码双门锁确认：未 import `casting_ignition_evaluator` / `casting_frame_evaluator` / `agents` / `workflows` / `model` / `planner`；未通过 `ast.Attribute` / `ast.Subscript` 访问 `scenario_parameters` / `evaluator_contract` / `_ignition_evaluation_state` / `set_ignition_evaluation_state` / `get_ignition_evaluation_state` / `clear_ignition_evaluation_state` / `FrozenFrameIdentity` / `IgnitionActionEvidence` / `PortalActivationEvidence` / `FrozenIgnitionEvaluationState` / `FrozenIgnitionEvaluationResult` / `FrozenIgnitionEvaluator` / `build_c4_c3_frame_identity`；
- 159 个专项测试通过：覆盖公开 context 严格解析与不可变性、required 缺失/错误类型、plan builder 闭集、整体计划防绕过、plan step validation、driver result contract、driver 完整执行、capability gate、340-step / 640s / 320 wait / 700 plan / 32 total recovery / 2 per-step recovery 预算失败、typed recoverable error 有限重试、RuntimeError / OSError / TypeError fail closed、backend terminated / truncated blocked、事件 / `as_dict()` 确定性 replay、闭集 status（不返回 `success` / `passed`）、Observation 8 字段 schema 不泄漏 ignition / latched_frame_identity / nether_portal 等 token、端到端 orchestrator（driver + `set_ignition_evaluation_state` + `FrozenIgnitionEvaluator`）返回 `success` / `activation_outside_window` / `activation_before_ignition` / `external_activation` / `wrong_ignition_agent` / `frame_not_built` / `truth_missing` / `step_budget_exceeded` / `ignition_action_missing` / `portal_activation_missing` / `wrong_ignition_action` / `wrong_ignition_item` / `wrong_ignition_target`、4-step 因果窗口 delta = 0/1/4 都 success、5/6 步 activation_outside_window、delta < 0 activation_before_ignition、FakeBackend C1/C2/C3/C4 槽位独立与互不污染、C1/C2/C3/C4 ignition evaluator 回归、离线 `--check` 与 `check_environment.py` 输出 `status: "ok"` / `phase: "r6_c4_deterministic_driver"`；
- 全量 938 个离线测试通过（779 个旧测试 + 159 个新测试）；`python -m obsidianlink --check` 与 `python scripts/check_environment.py` 通过；`git diff --check` 干净；
- 下一子任务是 `R6-C5-NETHER-ENTRY-EVALUATOR`；C5 evaluator 必须等 C4 driver 离线完成后再启动。该历史子阶段当时**没有**提前实现 C5 evaluator/driver、真实 MineRL、Gradle 或模型 API。

#### R6-C5-NETHER-ENTRY-EVALUATOR（完成，FakeBackend 离线证明）

- `obsidianlink/evaluation/casting_nether_entry_evaluator.py` 提供 frozen、类型严格、可序列化的 `NetherEntryEvidence` / `FrozenNetherEntryEvaluationState` / `FrozenNetherEntryEvaluationResult` 与纯确定性 `FrozenNetherEntryEvaluator`；
- evaluator 复用 `FrozenIgnitionEvaluator` 重新验证 C4 success，指定 `agent_1` 必须从 `minecraft:overworld` 切换到 `minecraft:the_nether`，transition step 不早于 activation，并要求切换前位置、`entered_via_episode_portal=True` 和同一个 typed `FrozenFrameIdentity`；
- 缺失归因与明确外部进入分别稳定分类为 `nether_entry_portal_unknown` 和 `nether_entry_not_via_episode_portal`；预算、异常终止、错误 Agent/维度、缺 transition/position、transition 早于 activation、identity 缺失/不匹配均 fail closed；
- FakeBackend 新增与 C1–C4 隔离的 C5 truth 槽，严格校验 workflow/episode/step/agent，且 reset/step/close 清空，Observation 不泄漏 evaluator-only 进入归因；
- 9 个 C5 专项测试与全量 947 个离线测试通过；
- `casting_s_c5_fixed` 仍保持 `implementation_status=contract_only` 与 `live_run_allowed=false`；没有启动真实 MineRL、Gradle 或模型 API；
- 下一子任务是 `R6-C5-DETERMINISTIC-DRIVER`，不得提前接真实 MineRL 或模型。

#### R6-C5-DETERMINISTIC-DRIVER（完成，FakeBackend 离线证明）

- `obsidianlink/drivers/casting_s_c5_nether_entry.py` 实现 C5 deterministic driver：14 cell × 24 step C3 浇筑子计划 + 4 step C4 ignition 子计划 + 7 step C5 portal approach / entry 子计划（4 approach moves + 1 alignment move + 1 portal-traversal move + 1 settle wait）= **347 step** default plan，落在 800 step 任务预算内；
- 公开 `nether_entry_goal` 验证：designated_agent_ids=`["agent_1"]`、source_dimension=`minecraft:overworld`、target_dimension=`minecraft:the_nether`、required=True 闭集；plan builder 与 `PublicC5NetherEntryDriverContext` 构造期只校验上述公开值；
- 动作白名单严格闭合：`equip_item` / `use_item` / `place_block` / `move` / `wait`；`move` 参数固定为有限前进、无横移/冲刺/跳跃；物品白名单 `water_bucket` / `lava_bucket` / `cobblestone` / `flint_and_steel`；永不放 `obsidian`、driver 不直接修改 dimension 或 portal truth；duration_ticks 1..40；
- 预算硬上限：environment step 800、game time 720 秒、plan wait 320、plan length 700、per-action recovery 2、total recovery 32；
- 恢复只响应 typed `RecoverableBackendError`，受 per-action 与 total 双重预算；其他异常 fail closed；
- driver status 闭集 `completed` / `blocked` / `failed`；永不返回 `success` / `passed`（这些 verdict 仍由 `FrozenNetherEntryEvaluator` 独立判定）；
- `CastingC5NetherEntryDriverResult` 不可变、类型严格、可序列化、暴露 `as_dict()`；新增 `nether_entry_step` / `nether_entry_target_offset` / `nether_entry_approach_step` 让 orchestrator 完全不读 evaluator truth 即可识别 C5 entry 步；`ignition_*` 字段复用 C4 表面给 `FrozenIgnitionEvaluator`；`per_cell_relevant_action_records` 复用 C3 表面给 `FrozenFrameEvaluator`；
- 4 个新 PHASE 常量与 4 个新 ROLE 常量（`entry_approach` / `entry_align` / `entry_teleport` / `entry_settle`）严格分离 C3 浇筑、C4 ignition 与 C5 Nether-entry 子计划；
- `obsidianlink/core/casting_s_c5_nether_entry_context.py` 是 driver 家族中**唯一**允许读取 task `scenario_parameters` 的函数；`build_public_c5_nether_entry_driver_context_from_task(task)` 只读取 `public_task_spec.frame_plan.fixed_offsets`、 `public_task_spec.ignition_plan` 与 `public_task_spec.nether_entry_goal`，**忽略** `evaluator_contract`；
- `nether_entry_goal.required` 必须显式存在且原始类型严格为 bool；固定初始库存必须精确等于 14 水桶、14 熔岩桶、28 圆石和 1 打火石；运行入口只接受与受控 builder 结果完全一致的完整计划，仅 entry、仅 ignition、仅 cast、截断、重排或重复 entry traversal 的计划在 reset 前 fail closed；
- `obsidianlink/env/capabilities.py` 的 `_GATED_WORKFLOWS` 已经包含 `casting_s_c5_fixed`（R6-C5-NETHER-ENTRY-EVALUATOR 阶段已加入），C5 driver 使用同一份 gate，缺桶/选物品/selected_item/target_block_truth/fluid_truth 任一能力时 fail closed；
- driver 源文件 AST + 源码双门锁确认：未 import `casting_nether_entry_evaluator` / `casting_ignition_evaluator` / `casting_frame_evaluator` / `agents` / `workflows` / `model` / `planner`；未通过 `ast.Attribute` / `ast.Subscript` 访问 `scenario_parameters` / `evaluator_contract` / `_nether_entry_evaluation_state` / `set_nether_entry_evaluation_state` / `get_nether_entry_evaluation_state` / `clear_nether_entry_evaluation_state` / `FrozenFrameIdentity` / `IgnitionActionEvidence` / `PortalActivationEvidence` / `FrozenIgnitionEvaluationState` / `FrozenNetherEntryEvaluationState` / `NetherEntryEvidence` / `FrozenNetherEntryEvaluator` / `agents_in_nether` / `entered_via_episode_portal` / `matched_frame_identity` / `latched_frame_identity` / `pre_transition_position`；
- 142 个专项测试通过：覆盖公开 context 严格解析与不可变性、required 缺失/错误类型（含字符串布尔值不被强制转换）、固定初始库存精确匹配、plan builder 闭集、整体计划防绕过、plan step validation、driver result contract、driver 完整执行、capability gate、347-step / 720s / 320 wait / 700 plan / 32 total recovery / 2 per-step recovery 预算失败、typed recoverable error 有限重试、RuntimeError / OSError / TypeError fail closed、backend terminated / truncated blocked、事件 / `as_dict()` 确定性 replay、闭集 status（不返回 `success` / `passed`）、Observation 8 字段 schema 不泄漏 evaluator-only token、端到端 orchestrator outcome 闭集、FakeBackend C1/C2/C3/C4/C5 槽位独立与互不污染及 evaluator 回归；
- C5 driver 142 个专项测试与全量 1089 个离线测试通过；`python -m obsidianlink --check`、`python scripts/check_environment.py` 与 `git diff --check` 均通过；
- 下一任务冻结为 `R6-C5-LIVE-MINERL-BACKEND-WIRING`；仅冻结任务名，真实 backend、MineRL/Minecraft、Gradle 或模型 API 操作仍需用户单独授权。

#### R6-C5-LIVE-MINERL-BACKEND-WIRING（完成，offline）

- `obsidianlink/actions/minerl_translator.py` 扩展 `PORTAL_A0_HOTBAR` 与闭集 `TRANSLATOR_EQUIPPABLE_ITEMS` / `TRANSLATOR_PLACEABLE_ITEMS`，覆盖 R6 C3 / C4 / C5 driver 全部动作（`equip_item(water_bucket|lava_bucket|cobblestone|flint_and_steel)` / `use_item(water_bucket|lava_bucket|flint_and_steel)` / `place_block(cobblestone)` / bounded forward `move` / `wait`）以及 legacy A0 `obsidian` / `flint_and_steel` / `dirt`；strict 类型检查、有限数值限制、闭包空间验证、失败一律 fail closed；
- `obsidianlink/core/types.py` `Observation` dataclass 新增 `selected_item: str | None = None` 公开字段（共 9 字段 schema）；非空字符串、严格不变；
- selected-item 改用 MineRL `HumanSurvival` 自带的 `EquippedItemObservation`，严格读取 `equipped_items.mainhand.type`；
- `obsidianlink/env/portal_spec.py` 只使用 Malmo schema 支持的 `ObservationFromGrid`；单一 `portal_grid` 闭集同时覆盖方块、水/流水、熔岩/流熔岩与 missing/other，范围扩展到 `(-3,-1,0)–(4,5,6)` 以覆盖 C2 x=4 目标；
- `obsidianlink/env/minerl_backend.py` 严格读取 `equipped_items.mainhand.type`，使用任务库存顺序动态映射 hotbar，有限执行 `duration_ticks`，拒绝动作不再推进环境，workflow 保持闭集；新增 typed truth 公共面：
  - `get_casting_evaluation_state(target_cell)` → C1 `CastingEvaluationState`；
  - `get_continuous_casting_evaluation_state(target_cells)` → C2 `ContinuousCastingEvaluationState`；
  - `get_frame_evaluation_state()` → C3 `FrozenFrameEvaluationState`；
  - `get_ignition_evaluation_state()` → C4 `FrozenIgnitionEvaluationState`；
  - `get_nether_entry_evaluation_state()` → C5 `FrozenNetherEntryEvaluationState`；
- 上述 5 个 typed truth 入口 **仅** 从 `raw["portal_grid"]` / `raw["portal_transition"]` 读 world truth；water/lava/obsidian 每个 cell 的 action step 只在合法当前 action 与唯一对应世界变化同时出现时 latch，**永不** 根据 driver intent / action 参数 / Agent prompt 伪造 world truth；
- cast credits 只在 macro translator 成功接受后追加；`action.target == "water_bucket" / "lava_bucket" / "flint_and_steel"` 才记入 credit history；翻译失败的 action **不**记入 credits（fail closed）；`duration_ticks=4` 也只记一次（macro 是一次翻译）；`MAX_TRANSLATOR_DURATION_TICKS=40` 硬上限；
- first-observed water / lava per cell 在 latched state 上独立保留；C1 / C3 evaluator 在同一个 typed state 里同时要求 `water_truth.present=True` 和 `lava_truth.present=True`（水+熔岩会反应为黑曜石，bridge 无法同时返回），backend 通过 first-observation latch 满足；
- pre-existing water / lava（baseline 已是水/熔岩）**不**算 causal credit；agent 不能在 episode 启动前免费拥有流体 truth；
- `Observation` 仍是 9 字段；`target_block_truth` / `fluid_truth` / `portal_grid` / `latched_frame_identity` / `matched_frame_identity` / `agents_in_nether` / `entered_via_episode_portal` / `pre_transition_position` / `nether_entry_evaluation` 全部缺席于 `Observation` / `BackendStep.info` / driver event；
- `reset` / `step` / `close` 都重新初始化 `cast_credit_history` / `first_obsidian_step_by_offset` / `first_water_step_by_offset` / `first_lava_step_by_offset` / `first_ignition_step` / `first_nether_portal_step`；
- `casting_c1_capabilities()` 在 typed truth surface 全部通过离线测试后，将 `exposes_target_block_truth` 与 `exposes_fluid_truth` 报告为 `True`（offline-only）；失败的 production manifest 仍由同一 gate fail closed 测试覆盖；
- 85 个专项离线测试覆盖：翻译器 allowlist / 正路径 / fail-closed / bounded forward `move` / `duration_ticks > 40` / selected item 严格 bridge 读取 / C1–C5 production-backend evaluator success / per-cell 因果归因 / 真值隔离；
- C1 / C2 / C3 / C4 / C5 / portal / frame geometry / CLI / catalog 回归全部通过；全量 1175 个离线测试通过（`Ran 1175 tests in 170.485s → OK`）；`python -m obsidianlink --check` 输出 `phase: "r6_c5_live_minerl_backend_wiring_done"`；`python scripts/check_environment.py` 通过；`git diff --check` 干净；
- **没有**启动真实 MineRL、Gradle 或模型 API；没有提交或推送；C5 仍保持 `implementation_status="contract_only"`、`live_run_allowed=false`；
- 真实 MineRL / Minecraft 中的 typed target-block / fluid / nether-transition truth 仍未验证；C1–C5 success 已在 stub raw trajectory 上通过 production backend + evaluator，但 C5 真实端到端仍取决于尚未验证的 `portal_transition` bridge。

#### R6-C1-LIVE-MINERL-SMOKE-VALIDATION-CONTRACT-FREEZE（完成，offline；只冻结合同）

- 冻结后续真实烟雾验证身份：family=`casting`、mode=`single`、level=`C1`、layout=`fixed`、compatibility task=`casting_c1_fixed`、designated agent=`agent_1`；
- 最小目标：只验证一个目标 cell；必须用水/熔岩与原版 block update 生成黑曜石；禁止预置或直接放置 obsidian；evaluator 独立验证 target/fluid/transition/因果；driver 完成或文本声称不构成成功；truth/坐标/身份/因果不足时 fail closed；
- 授权边界：每次真实 MineRL/Minecraft 运行与每次 Gradle 构建都需用户单独批准；修复后再次真实运行必须重新批准；合同冻结不得把 `live_run_allowed` 改为 `true`；
- 证据要求：`runs/` 至少包含 task_instance / experiment_config / capability_manifest / code_version / initial.png / final.png / events.jsonl / evaluator_events.jsonl / summary.json / manual_review.md；身份字段与 Agent-visible / evaluator-only 隔离保持不变；
- 操作说明：[C1_LIVE_MINERL_SMOKE.md](docs/runbooks/C1_LIVE_MINERL_SMOKE.md)；
- **没有**启动真实 MineRL、Gradle 或模型 API；合同冻结时尚未接线完整 live 入口。

#### R6-C1-LIVE-MINERL-SMOKE-RUNNER-WIRING（完成，offline）

- 核心入口：`obsidianlink/runners/casting_c1_live_smoke.py`；CLI：`scripts/run_c1_live_smoke.py --mode offline_stub --output-dir <绝对路径>`；
- 执行模式闭集仅 `offline_stub`；只接受受控 `OfflineC1StubEnvFactory`，拒绝任意 callable、外部 backend 注入和 live 请求；禁止写入正式 `runs/`；
- Preflight 在 env factory 前校验完整冻结 TaskInstance（含精确预算、精确库存、禁止额外 obsidian）、capability、plan 等值与 catalog `live_run_allowed=false`；
- 编排：C1 deterministic driver → `mark_terminated` → production backend typed truth → 独立 `CastingEvaluator` → evidence bundle → close；
- `driver_status` / `evaluator_success` / `evidence_complete` 分字段；driver completed 不冒充 success；
- 证据在同父目录 staging，完整 10 文件 bundle 经原子 rename 发布；已有输出目录一律拒绝；public events/summary 不含 evaluator-only token；
- **没有**启动真实 MineRL、Gradle 或模型 API；下一步必须是用户单独授权的一次 C1 live smoke run，不得直接进入 C5 live 或 R7。

### R6：完整门框、点火和进入 Nether（按子阶段推进）

R6 已在 R6-C3 / R6-C4 / R6-C5 合同与 FakeBackend 离线证明、MineRL backend typed truth wiring（offline）以及 C1 smoke runner wiring（offline stub）基础上推进。真实 MineRL 验证从 C1 smoke 开始，每次运行需单独授权；不得因 offline wiring 声称 live 已验证。

### R7：模型与受控变化

确定性路径与最小 live smoke 稳定后，才接入受安全约束的模型，并引入有限、可审计的场景变化。

### 后续 Casting 扩展

随机布局、资源不完整、执行噪声和错误恢复；每项变化都需要显式 difficulty parameters、baseline 和回归合同。

## Suite B — Ruined Portal

- P1：废弃传送门结构契约；
- P2：结构识别 evaluator；
- P3：固定场景确定性修复；
- P4：探索与定位；
- P5：完整修复、点火和进入 Nether；
- P6：随机布局与资源差异。

“找到废弃传送门”始终只是里程碑，P5 才覆盖该 family 的端到端成功。

## Suite C — Adaptive Routing

- A1：单可行路线判断；
- A2：路线选择 evaluator；
- A3：两条路线成本比较；
- A4：失败后的路线切换；
- A5：模型规划与动态重规划。

早期先使用只有一条路线可行的可审计场景。可行路线集合和参考成本只属于 evaluator，不能泄漏给 Agent。

## Suite D — Multi-Agent

- M1：多 Agent observation、memory、身份和 evaluator truth 隔离；
- M2：有限消息协议与共享任务板；
- M3：固定分工任务；
- M4：Casting-M；
- M5：Ruined-M；
- M6：Adaptive-M；
- M7：通信受限和协作失败测试。

Suite D 提供正交的 Agent mode 能力，不创建第四个任务族。每个 family 的 Multi-Agent 任务仍使用对应 C/R/A level。

## Benchmark Release

完成任务族与模式的基础验证后，依次建设：

- 场景生成器；
- train/dev/test 划分与去泄漏规则；
- baseline agents；
- 统一 runner；
- 统一结果格式和 evaluator versioning；
- benchmark 数据发布；
- leaderboard；
- 论文实验。

以上均为长期规划。B0 不创建相关实现代码，也不代表 Ruined、Adaptive、Multi-Agent 或真实 MineRL 已受支持。
