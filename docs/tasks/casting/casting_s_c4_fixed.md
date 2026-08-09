# `casting_s_c4_fixed` 门框浇筑与点火任务（C4 合同冻结）

`casting_s_c4_fixed` 是 **Casting-S-C4 / fixed** 合同。它完整继承 C3 的水/熔岩 full-ring 浇筑要求，再增加一次可归因的固定位置点火。本阶段不实现 evaluator、driver 或真实 MineRL 接入。

## 固定合同

- family / mode / level / layout：`casting` / `single` / `C4` / `fixed`
- Agent：`agent_1`
- 初始资源：`water_bucket=14`、`lava_bucket=14`、`cobblestone=28`、`flint_and_steel=1`
- 预算：700 environment steps、640 秒 game time、最多 1 次 model call
- 状态：`contract_only`

门框公开方案与 C3 完全相同：`plane_z`、`min_corner=[0,0,1]`、4×5、Minecraft 最小合法计数 10、本实例要求含四角的 14-block full ring。

## Agent-visible 点火方案

`scenario_parameters.public_task_spec.ignition_plan` 冻结：

| 字段 | 值 |
|---|---|
| `required` | `true` |
| `action` | `use_item` |
| `item` | `flint_and_steel` |
| `target_offset` | `[1,1,1]` |
| `target_policy` | `exact` |

`[1,1,1]` 是公开的唯一计分点火目标。其它内部 cell 上点火即使激活门框，也不满足此固定实例的 C4 合同。该规则同时出现在公开 instruction 中，不属于 evaluator-only truth。

## C4 evaluator 合同

C4 success 必须同时满足：

1. C3 的 14-cell 水/熔岩浇筑与归因条件全部成立；
2. `agent_1` 手持 `flint_and_steel`，在公开目标 `[1,1,1]` 执行合法 `use_item`；
3. 本 episode 建造的门框内部随后出现 `nether_portal`；
4. 激活证据位于上述动作后的 4-step 因果窗口内（inclusive on both ends，delta ∈ [0, 4] 视为窗口内，> 4 视为外部，< 0 视为早于 ignition）；
5. 激活绑定同一个 `latched_frame_identity`（typed `FrozenFrameIdentity`，orientation / min_corner / max_corner / width / height / target_offsets / interior_offsets 与 C3 固定 4×5 full-ring 公开方案一致），不能来自外部门框、命令或 evaluator/driver 写世界；
6. 缺少动作、激活、frame identity 或 step 证据时 fail closed；
7. `PortalActivationEvidence.agent_id` 必填且必须与 ignition action agent 归因一致（activation agent 错时产出 `OUTCOME_WRONG_IGNITION_AGENT`）。

这些规则已写入 `evaluator_contract.activation_attribution`，包括 `require_exact_public_target=true` 和 `require_latched_frame_identity_match=true`。

## C4 evaluator 现状（FakeBackend 离线证明）

`R6-C4-IGNITION-EVALUATOR` 已在 [`obsidianlink/evaluation/casting_ignition_evaluator.py`](../../../obsidianlink/evaluation/casting_ignition_evaluator.py) 完成离线实现，144 个专项测试通过、全量 779 个测试无回归。要点：

- **typed `FrozenFrameIdentity`**：frozen dataclass，13 个显式字段（`orientation` / `min_corner` / `max_corner` / `width` / `height` / `target_offsets` / `interior_offsets` / `required_corner_count` / `required_full_ring_count` / `activation_offsets` / `episode_id` / `step_id` / `agent_id`）；target/interior offsets 必须精确同序且无重复，activation offsets 必须为非空、无重复、canonical-order 的内部子集并包含实际观测激活点；`as_dict()` detached、JSON-serializable。
- **单一权威构造器 `build_c4_c3_frame_identity(episode_id, step_id, agent_id="agent_1", activation_offsets=())`**：从 C3 frozen 门框（`CASTING_S_C3_FRAME_CELLS` / `CASTING_S_C3_INTERIOR_CELLS` / `CASTING_S_C3_CORNER_CELL_COUNT` / `CASTING_S_C3_TARGET_CELL_COUNT`）与 C4 公开 4×5 几何常量（`CASTING_S_C4_FRAME_ORIENTATION = "plane_z"` / `CASTING_S_C4_FRAME_MIN_CORNER = (0,0,1)` / `CASTING_S_C4_FRAME_MAX_CORNER = (3,4,1)` / `CASTING_S_C4_FRAME_WIDTH = 4` / `CASTING_S_C4_FRAME_HEIGHT = 5`）拼出唯一可被 evaluator 接受的 episode-built 身份。任意 mapping 不再能冒充。
- **分层构造合同**：`IgnitionActionEvidence` / `PortalActivationEvidence` / `FrozenIgnitionEvaluationState` 构造期只做结构/类型校验（type / 范围 / bool 拒绝 / 空 identifier 拒绝 / `latched_frame_identity` 必须是 `FrozenFrameIdentity` 实例）；语义白名单（agent / action_type / item / target_cell / interior set 成员 / agent-id 一致 / identity geometry）由 evaluator 产出对应 fail-closed outcome。
- **4-step 因果窗口**：delta ∈ [0, 4] inclusive 视为窗口内；> 4 → `activation_outside_window`；< 0 → `activation_before_ignition`。
- **优先级 19 层**：step_budget → time_budget → abnormal_termination → state-level frame identity geometry check（priority 4，**在任何 C3 success 检查之前** fail closed）→ C3 `in_progress` 透传 → frame_not_built → ignition_action_missing → wrong_ignition_agent → wrong_ignition_action → wrong_ignition_item → wrong_ignition_target → portal_activation_missing → activation agent 错 → activation_before_ignition → activation_outside_window → external_activation → frame_identity_mismatch → in_progress → success。
- **闭集 outcome 19 个**（`IGNITION_OUTCOMES`）：`success` / `in_progress` / `frame_not_built` / `ignition_action_missing` / `wrong_ignition_agent` / `wrong_ignition_action` / `wrong_ignition_item` / `wrong_ignition_target` / `portal_activation_missing` / `activation_before_ignition` / `activation_outside_window` / `external_activation` / `frame_identity_missing` / `frame_identity_mismatch` / `truth_missing` / `step_budget_exceeded` / `time_budget_exceeded` / `invalid_initial_state` / `abnormal_termination`。
- **`external_activation`**：activation `nether_portal_offset` 必须在 `CASTING_S_C4_FRAME_INTERIOR_SET` = `{(1,1,1), (2,1,1), (1,2,1), (2,2,1), (1,3,1), (2,3,1)}` 之内；构造期不做该判断（构造器只校验 xyz tuple），由 evaluator 产出 `OUTCOME_EXTERNAL_ACTIVATION`。
- **FakeBackend 独立 C4 truth 槽位**（`obsidianlink/env/fake.py`）：`_ignition_evaluation_state` 与 `_casting_evaluation_state` (C1) / `_continuous_casting_evaluation_state` (C2) / `_frame_evaluation_state` (C3) 互不污染；`set_ignition_evaluation_state` 严格身份校验（`casting_s_c4_fixed` workflow / `episode_id` / `step_id` / `agent_id`），`reset` / `step` / `close` 自动清空；`Observation` 公开字段集保持 8 个字段不变，ignition truth 不进入 Observation。

## 信息隔离

点火动作、物品和目标位置属于公开规则。隐藏内容仅包括实际 `nether_portal` 方块变化、`first_activation_step`、`latched_activation_offsets`、`latched_frame_identity` 及 evaluator 判定。evaluator 源文件 AST 验证不 import `obsidianlink.agents` / `obsidianlink.workflows` / `obsidianlink.drivers`，不读取 `scenario_parameters` / `evaluator_contract` / `instruction`。

## 当前实现状态

已冻结 C4 合同、catalog、配置、文档、离线一致性测试与 FakeBackend 上的 ignition evaluator + typed `FrozenFrameIdentity`（R6-C4-IGNITION-EVALUATOR 完成）。**未**实现 ignition deterministic driver、C5 Nether-entry evaluator/driver、真实 MineRL 或模型接入；active compatibility 仍为 `casting_c3_fixed` (C2)。

下一子任务：`R6-C4-DETERMINISTIC-DRIVER`，但必须等本轮 ignition evaluator 收口并审计完成后再启动。
