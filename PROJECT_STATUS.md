# 当前状态

更新时间：2026-08-05

## 当前唯一目标

任务：`R3-CASTING-EVALUATOR`（已完成）

下一任务：`R4-DETERMINISTIC-CASTING-DRIVER`（尚未开始）

为 `casting_c1_fixed` 实现一个纯离线、类型严格、不可变、fail-closed 的 evaluator，用 evaluator-only 真值回答：

- 目标 cell 是否成功从非黑曜石变成黑曜石；
- 是否同时满足 reset 状态、流体证据、相关动作因果、预算、正常终止等完整证据链；
- 真值缺失时是否明确返回 `truth_missing` 而不是 `success`；
- 错误方块、超预算、非正常终止、因果证据不足时是否给出稳定结果。

## 建议修改位置（R3 实际完成）

- 新增 `obsidianlink/evaluation/casting.py` ✓
- 新增 `tests/test_casting_evaluation.py` ✓
- `obsidianlink/evaluation/__init__.py` 增加 casting 模块导出 ✓
- `FakeEnvironmentBackend` 增加 evaluator-only 表面：`set_casting_evaluation_state` / `get_casting_evaluation_state`，`reset` / `step` / `close` 时严格清空 ✓
- `obsidianlink/cli.py` 与 `scripts/check_environment.py` 把阶段标识推进到 `reset_3_casting_evaluator` ✓
- `tests/test_cli.py` 同步阶段断言 ✓

不实现 R4 driver、不接 VLM、不修改固定依赖、不动 `vendor/minerl`、不启动真实 MineRL。

## 交付内容

1. 一个不可变、类型严格的 `CastingEvaluationState` 数据对象，承载 `episode_id`、`step_id`、目标 cell、初始/当前方块真值、水/熔岩证据、目标更新证据、相关动作 step、因果窗口、终止信号、预算与时间戳等 evaluator-only 字段。 ✓
2. 一个不可变、JSON 可序列化的 `CastingEvaluationResult`，包含 `success`、`outcome`、`failure_type`、`blocking_conditions`、`evidence`、终止 / 预算 / 上次成功里程碑等。 ✓
3. 一个 `CastingEvaluator` 纯函数/纯对象：相同输入产生相同结果，从不读取 Agent 文本 / 图像 / Planner 输入。 ✓
4. 9 个稳定 outcome id（`success` / `in_progress` / `wrong_block` / `truth_missing` / `step_budget_exceeded` / `time_budget_exceeded` / `invalid_initial_state` / `causality_missing` / `abnormal_termination`）和封闭白名单。 ✓
5. 优先级锁定的判定规则（最具体优先；同一 state 不会因为实现顺序得到不同 outcome）。 ✓
6. `FakeEnvironmentBackend` 的 evaluator-only 状态注入/读取接口，含 `episode_id` / `step_id` 一致性校验，未注入 / 越步读取严格报错。 ✓
7. casting truth 不得进入 `Observation.frame` / `visible_inventory` / `messages` / `workflow_stage` / Planner 提示 / memory。 ✓
8. 当前真实 MineRL backend 仍因 7 项能力缺失而 fail closed；本轮不绕过 `assert_backend_can_start_task`，不启动真实环境。 ✓
9. PortalEvaluator 与 R2 全部测试不回归。 ✓

## R3 已完成

### evaluator 类型（不可变 + 类型严格）

`obsidianlink/evaluation/casting.py` 导出：

- `CastingEvaluationState`（`@dataclass(frozen=True)`）—— 全部字段在 `__post_init__` 严格校验：非空 `episode_id`、非负 `step_id`、`target_cell` 必须是 3 元组严格 int（拒绝 `bool`）、`current_target_block` / `initial_target_block` 必须在封闭白名单 `TARGET_BLOCK_IDS` 中、NaN / Infinity / 空字符串 / 负 step / 越界 `causality_window_steps` 全部被 `ValueError` 拒绝。
- `CastingFluidTruth`（`@dataclass(frozen=True)`）—— 三态布尔（`present: bool | None`）+ `evidence_step`。
- `CastingTransitionEvidence`（`@dataclass(frozen=True)`）—— `before_block` / `after_block`（封闭白名单） + `update_step`。
- `CastingEvaluationResult`（`@dataclass(frozen=True)`）—— `evidence` 递归冻结为只读树，`as_dict()` 返回与内部状态分离、可被 `json.dumps` 序列化的快照；`outcome` / `failure_type` 取自封闭 `OUTCOMES` / `_TERMINAL_FAILURE_OUTCOMES`。
- `CastingEvaluator` —— 单方法 `evaluate(self, state: CastingEvaluationState) -> CastingEvaluationResult`，签名只接受 `CastingEvaluationState`，无任何其他输入；无 I/O、无随机、无全局状态。

### 真值如何区分"明确 False"与"未知"

- 方块 / 更新证据：用 `str | None`；`None` 即"未知"，具体值即"明确观察到"。
- 流体证据：用 `CastingFluidTruth(present=...)`；`present is None` 即"未知"，`present is True / False` 即"明确观察到 / 明确未观察到"。
- 任何 `None` 命中 `truth_missing` 列表，evaluator 返回 `OUTCOME_TRUTH_MISSING` 且不会返回 `success`。
- `water_truth.present is False` / `lava_truth.present is False` 是明确反证；目标即使为黑曜石也返回 `causality_missing:*_not_present`，不能成功。

### 成功（OUTCOME_SUCCESS）的完整判定条件

只有以下**全部**满足才返回 `success=True`：

1. `step_id <= max_environment_steps`（step 预算未超）；
2. `current_time_seconds <= max_game_time_seconds`（时间预算未超）；
3. `initial_target_block is not None and initial_target_block != "obsidian"`（reset 状态合法）；
4. 全部 required truth 齐全：`initial_target_block` / `current_target_block` / `water_truth.present` / `lava_truth.present` / `target_update_evidence` 含 `before_block`、`after_block` 与 `update_step` / `relevant_action_steps` 非空；
5. `episode_terminated is True` 且 `terminated_reason in NORMAL_TERMINATION_REASONS`；
6. `current_target_block == "obsidian"`；
7. 水与熔岩真值都明确为 `True`，且 `target_update_evidence.after_block == current_target_block == "obsidian"`；
8. `target_update_evidence.update_step` 与最新相关 `relevant_action_steps` 的差值在 `[0, causality_window_steps]` 内（即方块变化发生在相关 Agent 动作之后的有限因果窗口内）。

### outcome / failure_type 稳定分类及优先级

evaluator 按以下固定顺序匹配，第一个命中即返回（最高 → 最低）：

| 优先级 | outcome | 何时返回 |
|---|---|---|
| 1 | `step_budget_exceeded` | `max(step_id, terminated_step) > max_environment_steps` |
| 2 | `time_budget_exceeded` | `current_time_seconds > max_game_time_seconds` |
| 3 | `invalid_initial_state` | `initial_target_block == "obsidian"` |
| 4 | `truth_missing` | 任何 required truth 为 `None`（具体字段在 `blocking_conditions` 与 `evidence.missing_truth` 列出） |
| 5 | `in_progress` | `episode_terminated is False`（即使已观察到目标 obsidian，也不算 success） |
| 6 | `abnormal_termination` | 已终止但 `terminated_reason` 不在 `NORMAL_TERMINATION_REASONS` 中 |
| 7 | `causality_missing` | 目标 obsidian 但水/熔岩明确缺席、transition 未生成 obsidian、update 早于相关动作或超过有限窗口；evidence 记录稳定 `causality_reason` |
| 8 | `wrong_block` | 终止且所有 truth 齐全，但 `current_target_block != "obsidian"`；evidence 记录 `actual_block` |
| 9 | `success` | 全部成功条件满足 |

`failure_type` 与 `outcome` 在 1-8 步同步；`success` / `in_progress` / `truth_missing` 的 `failure_type` 为 `None`，与 `EvaluationResult` 既有约定一致。

### 有限因果窗口如何验证

- `CastingEvaluationState.causality_window_steps` 是**类型化上界**，默认 `DEFAULT_CAUSALITY_WINDOW_STEPS = 4`，最大 `MAX_CAUSALITY_WINDOW_STEPS = 32`；构造时即校验 `1 <= value <= 32`。
- evaluator 计算 `update_step - last_relevant_action_step`，要求 `0 <= delta <= causality_window_steps`。
- 窗口不是软启发式而是硬上界：`causality_missing:outside_window` 永远会阻断 success。
- `relevant_action_steps`、流体 `evidence_step`、目标 `update_step`、`terminated_step` 都不得晚于当前 `step_id`；负时间与非正时间预算在构造时拒绝。

### FakeBackend 如何注入 / 读取 casting state

新增两个与旧 Portal 表面完全分离的方法：

```python
backend.set_casting_evaluation_state(state)   # 注入
backend.get_casting_evaluation_state()         # 读取
```

- `set_casting_evaluation_state` 拒绝非 `CastingEvaluationState`（`TypeError`）、`episode_id` 不匹配当前 task（`ValueError`）、`step_id` 不匹配当前 backend step（`ValueError`）。
- `get_casting_evaluation_state` 在 reset 之前 / 注入之前 / open 之前分别抛 `RuntimeError("not been reset")` / `RuntimeError("casting evaluation state is unavailable")` / `RuntimeError("not open")`。
- `reset` / `step` / `close` 都会清空 `_casting_evaluation_state`，使旧 step 的 stale state 立刻失效。

### evaluator-only 信息如何证明不进入 Observation

- `FakeEnvironmentBackend._observations` 只写入 `frame={"backend": "fake", "step_id": ...}` + `visible_inventory` + `workflow_stage`，从不读 `_casting_evaluation_state`。
- `tests.test_casting_evaluation.FakeBackendCastingStateTests.test_observation_does_not_leak_casting_truth` 在 reset 与 step 两条路径上同时注入丰富 casting 真值并检查 Observation / frame 不含 `target_cell` / `target_block` / `initial_target_block` / `current_target_block` / `fluid_truth` / `water_truth` / `lava_truth` / `casting_evaluator` / `casting_outcome` / `success` / `blocking_conditions` 等任一字段。
- `CastingEvaluatorIsolationTests.test_evaluator_source_does_not_import_agents_or_workflows` 通过源码扫描保证 casting 模块从不引用 `obsidianlink.agents` / `obsidianlink.workflows` / `obsidianlink.drivers` / `Observation` / `MacroAction` / `VLM` / `vlm` / `Qwen`。
- `CastingEvaluatorIsolationTests.test_evaluator_signature_only_accepts_state` 用 `typing.get_type_hints` 锁住 `evaluate` 只接收 `CastingEvaluationState`。

### 当前真实 MineRL 为什么仍不能运行 casting episode

`MineRLEnvironmentBackend.casting_c1_capabilities()`（与 R2 修一一致）仍声明 7 项缺口：

- `can_select_water_bucket` = False（翻译器 hotbar 无 water_bucket）
- `can_select_lava_bucket` = False（翻译器 hotbar 无 lava_bucket）
- `can_use_water_bucket` = False（翻译器无 use_item 翻译）
- `can_use_lava_bucket` = False（翻译器无 use_item 翻译）
- `exposes_selected_item` = False（bridge 不暴露当前手持栏）
- `exposes_target_block_truth` = False（`get_evaluation_state()` 不携带类型明确的目标 cell 真值）
- `exposes_fluid_truth` = False（bridge 无流体真值接口）

`assert_backend_can_start_task` 在 `MineRLEnvironmentBackend.reset` 最早处触发门禁，env_factory 一次都不被调用（`test_minerl_casting_reset_still_rejected_before_env_creation` 用 spy 跟踪验证）。R3 没有新增任何 casting 能力实现，也没有把上述任一字段改为 True；`CurrentMineRLStateTests` 锁住这个事实。

### 新增测试数量与覆盖范围

`tests/test_casting_evaluation.py` 共 63 个用例：

- `CastingStateImmutabilityTests`（7）—— state/result 均为 `frozen`；嵌套 evidence 递归只读；`as_dict()` 可 JSON 序列化且返回分离快照；同一 state 重复 evaluate 产生完全相同结果。
- `CastingStateValidationTests`（10）—— 除原有严格类型/数值测试外，新增非正时间预算、负当前时间、未来 action/fluid/update/terminated step 全部拒绝。
- `CastingEvaluatorOutcomeTests`（26）—— 除 9 个 outcome 与优先级覆盖外，新增明确无水/无熔岩、transition 非 obsidian、缺终止原因、终止后 current step 超预算等 fail-closed 反例。
- `CastingEvaluatorIsolationTests`（3）—— evaluator 不会读取 Observation（即使传入包含假 success payload 的 Observation，仍返回 `truth_missing`）；签名只接受 `CastingEvaluationState`；源码不引用 agents/workflows/drivers/Observation/MacroAction/VLM/Qwen。
- `FakeBackendCastingStateTests`（10）—— set/get 正常路径；reset 前 / open 前 / 未注入读取分别报错；wrong episode_id / wrong step_id / 非 `CastingEvaluationState` 注入被拒；`step` 后 casting state 自动清空（旧 step_id 失效）；`close` 后状态彻底清空；`Observation` / `Observation.frame` 在 reset 与 step 路径均不含 casting truth。
- `CurrentMineRLStateTests`（2）—— MineRL `casting_c1_capabilities` 仍缺 7 项；`MineRLEnvironmentBackend.reset(casting_c1_task)` 仍在 env_factory 之前 fail closed（env_factory spy 验证）。
- `PortalEvaluatorRegressionTests`（3）—— R3 注入的 casting truth 不影响 PortalEvaluator（`frame_never_valid` 仍命中）；仅注入 Portal truth 时 `get_casting_evaluation_state` 拒绝（表面互不干扰）；旧 `set_evaluation_state` / `get_evaluation_state` 仍工作。
- `BlockIdWhitelistTests`（2）—— 白名单含 `obsidian` / `air` / `cobblestone` / `stone`；任意非白名 block（如 `dirt`）被 `__post_init__` 拒绝。

合计 R3 单测 63 个；R3 之前 180 个；总计 243 个，离线用例全部通过。

### 本轮没有启动以下任何一项

- Minecraft / MineRL / Gradle / 付费模型 API；
- `vendor/minerl` 未修改（`git status --short -- vendor/minerl` 为空）；
- 未实现 R4 driver、未实现真实水/熔岩模拟、未实现 VLM 接入；
- 未生成 `runs/` 真实运行证据；
- 未 `git commit`、未 `git push`；
- 未修改固定依赖版本。

### 本轮运行的测试与实际结果

```text
python -m obsidianlink --check                                      → status=ok, phase=reset_3_casting_evaluator
python scripts/check_environment.py                                 → project_files 全部存在, phase=reset_3_casting_evaluator
python -m unittest tests.test_capabilities -v                       → Ran 34 tests in 0.001s — OK
python -m unittest tests.test_casting_evaluation -v                 → Ran 63 tests — OK
python -m unittest tests.test_evaluation -v                         → Ran 22 tests in 0.001s — OK
python -m unittest discover -s tests -p 'test_*.py'                 → Ran 243 tests in 52.876s — OK
                                                                    (R2 收尾后 180 + R3 新增 63 = 243)
git diff --check                                                    → 干净
git status --short                                                  → 15 个变动条目（11 M + 4 ??），全在项目内
git status --short -- vendor/minerl                                 → 空（vendor 未修改）
```

## 当前限制

- 真实 MineRL backend 仍缺 7 项能力（见上表）。任何真实 casting episode 必须先把能力补齐并诚实更新 `casting_c1_capabilities()`，否则 `assert_backend_can_start_task` 会 fail closed。
- 当前 evaluator 只能从 `FakeEnvironmentBackend` 的 `set_casting_evaluation_state` / `get_casting_evaluation_state` 接收 casting truth。真实 MineRL 仍没有把 `target_cell` / 流体 / `target_update_evidence` 接到 `get_evaluation_state()` 的 casting-only 表面。
- `casting_c1_fixed` 还没有 driver（铸造 + 等待 + 验证 + 终止的有限循环）；本轮只实现 evaluator 契约与 FakeBackend 离线证明，driver 留给 R4。
- 真实 episode 仍禁止真实 MineRL、Gradle 和模型调用。

## 已完成

- 核心类型、动作白名单、FakeBackend 和结构化日志；
- MineRL 单角色生命周期与 Portal 环境桥接；
- Portal 框架、激活和进入下界的自动评估；
- 活动任务 `benchmark/instances/active/casting_c1_fixed.json`；
- 离线实验契约 `configs/experiments/active/casting_c1_contract.json`；
- `R2-CAPABILITY-MANIFEST`（已完成）：不可变 + JSON 可序列化 `BackendCapabilities`、workflow-aware `assert_backend_can_start_task` 门禁、`reset` 路径强制门禁、FakeBackend 与 MineRL backend 正反例测试。
- `R3-CASTING-EVALUATOR`（当前完成）：不可变 + JSON 可序列化 `CastingEvaluationState` / `CastingEvaluationResult`、`CastingEvaluator` 纯函数、9 个稳定 outcome id、9 步优先级锁定的 fail-closed 规则、FakeBackend 注入/读取表面与 `episode_id` / `step_id` 校验、信息隔离与全量回归测试。

## 下一任务

任务：`R4-DETERMINISTIC-CASTING-DRIVER`

在 R3 已经固化的 evaluator 契约基础上，用有限动作、有限等待、有限重试在 FakeBackend 上跑通 `casting_c1_fixed` 的 `success` 路径；驱动必须：

- 只使用 Agent 可见的 Observation（`pov` / `visible_inventory` / `workflow_stage`），不读 evaluator-only 真值；
- 严格使用 `MacroAction` 白名单（`select_water_bucket` / `select_lava_bucket` / `use_item` / `place_block` / `wait`）的翻译与数值限制；
- 维护 step / 时间预算并与 evaluator 的 `step_budget_exceeded` / `time_budget_exceeded` 兼容；
- 真实 MineRL driver 另行申请授权，本轮仅在 FakeBackend 上证明可重放。

不要实现 R5（连续浇筑）、R6（完整门框与点火）、R7（VLM），不要改 `vendor/minerl`，不要启动 MineRL。

## 下一 Agent 直接执行

> R3 已完成：`obsidianlink/evaluation/casting.py` 提供 `CastingEvaluationState` / `CastingEvaluationResult` / `CastingEvaluator` 与 9 个稳定 outcome id；成功链要求水/熔岩明确存在、transition 生成 obsidian、正常终止和时间线一致；`FakeEnvironmentBackend` 暴露隔离的 casting truth 表面；243 个离线用例全过（其中 R3 新增 63 个）；当前真实 MineRL 仍因 7 项能力缺失 fail closed，vendor/minerl 未动。
> 本轮**不要**重做 R3；本轮**不要**启动 MineRL / Gradle / 模型 API。
> 开始 R4：在 FakeBackend 上用 `CastingEvaluator` 跑通一个确定性 driver，先证明 `success` 路径可重放，再考虑边缘场景。
