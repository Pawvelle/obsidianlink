# 当前状态

更新时间：2026-08-05

## 当前唯一目标

任务：`R5-CONTINUOUS-CASTING`（已完成）

下一任务：`R6-COMPLETE-PORTAL-FRAME`（尚未开始）

把 `casting_c1_fixed` 的单块浇筑扩展为 `casting_c3_fixed`（3 个有序目标 cell 的固定直线区段），在 FakeBackend 上验证多次液体操作的因果证据分别归属于对应 cell、单个 cell 的失败不会被掩盖、恢复/重试有严格硬上限、driver 不读取 evaluator-only 真值、evaluator 独立判定。

## 建议修改位置（R4 实际完成）

- 新增 `obsidianlink/drivers/casting_c1.py`（driver + 计划构建器 + 有限循环/等待/重试封顶） ✓
- 新增 `tests/test_casting_driver.py`（driver contract + 编排器 outcome + 重放稳定 + 隔离 + 漏出检查） ✓
- `obsidianlink/drivers/__init__.py` 导出新增的 casting_c1 表面 ✓
- `obsidianlink/cli.py` 把 R4 driver 接入离线 contract check，阶段推进到 `reset_4_deterministic_casting_driver` ✓
- `scripts/check_environment.py` 同步阶段标识 ✓
- `tests/test_cli.py` 同步阶段断言 + 新增 driver outcome 字段 ✓

不实现 R5 连续浇筑、不接 VLM、不修改固定依赖、不动 `vendor/minerl`、不启动真实 MineRL。

## R4 交付内容

1. 一个**确定性、有限动作、有限循环**的 `run_casting_c1_driver(backend, task, ...)` 入口：基于 `build_casting_action_plan()` 输出的固定计划（24 步），只使用公共动作协议中的 R4 白名单（`equip_item` / `use_item` / `place_block` / `wait`），不调用任何模型代码、不打开 shell、不接 VLM。
2. 一个**递归不可变、可 JSON 序列化**的 `CastingC1DriverResult`（`status` ∈ {`completed` / `blocked` / `failed`}、有限计数、事件、相关动作、最终 observation、后端 `terminated` / `truncated` 与阻断原因），并提供深度分离的 `as_dict()` 快照。
3. 三类运行**硬上限**与一个计划上限：`max_environment_steps` / `max_game_time_seconds` 不得超过 task 合同，`max_wait_steps <= 32`，构建计划的总 wait 也不得超过 32；step/wait 在动作提交前拦截，任何超限以 `status="blocked"` 报告。
4. 一个**信息隔离**的 source 守卫：driver 永不 `import` `CastingEvaluator` / `CastingEvaluationState` / `CastingFluidTruth` / `CastingTransitionEvidence` / `CastingWorldTruth`（AST 扫描测试锁定），永不调用 `backend.set_casting_evaluation_state` / `backend.get_casting_evaluation_state`（spy 测试锁定），永不读取 `Observation` 上任何 `target_cell` / `target_block` / `initial_target_block` / `current_target_block` / `fluid_truth` / `water_truth` / `lava_truth` / `casting_*` / `success` / `outcome` / `failure_type` 字段（monkey-patch 测试锁定）。
5. 一个**测试侧编排器**（`tests/test_casting_driver.py` 内的 `run_orchestrator` / `build_evaluation_state` / `CastingWorldTruth`）：唯一允许调用 `set_casting_evaluation_state` / `get_casting_evaluation_state` 和 `CastingEvaluator` 的位置；从 `driver_result.relevant_action_steps` 与 `CastingWorldTruth` 构造 `CastingEvaluationState`，并把 backend step 上限对齐到 `_step_id`。
6. 9 步 outcome 路径全部在 driver + 编排器 + evaluator 端到端测试中跑过：`success`（正常路径）、`in_progress`（未到 step 截止）、`wrong_block`（target 不是 obsidian）、`truth_missing`（water / lava / transition / relevant_action 缺失）、`step_budget_exceeded`（orchestrator 报告超 step 预算）、`time_budget_exceeded`（orchestrator 报告超时间预算）、`invalid_initial_state`（reset 即 obsidian）、`causality_missing`（transition 在 relevant action 窗口外）、`abnormal_termination`（reason 不在白名单）。
7. 关键时间/重放/隔离/漏出测试：driver 顺序的 step_id 与 backend 严格单调对应、同一 driver 在相同输入下产生相同 `action_label_for_step` 与 `relevant_action_steps`、跨两次完整 driver+orchestrator run 的 outcome 一致、driver 结束后的 `Observation` 仍不含 casting 字段、driver 任何 `Observation` 都不含 casting 字段、driver 拒绝 workflow ≠ `casting_c1_fixed`、driver 拒绝 missing inventory。
8. `BackendStep.__post_init__` 守住单个返回值内部的 step_id / episode_id / agent_id 契约；driver 另行校验跨调用 step 必须恰好递增 1，拒绝跳步或 stale 返回。
9. CLI 离线 contract check 与 `scripts/check_environment.py` 阶段都推进到 `reset_4_deterministic_casting_driver`，并在 CLI 报告 `driver_status="completed"` 与 `driver_success_outcome="success"`。

## R4 已完成

### Driver 公开表面（`obsidianlink/drivers/casting_c1.py`）

- `ALLOWED_R4_ACTION_TYPES`（frozenset，4 项）+ `ALLOWED_R4_TARGETS`（frozenset，3 项）：所有计划动作都能通过项目公共 `parse_macro_action`。
- `DRIVER_STATUS_*`（`completed` / `blocked` / `failed`）：driver 永不返回 `passed` / `success`——verdict 由 evaluator 决定。
- `build_casting_action_plan(*, support_block_wait_steps, fluid_settle_wait_steps, obsidian_wait_steps) -> tuple[CastingPlanStep, ...]`：固定顺序（select lava → 2× place_block cobblestone → use lava → select water → use water → 等待 obsidian 凝固），所有 wait 数都是有限正整数。
- `run_casting_c1_driver(backend, task, *, plan, max_wait_steps, max_environment_steps, max_game_time_seconds, event_sink) -> CastingC1DriverResult`：driver 入口；按计划调 `backend.step()`，每步前置 inventory / allowlist / step / wait 检查，后置严格校验 episode 与 step 必须恰好递增 1；提前 termination/truncation 会 fail closed。

### Driver 永不触碰 evaluator-only 表面

- AST 扫描（`test_driver_source_does_not_import_casting_evaluator`）锁定 `obsidianlink/drivers/casting_c1.py` 顶层 `import` / `import from` 集合不含 `CastingEvaluator` / `CastingEvaluationState` / `CastingEvaluationResult` / `CastingFluidTruth` / `CastingTransitionEvidence` / `CastingWorldTruth`，且模块内任何 `Attribute` 节点都不拼出 `set_casting_evaluation_state` / `get_casting_evaluation_state`。
- Spy 测试（`test_driver_does_not_call_casting_truth_surface`）在 backend 上 monkey-patch 这两个方法，断言 driver 整个生命周期调用次数为 0。
- `Observation.__getattribute__` 守卫（`test_driver_does_not_leak_truth_into_observation`）拦截任何对 `target_cell` / `target_block` / `initial_target_block` / `current_target_block` / `fluid_truth` / `water_truth` / `lava_truth` / `casting_evaluator` / `casting_outcome` / `success` / `blocking_conditions` 的访问，driver 跑通完整 plan 不触发。
- 截取所有 driver 期间的 `Observation`（`test_all_driver_observations_are_clean`）逐个检查 `hasattr(observation, f)` 与 `observation.frame` 中都不含上述 forbidden 字段。

### Driver 固定动作流程（默认 plan 24 步）

| step_id 区间 | 动作 | 阶段 | `relevant_action` |
|---|---|---|---|
| 1 | `equip_item(lava_bucket)` | `prepare` | False |
| 2 | `wait` | `prepare` | False |
| 3 | `place_block(cobblestone)` | `place_support` | True |
| 4 | `wait` | `place_support` | False |
| 5 | `place_block(cobblestone)` | `place_support` | True |
| 6 | `wait` | `place_support` | False |
| 7 | `equip_item(lava_bucket)` | `place_lava` | False |
| 8 | `wait` | `place_lava` | False |
| 9 | `use_item(lava_bucket)` | `place_lava` | True |
| 10–13 | `wait` × 4 | `place_lava` | False |
| 14 | `equip_item(water_bucket)` | `place_water` | False |
| 15 | `wait` | `place_water` | False |
| 16 | `use_item(water_bucket)` | `place_water` | True |
| 17–20 | `wait` × 4 | `place_water` | False |
| 21–24 | `wait` × 4 | `wait_for_obsidian` | False |

`relevant_action_steps` 在 FakeBackend 上稳定为 `(3, 5, 9, 16)`（4 个 relevant action）；orchestrator 直接把它传给 `CastingEvaluationState.relevant_action_steps`。

### 如何保证有限循环与信息隔离

- **动作白名单 + 严格类型/数值限制**：`ALLOWED_R4_ACTION_TYPES` / `ALLOWED_R4_TARGETS` 是封闭 `frozenset`；`build_casting_action_plan` 构造时调 `_require_r4_action` 逐条 plan 步验证；`run_casting_c1_driver` 在每次 `step` 调用前再校验一次 plan step 的 action + target + duration。
- **总步数硬上限**：默认读取 task 的 160 step，调用方只能收紧不能放宽；在提交下一动作前检查，step 10 的预算绝不会执行 step 11。
- **总时间硬上限**：`max_game_time_seconds` 默认 120.0，构造时拒绝 0 / 负数 / `inf` / `nan`；每步后置检查 `next_observation.timestamp - reset_timestamp > max_game_time_seconds`（**elapsed time 不是 wall-clock**），超出立即 `status="blocked"`，event `budget_exceeded: "time"`。
- **等待/计划硬上限**：`max_wait_steps <= 32` 且在 wait 提交前检查；计划构建器也在分配 tuple 前检查总 wait，不接受超大输入。
- **提前终止**：后端在完整计划前返回 `terminated` / `truncated` 时，driver 返回 `blocked` 并原样记录标志，不再误报 `completed`。
- **无重试 / 无循环**：plan 是固定 tuple，driver 仅 `for plan_step in plan` 单次遍历；遇到 `step.terminated=True` 或后端抛 `RuntimeError` / `OSError` / `TypeError` 立即 `break` 并设 `blocked_reason`。
- **信息隔离三道闸**：(a) driver 不 import casting 模块（AST 扫描），(b) driver 不调 `set/get_casting_evaluation_state`（spy），(c) driver 不读 `Observation` 上的 casting 字段（`__getattribute__` 守卫 + 截取所有 observation 静态扫描）。evaluator 表面只在测试编排器 `run_orchestrator` / `build_evaluation_state` 内出现。

### 测试结果（R4 新增 43 个 + R3 保留 63 个 + R3 之前 180 个 = 286 个）

```text
python -m obsidianlink --check                              → status=ok, phase=reset_4_deterministic_casting_driver, driver_status=completed, driver_success_outcome=success
python scripts/check_environment.py                         → project_files 全部存在, phase=reset_4_deterministic_casting_driver
python -m unittest tests.test_casting_driver -v             → Ran 43 tests — OK
python -m unittest tests.test_casting_evaluation -v         → Ran 63 tests — OK (无回归)
python -m unittest tests.test_evaluation -v                 → Ran 22 tests — OK (无回归)
python -m unittest tests.test_capabilities -v               → Ran 34 tests — OK (无回归)
python -m unittest discover -s tests -p 'test_*.py'         → Ran 286 tests in 53.584s — OK
git diff --check                                            → 干净
git status --short                                          → 9 个变动条目（7 M + 2 ??）
git status --short -- vendor/minerl                         → 空（vendor 未修改）
```

### R4 新增测试覆盖范围

`tests/test_casting_driver.py` 共 43 个用例，分布在 6 个 TestCase 中：

- `DriverContractTests`（13）—— 动作/目标/状态白名单、公共 parser 兼容、plan 长度与总 wait 上限、参数验证、task 预算不可放宽、workflow/type/source 隔离等契约。
- `FakeBackendDriverTests`（12）—— 完整 plan、step/time/wait 前置预算、提前 termination fail closed、结果深度不可变、event sink 隔离、缺库存、重放稳定和 truth surface 隔离。
- `OrchestratorOutcomeTests`（8）—— 正常路径 → `success`、causality delta=0 → `success`、transition 远超窗口 → `causality_missing:outside_window`、target 是 `cobblestone` → `wrong_block`、water/lava/transition 全 None → `truth_missing`、orchestrator 报 step budget 超限 → `step_budget_exceeded`、orchestrator 报时间预算超限 → `time_budget_exceeded`、relevant_action_steps 空 → `truth_missing`。
- `StaleStepAndIsolationTests`（7）—— typed BackendStep 身份校验、跨调用 step 跳跃拒绝、driver event 单调、重放稳定以及 evaluator 使用 driver 相关动作证据。
- `ObservationLeakageTests`（2）—— driver 结束 + 编排器跑完后，`final_observation` 与 `frame` 仍无 casting 字段；driver 跑期间所有 `Observation` 都不漏出 casting 字段。
- `DriverBackendShapeTests`（1）—— 一个最小 backend（仅 `reset` / `step`）即可被 driver 接受并跑完完整 plan。

合计 R4 单测 43 个；R3 之前 243 个；总计 286 个，离线用例全部通过。

### 本轮没有启动以下任何一项

- Minecraft / MineRL / Gradle / 付费模型 API；
- `vendor/minerl` 未修改（`git status --short -- vendor/minerl` 为空）；
- 未实现 R5 连续浇筑、未实现 VLM 接入；
- 未生成 `runs/` 真实运行证据；
- 未 `git commit`、未 `git push`；
- 未修改固定依赖版本。

## 当前限制

- 真实 MineRL backend 仍缺 7 项能力（详见 R3 段）。任何真实 casting episode 必须先把能力补齐并诚实更新 `casting_c1_capabilities()`，否则 `assert_backend_can_start_task` 在 `reset` 最早处 fail closed。
- `CastingEvaluator` 仍只能从 `FakeEnvironmentBackend` 的 `set_casting_evaluation_state` / `get_casting_evaluation_state` 接收 casting truth。真实 MineRL 仍没有把 `target_cell` / 流体 / `target_update_evidence` 接到 `get_evaluation_state()` 的 casting-only 表面。
- R4 driver 仍只在 FakeBackend 上验证。真实 MineRL driver 需用户单独授权（按 AGENTS.md §5 单独申请 Gradle + 真实环境运行）。
- `run_orchestrator` / `build_evaluation_state` / `CastingWorldTruth` 故意只放在 `tests/test_casting_driver.py`——它们是测试编排，不是 driver / 环境的一部分。
- 真实 episode 仍禁止真实 MineRL、Gradle 和模型调用。

## 已完成

- 核心类型、动作白名单、FakeBackend 和结构化日志；
- MineRL 单角色生命周期与 Portal 环境桥接；
- Portal 框架、激活和进入下界的自动评估；
- 活动任务 `benchmark/instances/active/casting_c1_fixed.json`；
- 离线实验契约 `configs/experiments/active/casting_c1_contract.json`；
- `R2-CAPABILITY-MANIFEST`（已完成）：不可变 + JSON 可序列化 `BackendCapabilities`、workflow-aware `assert_backend_can_start_task` 门禁、`reset` 路径强制门禁、FakeBackend 与 MineRL backend 正反例测试。
- `R3-CASTING-EVALUATOR`（已完成）：不可变 + JSON 可序列化 `CastingEvaluationState` / `CastingEvaluationResult`、`CastingEvaluator` 纯函数、9 个稳定 outcome id、9 步优先级锁定的 fail-closed 规则、FakeBackend 注入/读取表面与 `episode_id` / `step_id` 校验、信息隔离与全量回归测试。
- `R4-DETERMINISTIC-CASTING-DRIVER`（当前完成）：不可变 + JSON 可序列化 `CastingC1DriverResult` / `CastingPlanStep`、封闭 R4 动作白名单、24 步固定有限 plan、`run_casting_c1_driver` 仅依赖 `Observation.visible_inventory` / `workflow_stage` / `step_id`、step / time / wait 三类硬上限、信息隔离三道闸（AST / spy / `__getattribute__`）、测试编排器 `run_orchestrator` 端到端跑过 9 个 outcome。

## 下一任务

任务：`R6-COMPLETE-PORTAL-FRAME`（尚未开始）

在固定受控场景完成门框、点火和进入下界，由 PortalEvaluator 验证。

不要实现 R7（VLM），不要改 `vendor/minerl`，不要启动 MineRL。

## R5 合同（冻结）

- 新 workflow：`casting_c3_fixed`。
- 新 task instance：`benchmark/instances/active/casting_c3_fixed.json`。
- 新 experiment config：`configs/experiments/active/casting_c3_contract.json`。
- 3 个有序目标 cell，固定直线区段：(2, 4, 3) / (3, 4, 3) / (4, 4, 3)。
- 初始库存：water_bucket=3、lava_bucket=3、cobblestone=6。
- 预算：max_environment_steps=240、max_game_time_seconds=180、max_wait_steps≤96、max_recoveries_per_action=2、total_recovery_budget≤8（默认 1 / 3）。
- evaluator outcome 封闭集合（10 个）：`success` / `in_progress` / `partial_completion` / `wrong_block` / `truth_missing` / `step_budget_exceeded` / `time_budget_exceeded` / `invalid_initial_state` / `causality_missing` / `abnormal_termination`。
- 优先级（高→低）：step_budget → time_budget → invalid_initial_state → abnormal_termination → truth_missing → in_progress → causality_missing → ordered-prefix partial_completion / wrong_block → success。
- driver 动作白名单：`equip_item` / `use_item` / `place_block` / `wait`，目标白名单 `water_bucket` / `lava_bucket` / `cobblestone`。
- 可恢复信号：仅来自公开 `Observation` 或类型受控的 `RecoverableBackendError` 异常（来自 `obsidianlink.core.types`）；`use_item` 步可携带 `recoveries_allowed` 预算，总恢复次数受 `total_recovery_budget` 硬约束。
- 不实现 R6 / R7，不启动 MineRL / Gradle / 模型 API。

## R5 交付内容

1. 一个不可变、类型严格、可 JSON 序列化的 `ContinuousCastingCellTruth`（`obsidianlink/evaluation/continuous_casting.py`），承载每个 cell 的 `target_cell` / `initial_block` / `current_block` / `water_truth` / `lava_truth` / `transition_evidence` / `relevant_action_steps`。 ✓
2. 一个不可变、JSON 可序列化的 `ContinuousCastingEvaluationState`（`episode_id` / `step_id` / `agent_id` / 有序 `cells` tuple / 因果窗口 / 终止信号 / 时间预算 / 辅助 evidence），构造时严格要求恰好 3 个冻结、有序 target cell，拒绝跨 cell 重复声明 action step，并校验所有证据不来自未来。 ✓
3. 一个不可变、JSON 可序列化的 `ContinuousCastingEvaluationResult`（`success` / `outcome` / `completed_cells` / `total_cells` / `per_cell_outcomes` / `first_failed_cell` / `blocking_conditions` / `evidence` / `failure_type` / `failure_step` / `episode_terminated` / `terminated_step` / `terminated_reason`），`as_dict()` 返回与内部状态分离、可被 `json.dumps` 序列化的快照。 ✓
4. 一个 `ContinuousCastingEvaluator` 纯函数/纯对象：相同输入产生相同结果，从不读取 Agent 文本 / 图像 / Planner 输入；AST 扫描锁住 `obsidianlink.evaluation.continuous_casting` 不引用 `MacroAction` / `VLM` / `Qwen` / `obsidianlink.drivers` / `obsidianlink.agents` / `obsidianlink.workflows`。 ✓
5. 10 个稳定 outcome id + 锁定优先级：完成非空严格有序前缀且剩余 cell 真值完整但不是黑曜石时返回 `partial_completion`；中间有空洞或零进展返回 `wrong_block`；success 只在 3 个冻结 cell 全部为 `cell_success` 且 episode 正常终止、预算未超时的情况下返回。 ✓
6. 有限因果窗口：R5 复用 R3 `causality_window_steps`（默认 4，上限 32），每个 cell 独立计算。 ✓
7. `FakeEnvironmentBackend` 暴露 `set_continuous_casting_evaluation_state` / `get_continuous_casting_evaluation_state`：与 R3 单块表面对称，identity-guarded（`episode_id` 匹配当前 task、`step_id` 匹配当前 backend step）；未注入 / 越步 / open 之前分别报错；`reset` / `step` / `close` 自动清空；`Observation.frame` / `visible_inventory` / `messages` / `workflow_stage` 不含任何 R5 真值。 ✓
8. 一个 `run_casting_c3_driver` 入口：72 步固定 plan（3 cell × 24 step），只使用公共动作协议中的 R5 白名单；每个 `use_item` 步携带 `recoveries_allowed`；不调用任何模型代码、不打开 shell、不接 VLM。 ✓
9. 一个 `CastingC3DriverResult`（`status` ∈ {`completed` / `blocked` / `failed`}、有限计数、events 带 `episode_id` / `agent_id` / `step_id` / `cell_index` / `label` / `phase` / `action_type` / `target` / `relevant_action` / `attempt`、per_cell_relevant_action_steps、recovery 计数、终止标志 + 阻断原因、recovery metadata）；`as_dict()` 提供分离的 JSON 快照。 ✓
10. 三类硬上限 + 一个计划上限 + 两个恢复上限：`max_environment_steps` / `max_game_time_seconds` / `max_wait_steps` / `plan length` / `max_recoveries_per_action` / `total_recovery_budget`；`max_environment_steps` / `max_game_time_seconds` 不得超过 task 合同，调用方只能收紧不能放宽；budget 耗尽以 `status="blocked"` 报告。 ✓
11. 恢复协议：driver 捕获类型受控的 `RecoverableBackendError`，递增 `recovery_attempts` 与该步 `attempt`，按预算决定是否重试；预算不足时 `status="blocked"`，event `budget_exceeded: "recovery"` / `"recovery_per_step"`；`RuntimeError` / `OSError` / `TypeError`（非 `RecoverableBackendError` 子类）依旧 fail closed。 ✓
12. 一个信息隔离的 source 守卫：driver 永不 `import` `ContinuousCastingEvaluator` / `ContinuousCastingEvaluationState` / `ContinuousCastingEvaluationResult` / `ContinuousCastingCellTruth` / `ContinuousCastingWorldTruth` / 任何 `OUTCOME_*` 常量（AST 扫描测试锁定）；永不调用 `backend.set_continuous_casting_evaluation_state` / `backend.get_continuous_casting_evaluation_state`（spy 测试锁定）；永不读取 `Observation` 上任何 `target_cell` / `target_cells` / `initial_block` / `current_block` / `water_truth` / `lava_truth` / `transition_evidence` / `relevant_action_steps` / `casting_*` / `success` / `outcome` / `failure_type` / `per_cell_outcomes` / `first_failed_cell` / `completed_cells` 字段（monkey-patch 测试锁定）。 ✓
13. 一个测试侧编排器（`tests/test_continuous_casting_driver.py` 内的 `run_orchestrator` / `_state` / `ContinuousCastingWorldTruth`）：唯一允许调用 `set_continuous_casting_evaluation_state` / `get_continuous_casting_evaluation_state` 和 `ContinuousCastingEvaluator` 的位置；从 `driver_result.per_cell_relevant_action_steps` 与 `ContinuousCastingWorldTruth` 构造 `ContinuousCastingEvaluationState`。 ✓
14. 完整端到端覆盖：3-cell success、只完成第 1 / 前 2 个 cell → `partial_completion`、中间 cell wrong_block、严格三格/固定顺序、跨 cell 重复动作 step 在 state 构造时拒绝、缺 water / lava / transition 证据、transition 早于相关动作、transition 超出因果窗口、reset 时 cell 已是 obsidian、step/time budget 超限、abnormal termination、提前 terminated / truncated、有限恢复、step_id 跳跃 / stale、replay 稳定、as_dict 深度分离和信息隔离。 ✓
15. CLI 离线 contract check 与 `scripts/check_environment.py` 阶段都推进到 `reset_5_continuous_casting`，并在 CLI 报告 `r5.c3_driver_status="completed"`、`r5.c3_evaluator_outcome="success"`、`r5.c3_evaluator_completed_cells=3`。 ✓

## R5 已完成

### Driver 公开表面（`obsidianlink/drivers/casting_c3.py`）

- `ALLOWED_C3_ACTION_TYPES`（frozenset，4 项）+ `ALLOWED_C3_TARGETS`（frozenset，3 项）：所有计划动作都能通过项目公共 `parse_macro_action`。
- `DRIVER_STATUS_*`（`completed` / `blocked` / `failed`）：driver 永不返回 `passed` / `success`——verdict 由 evaluator 决定。
- `PHASE_*`（`prepare` / `place_support` / `place_lava` / `place_water` / `wait_for_obsidian` / `recovery`）：`recovery` 仅由 driver 内部事件使用，plan builder 不发射。
- `ContinuousCastingPlanStep`（`cell_index` / `label` / `phase` / `action` / `relevant_action` / `recoveries_allowed`）：`__post_init__` 严格校验 cell_index 非负、phase 封闭、recoveries_allowed 在 `[0, MAX_RECOVERIES_PER_ACTION]`、relevant_action 与 `place_block` / `use_item` 一致。
- `build_continuous_casting_action_plan(*, cell_count, support_block_wait_steps, fluid_settle_wait_steps, obsidian_wait_steps, recoveries_per_use_item) -> tuple[ContinuousCastingPlanStep, ...]`：默认 72 步（3 cell × 24 step），固定顺序，wait 计数受 `MAX_C3_PLAN_WAIT_STEPS=96` 约束，参数严格类型校验。
- `run_casting_c3_driver(backend, task, *, plan, max_wait_steps, max_environment_steps, max_game_time_seconds, total_recovery_budget, recoveries_per_use_item, event_sink) -> CastingC3DriverResult`：driver 入口；按计划调 `backend.step()`，每步前置 inventory / allowlist / step / wait / 恢复预算 检查，后置严格校验 episode 与 step 必须恰好递增 1；提前 termination/truncation / 超限 / `RecoverableBackendError` 超预算 / 非 `RecoverableBackendError` 异常 都会 fail closed。

### Driver 永不触碰 evaluator-only 表面

- AST 扫描（`test_driver_source_does_not_import_continuous_evaluator`）锁定 `obsidianlink/drivers/casting_c3.py` 顶层 `import` / `import from` 集合不含 `ContinuousCastingEvaluator` / `ContinuousCastingEvaluationState` / `ContinuousCastingEvaluationResult` / `ContinuousCastingCellTruth` / `ContinuousCastingWorldTruth` / 任何 `OUTCOME_*` 常量；模块内任何 `Attribute` 节点都不拼出 `set_continuous_casting_evaluation_state` / `get_continuous_casting_evaluation_state`。
- Spy 测试（`test_driver_does_not_call_casting_truth_surface`）在 backend 上 monkey-patch 这两个方法，断言 driver 整个生命周期调用次数为 0。
- `Observation.__getattribute__` 守卫（`test_driver_does_not_leak_truth_into_observation`）拦截任何对 `target_cell` / `target_cells` / `initial_block` / `current_block` / `water_truth` / `lava_truth` / `transition_evidence` / `relevant_action_steps` / `casting_*` / `success` / `outcome` / `failure_type` / `per_cell_outcomes` / `first_failed_cell` / `completed_cells` 的访问，driver 跑通完整 plan 不触发。
- 截取所有 driver 期间的 `Observation`（`test_all_driver_observations_are_clean`）逐个检查 `hasattr(observation, f)` 与 `observation.frame` 中都不含上述 forbidden 字段。

### Driver 固定动作流程（默认 plan 72 步）

| step_id 区间 | 动作 | phase | cell_index | `relevant_action` |
|---|---|---|---|---|
| 1 | `equip_item(lava_bucket)` | `prepare` | 0 | False |
| 2 | `wait` | `prepare` | 0 | False |
| 3 | `place_block(cobblestone)` | `place_support` | 0 | True |
| 4 | `wait` | `place_support` | 0 | False |
| 5 | `place_block(cobblestone)` | `place_support` | 0 | True |
| 6 | `wait` | `place_support` | 0 | False |
| 7 | `equip_item(lava_bucket)` | `place_lava` | 0 | False |
| 8 | `wait` | `place_lava` | 0 | False |
| 9 | `use_item(lava_bucket)` | `place_lava` | 0 | True |
| 10–13 | `wait` × 4 | `place_lava` | 0 | False |
| 14 | `equip_item(water_bucket)` | `place_water` | 0 | False |
| 15 | `wait` | `place_water` | 0 | False |
| 16 | `use_item(water_bucket)` | `place_water` | 0 | True |
| 17–20 | `wait` × 4 | `place_water` | 0 | False |
| 21–24 | `wait` × 4 | `wait_for_obsidian` | 0 | False |
| 25–48 | (cell 1 镜像 cell 0) | … | 1 | … |
| 49–72 | (cell 2 镜像 cell 0) | … | 2 | … |

`per_cell_relevant_action_steps` 在 FakeBackend 上稳定为 `{0: (3, 5, 9, 16), 1: (27, 29, 33, 40), 2: (51, 53, 57, 64)}`；orchestrator 把它按 cell 注入 evaluator 的 `ContinuousCastingCellTruth.relevant_action_steps`。

### 恢复协议（确定性、有限、公开信号）

- **可恢复信号**：`backend.step()` 抛 `obsidianlink.core.types.RecoverableBackendError`（带 `recoverable_kind: str`、`attempt: int`）。其他 `RuntimeError` / `OSError` / `TypeError` 不被识别为可恢复，依旧 fail closed。
- **per-step 预算**：每个 `use_item` plan 步的 `recoveries_allowed` ∈ [0, 2]；driver 内部用 `attempt` 计数，第二次 raise 触发 `per-step recovery budget exhausted`。
- **总预算**：`total_recovery_budget` ∈ [0, 8]；`recovery_attempts > total_recovery_budget` 触发 `recovery budget exhausted`。
- **重试动作**：与原步完全相同的 `MacroAction`，重新过白名单和预算检查；重试前不打 step 预算（被异常消耗的 step_id 不算 backend step）。
- **不允许无界重试**：`while not submitted` 严格受 `attempt` 与 `recovery_attempts` 控制；无 `for` 重试外层。
- **不允许执行模型代码**：`run_casting_c3_driver` 没有任何 `eval` / `exec` / `subprocess` 入口；`event_sink` 接收的是 detached snapshot。

### 如何保证有限循环与信息隔离

- **动作白名单 + 严格类型/数值限制**：`ALLOWED_C3_ACTION_TYPES` / `ALLOWED_C3_TARGETS` 是封闭 `frozenset`；`build_continuous_casting_action_plan` 构造时调 `_require_c3_action` 逐条 plan 步验证；`run_casting_c3_driver` 在每次 `step` 调用前再校验一次 plan step 的 action + target + duration。
- **总步数硬上限**：默认读取 task 的 240 step，调用方只能收紧不能放宽；在提交下一动作前检查，step 240 的预算绝不会执行 step 241。
- **总时间硬上限**：`max_game_time_seconds` 默认 180.0，构造时拒绝 0 / 负数 / `inf` / `nan`；每步后置检查 `next_observation.timestamp - reset_timestamp > max_game_time_seconds`（**elapsed time 不是 wall-clock**），超出立即 `status="blocked"`，event `budget_exceeded: "time"`。
- **等待/计划硬上限**：`max_wait_steps <= 96` 且在 wait 提交前检查；计划构建器也在分配 tuple 前检查总 wait，不接受超大输入。
- **提前终止**：后端在完整计划前返回 `terminated` / `truncated` 时，driver 返回 `blocked` 并原样记录标志，不再误报 `completed`。
- **恢复预算**：`total_recovery_budget` 与 `recoveries_per_use_item` 都在 plan 执行前校验，driver 内部不允许 `attempt` 超过 `recoveries_allowed + 1`。
- **无重试 / 无循环**：plan 是固定 tuple，driver 仅 `for plan_step in plan` 单次遍历；遇到 `step.terminated=True` 或后端抛 `RuntimeError` / `OSError` / `TypeError` 立即 `break` 并设 `blocked_reason`。
- **信息隔离三道闸**：(a) driver 不 import 连续 evaluator 模块（AST 扫描），(b) driver 不调 `set/get_continuous_casting_evaluation_state`（spy），(c) driver 不读 `Observation` 上的 casting 字段（`__getattribute__` 守卫 + 截取所有 observation 静态扫描）。evaluator 表面只在测试编排器 `run_orchestrator` / `_state` / `ContinuousCastingWorldTruth` 内出现。

### 测试结果（R5 及审查修复新增 116 个 + R4 保留 286 个 = 402 个）

```text
python -m obsidianlink --check                              → status=ok, phase=reset_5_continuous_casting, r4.driver_status=completed, r4.driver_success_outcome=success, r5.c3_driver_status=completed, r5.c3_evaluator_outcome=success, r5.c3_evaluator_completed_cells=3
python scripts/check_environment.py                         → project_files 全部存在, phase=reset_5_continuous_casting
python -m unittest tests.test_continuous_casting_evaluation -v → Ran 56 tests — OK
python -m unittest tests.test_continuous_casting_driver -v     → Ran 56 tests — OK
python -m unittest tests.test_casting_evaluation -v             → Ran 63 tests — OK (无回归)
python -m unittest tests.test_casting_driver -v                 → Ran 43 tests — OK (无回归)
python -m unittest tests.test_cli -v                            → Ran 1 test — OK
python -m unittest discover -s tests -p 'test_*.py'             → Ran 402 tests in 54.181s — OK
git diff --check                                                → 干净
git status --short -- vendor/minerl                             → 空（vendor 未修改）
```

### R5 新增测试覆盖范围

`tests/test_continuous_casting_evaluation.py` 共 56 个用例；新增严格三格与冻结目标顺序反例，并保留全部 outcome、隔离和 FakeBackend 状态测试：

- `OutcomeContractTests`（3）—— outcome 常量互不相同、覆盖必需集合、`partial_completion` id 稳定。
- `StateImmutabilityTests`（14）—— 不可变；空 / 少于或多于三格 / 重复 target / 冻结目标顺序 / 未来 step / 非 cell truth / terminated/time/budget 参数 / result 快照。
- `DeterminismTests`（2）—— 相同 state 重复 evaluate 完全相同；混合失败时优先级稳定。
- `SuccessPathTests`（2）—— 3-cell success；causality delta=0 仍 success。
- `PartialCompletionTests`（5）—— 只第 1 / 前 2 / 错位 cell 等多 cell 混合的 outcome 路径。
- `WrongBlockTests`（1）—— 中间 cell wrong_block 的 cell-N 条件格式。
- `TruthMissingTests`（4）—— 单 cell 缺 water / lava / transition / 借其他 cell 的 relevant actions。
- `CausalityTests`（5）—— transition 早于动作 / 超窗口 / 缺水 / 缺熔岩 / transition 非 obsidian。
- `ResetStateTests`（1）—— reset 时 cell 已是 obsidian → invalid_initial_state。
- `InProgressTests`（1）—— 未终止 → in_progress 且 per_cell_outcomes 全 `cell_not_evaluated`。
- `AbnormalTerminationTests`（2）—— 非 NORMAL 原因 → abnormal_termination；NORMAL 原因 → 透传到 per-cell。
- `BudgetOutcomeTests`（2）—— step / time budget 超限。
- `FakeBackendContinuousCastingStateTests`（8）—— set/get 正常路径；未注入 / wrong episode_id / wrong step_id / 非 state 类型 / step 清空 / close 清空 / reset+step 路径 observation 不漏 R5 真值。
- `EvaluatorIsolationTests`（2）—— 源码不引用 `MacroAction` / `VLM` / `Qwen` / `obsidianlink.drivers` / `obsidianlink.agents` / `obsidianlink.workflows`；签名只接收 `ContinuousCastingEvaluationState`。

`tests/test_continuous_casting_driver.py` 共 56 个用例，分布在 7 个 TestCase 中：

- `DriverContractTests`（13）—— 动作 / 目标 / 状态白名单；plan 长度 + 12 个相关动作；phase 集合；allowlist 全覆盖；recoveries_allowed 边界；`parse_macro_action` 公共协议兼容；参数校验；`cell_count` / `recoveries_per_use_item` / `total_waits` 边界。
- `DriverArgumentValidationTests`（9）—— task 类型 / 错误 workflow / max_wait_steps / max_environment_steps / max_game_time_seconds / total_recovery_budget / plan 长度上限。
- `FakeBackendDriverTests`（13）—— 完整 plan；event step_id 严格递增；cell_index 标对；step / time / wait budget；早期 termination / truncation fail closed；evidence 不可变；event_sink 不可改 evidence；缺 cobblestone；replay 稳定；不调 evaluator truth 表面；不漏 truth 到 observation。
- `RecoveryProtocolTests`（7）—— 可恢复异常一次后成功；总预算耗尽；per-step 预算耗尽；recoverable metadata 写进 event；恢复与 step budget 解耦；非可恢复异常 fail closed；`total_recovery_budget=0` 仍可跑。
- `OrchestratorOutcomeTests`（6）—— 3-cell success；only cell 0；first two cells；中间 wrong_block；truth_missing via orchestrator；invalid_initial_state；partial_completion 路径（被 per-cell truth 压制）。
- `StaleStepAndReplayTests`（4）—— step 跳跃拒绝；event step_id 唯一且单调；replay+orchestrator 稳定；evaluator 用 driver 的 per-cell actions。
- `ObservationLeakageTests`（2）—— driver 后 + orchestrator 后 observation 不含 R5 真值；驱动跑期间所有 observation 静态扫描均无 R5 真值。
- `DriverBackendShapeTests`（1）—— 最小 backend（仅 `reset` / `step`）可被 driver 接受并跑完完整 plan。

合计 R5 evaluator/driver 单测 112 个；另新增 3 个 capability gate 与 1 个 benchmark 文件合同测试；R4 之前 286 个；总计 402 个离线用例全部通过。

### 本轮没有启动以下任何一项

- Minecraft / MineRL / Gradle / 付费模型 API；
- `vendor/minerl` 未修改（`git status --short -- vendor/minerl` 为空）；
- 未实现 R6 完整门框 / 点火 / 进入下界 / VLM 接入；
- 未生成 `runs/` 真实运行证据；
- 未 `git commit`、未 `git push`；
- 未修改固定依赖版本。

## 当前限制

- 真实 MineRL backend 仍缺 7 项能力（详见 R2 段）。任何真实 casting episode 必须先把能力补齐并诚实更新 `casting_c1_capabilities()`，否则 `assert_backend_can_start_task` 在 `reset` 最早处 fail closed。R5 的 `casting_c3_fixed` 任务对应的 capability 清单与 R3 一致；真实 MineRL 上线前需要先补齐。
- `CastingEvaluator` 与 `ContinuousCastingEvaluator` 仍只能从 `FakeEnvironmentBackend` 的对应 truth 表面接收真值。真实 MineRL 仍没有把目标 cell / 流体 / `target_update_evidence` 接到 `get_evaluation_state()`。
- R5 driver 仍只在 FakeBackend 上验证。真实 MineRL driver 需用户单独授权（按 AGENTS.md §5 单独申请 Gradle + 真实环境运行）。
- `run_orchestrator` / `_state` / `ContinuousCastingWorldTruth` 故意只放在 `tests/test_continuous_casting_driver.py`——它们是测试编排，不是 driver / 环境的一部分。
- 当前后端默认不抛 `RecoverableBackendError`，driver 的恢复协议是"准备好但不被触发"。测试通过 fault-injection backend 子类验证恢复逻辑。
- 真实 episode 仍禁止真实 MineRL、Gradle 和模型调用。

## 下一 Agent 直接执行

> R5 已完成离线实现并修复合同审查问题：`casting_c3_fixed` 严格要求 3 个冻结、有序目标 cell；跨 cell 不能重复声明相关动作 step；完成非空有序前缀返回 `partial_completion`，只有三格全部成功才返回 `success`；`casting_c3_fixed` 已纳入 capability fail-closed 门禁。
> 下一任务是 R6 完整门框与点火。不要重做 R3 / R4 / R5，不要启动 MineRL / Gradle / 模型 API，除非用户另行明确授权。

## R3 历史交付内容

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
- R3 阶段当时未实现 R4 driver；R4 现已完成离线确定性 driver，但仍未实现真实水/熔岩模拟或 VLM 接入；
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

## R3 阶段限制（历史）

- 真实 MineRL backend 仍缺 7 项能力（见上表）。任何真实 casting episode 必须先把能力补齐并诚实更新 `casting_c1_capabilities()`，否则 `assert_backend_can_start_task` 会 fail closed。
- 当前 evaluator 只能从 `FakeEnvironmentBackend` 的 `set_casting_evaluation_state` / `get_casting_evaluation_state` 接收 casting truth。真实 MineRL 仍没有把 `target_cell` / 流体 / `target_update_evidence` 接到 `get_evaluation_state()` 的 casting-only 表面。
- R3 阶段的 `casting_c1_fixed` 还没有 driver；该限制已由 R4 的离线确定性 driver 消除。
- 真实 episode 仍禁止真实 MineRL、Gradle 和模型调用。

## R3 阶段已完成（历史）

- 核心类型、动作白名单、FakeBackend 和结构化日志；
- MineRL 单角色生命周期与 Portal 环境桥接；
- Portal 框架、激活和进入下界的自动评估；
- 活动任务 `benchmark/instances/active/casting_c1_fixed.json`；
- 离线实验契约 `configs/experiments/active/casting_c1_contract.json`；
- `R2-CAPABILITY-MANIFEST`（已完成）：不可变 + JSON 可序列化 `BackendCapabilities`、workflow-aware `assert_backend_can_start_task` 门禁、`reset` 路径强制门禁、FakeBackend 与 MineRL backend 正反例测试。
- `R3-CASTING-EVALUATOR`（已完成）：不可变 + JSON 可序列化 `CastingEvaluationState` / `CastingEvaluationResult`、`CastingEvaluator` 纯函数、9 个稳定 outcome id、9 步优先级锁定的 fail-closed 规则、FakeBackend 注入/读取表面与 `episode_id` / `step_id` 校验、信息隔离与全量回归测试。
