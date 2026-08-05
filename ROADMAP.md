# 路线图

项目采用小步验证：先证明一块黑曜石，再扩展到门框和完整任务。

## 已完成：基础层

- 安全动作协议、FakeBackend、日志和回放；
- MineRL Portal 环境桥接；
- Portal 框架、激活和进入下界 evaluator；
- 确定性建门校准流程。

## R1：冻结真实任务契约（完成）

活动任务为 `casting_c1_fixed`：固定场景、单 Agent、用水和熔岩生成一块黑曜石。

## R2：后端能力清单（完成）

纯离线确认桶、use 动作、公开物品状态、目标方块和流体真值的接口。缺能力时必须提前失败。

完成条件：manifest、FakeBackend 正反例、reset 路径门禁、JSON 序列化与离线测试通过。详见 [PROJECT_STATUS.md](PROJECT_STATUS.md) 的 R2 段。

## R3：单块 evaluator（完成）

在 R2 已暴露的 backend 能力缺口上，为 `casting_c1_fixed` 实现 evaluator：比较动作前后目标 cell 的方块，给出 `success`、`wrong_block`、`truth_missing` 或超预算等明确结果。evaluator 必须：

- 只读取 evaluator-only 真值，不读取 Agent 文本 / 图像 / Planner 输入；
- 在 `target_block_truth` / `fluid_truth` 缺失时返回 `truth_missing`，不能误判为 `success`；
- 通过 `BackendCapabilities` 显式校验 backend 是否能提供 `exposes_target_block_truth` 等真值接口，缺则 fail closed；
- 复用现有 `EvaluationState` / `PortalEvaluator` 的失败分类结构；
- 不实现真实 driver；driver 留给 R4。

完成条件（已达成）：

- `obsidianlink/evaluation/casting.py` 提供 `CastingEvaluationState`（frozen + 类型严格 + 容器递归不可变 + `__post_init__` 跨字段时间校验）、`CastingEvaluationResult`（frozen，`as_dict()` 提供分离的 JSON 快照）、`CastingEvaluator`（纯函数，签名只接受 `CastingEvaluationState`）。
- 9 个稳定 outcome id（`success` / `in_progress` / `wrong_block` / `truth_missing` / `step_budget_exceeded` / `time_budget_exceeded` / `invalid_initial_state` / `causality_missing` / `abnormal_termination`），固定优先级（step 预算 → 时间预算 → invalid_initial_state → truth_missing → in_progress → abnormal_termination → causality_missing → wrong_block → success）。
- 有限因果窗口 `causality_window_steps`（默认 4，上限 32），方块更新必须落在最新相关动作之后且在窗口内；水/熔岩必须明确存在，transition 必须明确以 obsidian 结束，所有证据 step 不得来自未来。
- `FakeEnvironmentBackend` 暴露 `set_casting_evaluation_state` / `get_casting_evaluation_state`：`episode_id` / `step_id` 严格一致；未注入 / 越步读取 / 未 reset 读取分别报错；`reset` / `step` / `close` 自动清空；`Observation.frame` / `visible_inventory` / `messages` / `workflow_stage` 不含任何 casting truth。
- 当前真实 MineRL 仍因 7 项能力缺失 fail closed（`assert_backend_can_start_task` 在 `reset` 最早处触发，`env_factory` 不被调用）。
- 全部离线测试通过；`git diff --check` 干净；`vendor/minerl` 未修改；未启动 Minecraft / MineRL / Gradle / 模型 API。

## R4：确定性单块 driver（完成）

使用有限动作、有限等待和有限重试完成单块浇筑。先跑 FakeBackend；真实 MineRL 另行申请授权。

完成条件（已达成）：公共 `MacroAction` 协议、24 步固定计划、计划/step/time/wait 硬上限、后端提前终止 fail closed、driver/evaluator 信息隔离、FakeBackend 可重放与全量离线测试通过。

## R5：连续浇筑（完成）

把 R3 / R4 的单块扩展为短区段，验证多次液体操作、每 cell 独立因果证据、单 cell 失败不掩盖、有限恢复协议。

完成条件（已达成）：

- 任务实例 `casting_c3_fixed`（3 个有序目标 cell 的固定直线区段）已冻结在 `benchmark/instances/active/casting_c3_fixed.json`。
- 新 `obsidianlink/evaluation/continuous_casting.py` 提供 `ContinuousCastingCellTruth` / `ContinuousCastingEvaluationState` / `ContinuousCastingEvaluationResult` / `ContinuousCastingEvaluator`，所有容器都是 frozen + 递归不可变 + `__post_init__` 严格校验。
- 10 个稳定 outcome id（success / in_progress / partial_completion / wrong_block / truth_missing / step_budget_exceeded / time_budget_exceeded / invalid_initial_state / causality_missing / abnormal_termination）；`partial_completion` 表示完成非空有序前缀，中间空洞仍是 `wrong_block`。
- evaluator 严格要求 3 个冻结、有序目标 cell；每个 cell 的 `relevant_action_steps` / `water_truth` / `lava_truth` / `transition_evidence` 都是 per-cell 的，相关动作 step 不得被多个 cell 重复声明。
- 新 `obsidianlink/drivers/casting_c3.py` 提供 `run_casting_c3_driver` + 72 步固定 plan + 公共 `MacroAction` 协议 + 封闭 R5 白名单。
- 恢复协议基于公开信号：后端 `backend.step()` 抛 `RecoverableBackendError` 时 driver 在 per-step 预算和总预算内重试同一动作。预算耗尽后 fail closed，恢复动作仍经过同一白名单和预算检查。
- `FakeEnvironmentBackend` 暴露 `set_continuous_casting_evaluation_state` / `get_continuous_casting_evaluation_state`，与 R3 单块表面对称，identity-guarded，`reset` / `step` / `close` 自动清空。
- 3-cell success 可在 FakeBackend 上确定性重放；部分成功和中间失败不会误报 success；R4 / R3 / R2 全部测试无回归。
- `vendor/minerl` 未修改；未启动 MineRL / Gradle / 模型 API。
- `casting_c3_fixed` 已纳入 R2 capability gate，缺桶动作或 evaluator 真值能力时在 reset 前 fail closed。
- CLI 阶段推进到 `reset_5_continuous_casting`，`python -m obsidianlink --check` 同时验证 R4 单块合同和 R5 连续浇筑合同。
- R5 evaluator 56 个 + R5 driver 56 个，并新增 3 个 capability gate 与 1 个 benchmark 文件合同测试；R4 之前 286 个，总计 402 个离线用例全过。

## R6：完整门框与点火

在固定受控场景完成门框、点火和进入下界，并由 PortalEvaluator 验证。

## R7：模型与更完整任务

确定性流程稳定后再接入 VLM。之后才考虑资源获取、废弃传送门、随机布局和双 Agent。

每个阶段都必须有自动评估、受控证据和必要的人工复核。
