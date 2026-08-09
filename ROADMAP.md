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

### R6-COMPLETE-PORTAL-FRAME — CONTRACT FREEZE（合同冻结完成；C3 evaluator + driver 已离线验证；C4 / C5 仍未实现）

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

### R6：完整门框、点火和进入 Nether（按子阶段推进）

R6 将在 R6-C3 / R6-C4 / R6-C5 三阶段合同冻结基础上，由独立 evaluator 验证完整门框、点火和 Nether entry，并依次在 FakeBackend 上接确定性 driver；接真实 MineRL 与模型仍需要单独授权。

### R7：模型与受控变化

确定性路径稳定后接入受安全约束的模型，并引入有限、可审计的场景变化。

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
