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

### R6：完整门框、点火和进入 Nether（下一工程任务）

在固定受控场景完成有效门框、点火和 Nether entry，由独立 evaluator 验证。

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
