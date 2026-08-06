# 当前状态

更新时间：2026-08-06

## 当前唯一目标

任务：`R6-COMPLETE-PORTAL-FRAME — CONTRACT FREEZE`（C3 / C4 / C5 合同已冻结，C3 frame evaluator + FakeBackend truth path 在本轮已离线验证；C3 driver、C4/C5 evaluator、真实 MineRL 仍未实现）

下一任务：`R6-C3-DETERMINISTIC-DRIVER`（C3 deterministic driver；只在上一轮 C3 frame evaluator + FakeBackend truth path + 全部离线回归真正完成后才能启动）

当前 active implementation 仍是 Casting-S-C2 / fixed 的 `casting_c3_fixed`。旧 ID 中的 `c3` 表示三个 cell，不表示 B0 taxonomy 的 C3；文档级兼容名称为 `casting_s_c2_fixed`。`casting_c1_fixed` 保留为 Casting-S-C1 回归合同。

R6 的 Casting-S-C3 / C4 / C5 任务合同**已经冻结**（catalog 可见、taxonomy 正确、scenario_parameters 显式、live_run_allowed=false），并且 R6-C3-FRAME-EVALUATOR 子阶段在 FakeBackend 上完成了 C3 frame evaluator + task-origin / truth-grid 坐标锚定；C3 driver、C4 ignition evaluator、C5 Nether-entry evaluator、真实 MineRL 接入、Gradle、模型 API 仍然不在本阶段范围。


## B1 已完成

1. 新增权威 [`benchmark/catalog/tasks.json`](benchmark/catalog/tasks.json)，统一 canonical name、compatibility ID、task instance ID、workflow、taxonomy、实例/实验路径、实现状态、发布可见性和 live-run policy。
2. 新增严格 [`TaskCatalog`](obsidianlink/core/task_catalog.py) loader：frozen 类型、未知字段拒绝、family/level 匹配、唯一性、安全相对路径和 active entry 约束全部 fail closed。
3. `validate_catalog_references()` 校验所有 task/experiment 路径存在，task ID、workflow、taxonomy、compatibility name、实验引用和 live-run policy 相互一致。
4. 现有 `casting_c1_fixed` / `casting_c3_fixed` 保持原 ID、workflow 和路径；正式分类分别是 Casting-S-C1 / C2。
5. `route_a_a0_development` / `route_a_a0_phase3` 保持历史路径，但明确分类为 `calibration`、`benchmark_visible=false`、`legacy_regression`，不得混入正式 Benchmark 指标。
6. CLI `--check` 与 `scripts/check_environment.py` 从 catalog 解析 active task 和 taxonomy，不再各自硬编码分类，并验证 catalog 的所有文件引用。
7. 新增 [TASK_REGISTRY.md](docs/architecture/TASK_REGISTRY.md)，冻结 R6 起的 canonical 目录规则和历史迁移边界。
8. B1 没有移动历史文件，没有修改 evaluator、driver、backend、任务语义或依赖。

## R5 冻结合同

- workflow / task ID：保留历史兼容 ID `casting_c3_fixed`；
- taxonomy：family=`casting`、mode=`single`、level=`C2`、layout=`fixed`；
- 目标：三个冻结、有序 cell `[2,4,3]`、`[3,4,3]`、`[4,4,3]`；
- 初始资源：water_bucket=3、lava_bucket=3、cobblestone=6；
- task 预算：240 environment steps、180 秒 game time、最多 1 次 model call；
- driver：默认 72 步固定 plan、最多 96 wait、per-action recovery≤2、total recovery≤8；
- 动作白名单：`equip_item` / `use_item` / `place_block` / `wait`；
- outcome：`success` / `in_progress` / `partial_completion` / `wrong_block` / `truth_missing` / `step_budget_exceeded` / `time_budget_exceeded` / `invalid_initial_state` / `causality_missing` / `abnormal_termination`；
- 世界变化只能来自 Agent 白名单动作和原版水/熔岩方块更新。

完整任务规则见 [`casting_c3_fixed` 任务页](docs/tasks/casting/casting_c3_fixed.md)。

## R5 已完成

1. `ContinuousCastingCellTruth`、`ContinuousCastingEvaluationState`、`ContinuousCastingEvaluationResult` 和 `ContinuousCastingEvaluator` 均为不可变、类型严格、可序列化的 evaluator-only 表面。
2. Evaluator 严格要求三个冻结、有序 cell；每个 cell 的 relevant actions、水、熔岩和 transition 证据独立归属，拒绝跨 cell 重复 action step。
3. 完成非空严格有序前缀返回 `partial_completion`；中间空洞或零进展不能冒充成功。
4. `run_casting_c3_driver` 使用公共 `MacroAction`、封闭白名单、有限计划和有限预算，不调用模型、不执行代码或命令。
5. 恢复只响应类型受控的 `RecoverableBackendError`，同时受 per-action 和 total recovery 硬上限约束；其他异常 fail closed。
6. Driver 不 import continuous evaluator、不调用 truth set/get 表面、不读取 observation 上的 target/truth/outcome 字段；专项 AST、spy 和 observation guard 测试锁定隔离。
7. FakeBackend truth 注入接口校验 `episode_id`、`step_id`、`agent_id`，并在 reset/step/close 后清空陈旧状态。
8. 任务实例、CLI 和环境检查现已显式报告 Casting-S-C2 taxonomy，同时保留 `casting_c3_fixed` 兼容 ID。
9. C1/C2 任务页、README、BENCHMARK_SPEC、ROADMAP、DATASET_CARD 和 taxonomy 已与实际离线能力统一。

## R6-C3-FRAME-EVALUATOR 已完成（FakeBackend 离线证明）

1. **新增 C3 frozen-frame evaluator** [`obsidianlink/evaluation/casting_frame_evaluator.py`](obsidianlink/evaluation/casting_frame_evaluator.py)：
   - `FrozenFrameActionEvidence` / `FrozenFrameCellTruth` / `FrozenFrameInteriorCellTruth` / `FrozenFrameEvaluationState` / `FrozenFrameEvaluationResult` / `FrozenFrameEvaluator` 都是不可变、类型严格、可序列化的 evaluator-only 表面；动作证据绑定 episode / step / Agent / target cell，并只允许水桶或熔岩桶的 `use_item`；
   - 14 个 target cell 顺序与 [`casting_s_c3_fixed.json`](benchmark/instances/casting/single/casting_s_c3_fixed.json) 的 `public_task_spec.frame_plan.fixed_offsets` 严格一致；6 个 interior cell 同样在合同中冻结；
   - 14 cell 全部 obsidian、6 interior cell 全部在 `INTERIOR_ALLOWED` 内、且因果窗口合法 ⇒ `success`；
   - `partial_completion` 覆盖任意无序的 1–13 cell 完成子集，只要其余目标仍为 `air` / `water` / `lava`；缺四角的 vanilla 10-cell 形式因此判为 partial，绝不冒充 success；真正的阻挡方块仍判为 `wrong_block`；
   - 闭集 outcome：`success` / `in_progress` / `partial_completion` / `wrong_block` / `truth_missing` / `step_budget_exceeded` / `time_budget_exceeded` / `invalid_initial_state` / `causality_missing` / `abnormal_termination` / `interior_blocked`；
   - 优先级与 R3 / R5 对齐：truth_missing 和 interior_blocked 都 outrank in_progress；budget / invalid_initial_state / abnormal_termination 仍最先判定；
   - `causality_window_steps` 默认 4（与 R3 合同一致），最大 32；任何 out-of-window transition 直接 `causality_missing`；
   - 优先级由测试 `test_priority_is_stable_for_same_input` 锁定；`evaluate()` 重复调用产生完全相同的 `FrozenFrameEvaluationResult` 与 `as_dict()` 快照。
2. **task-origin / truth-grid 坐标锚定**：
   - `FrozenFrameOriginAnchor` 是纯函数、不可变、类型严格的转换器，签名只接收 `(task_origin_relative_offset)`；
   - 默认 `default_c3_anchor()` 把 task-origin 标记对齐到 grid 原点 `(0, 0, 0)`，grid 范围沿用 `obsidianlink.env.portal_spec.PORTAL_GRID_MIN/MAX` (`(-3,-1,0)`–`(3,5,6)`)；
   - 14 个公开 cell 落点 `x=0..3` / `y=0..4` / `z=1`，完全在现有 truth grid 数值范围内；越界、缺失 origin、类型错误、bool 混入、grid 边界反向（`grid_min > grid_max`）全部 fail closed；
   - 不会为了本轮合同扩展 grid；不会修改 MineRL `PortalA0EnvSpec`。
3. **FakeBackend evaluator-only truth 注入路径** [`obsidianlink/env/fake.py`](obsidianlink/env/fake.py)：
   - 新增 `_frame_evaluation_state` 槽位，与 `_casting_evaluation_state`（C1）和 `_continuous_casting_evaluation_state`（C2）严格隔离；
   - `set_frame_evaluation_state` 只接受 `casting_s_c3_fixed` workflow，并校验类型、`episode_id` 与当前 task 一致、`step_id` 与当前 backend step 一致、`agent_id` 必须在 `task.agent_ids` 内，否则 fail closed；
   - `get_frame_evaluation_state` / `clear_frame_evaluation_state` 显式可调用；
   - `reset()` / `step()` / `close()` 一律清空该槽位，杜绝跨 step 的 truth 泄漏。
4. **C1 / C2 / portal 回归不受影响**：
   - R3 `CastingEvaluationState` + `CastingEvaluator` 测试套件全绿；
   - R5 `ContinuousCastingEvaluationState` + `ContinuousCastingEvaluator` 测试套件全绿；
   - `PortalEvaluator` + `EvaluationState` 测试套件全绿；
   - FakeBackend 三个 truth 槽位独立存在；注入 C3 state 不会让 C1 / C2 slot 出现 truth。
5. **信息隔离**：
   - 整个 evaluator 源文件不 import `obsidianlink.agents` / `obsidianlink.workflows` / `obsidianlink.drivers`（AST 检查锁定）；
   - 不读取 `scenario_parameters` / `evaluator_contract` / `instruction`（AST 检查锁定）；
   - `evaluate()` 唯一参数是 `state: FrozenFrameEvaluationState`；
   - FakeBackend 生成的 `Observation`（`step_id=0` 与 `step_id=1`）都不携带任何 frame / cell / outcome / 归因 truth，公开 schema 字段集严格不变（`episode_id` / `agent_id` / `step_id` / `timestamp` / `frame` / `visible_inventory` / `messages` / `workflow_stage`）；
   - C3 frame evaluator 103 个专项测试通过；全量 539 个离线测试通过。
6. **公开 / 隐藏边界**：
   - 公开 `public_task_spec` 可以进入未来公开上下文（instruction 已显式声明 `task-origin marker` / `plane_z` / `[0,0,1]` / `width 4` / `height 5` / `14 obsidian cells`）；
   - `evaluator_contract` 仍是 policy-only（baseline_policy / required_mechanism / required_items / causality_window_steps / fail_closed_on_missing_truth），runtime truth 仍由 FakeBackend evaluator-only 状态独立验证；
   - 真实 MineRL 接入、task-origin marker 与 truth-grid origin 的世界坐标锚定仍需在 R6 之后 driver / backend 阶段验证；本轮只能证明 evaluator / FakeBackend 边界，不声称已端到端隔离未来 driver。

## 下一任务

`R6-C3-DETERMINISTIC-DRIVER`：在 R6-C3 frame evaluator + FakeBackend truth path + 全部离线回归完成后启动 C3 deterministic driver；仍不接通真实 MineRL，不运行 Gradle，不调用模型。

## R6-COMPLETE-PORTAL-FRAME — CONTRACT FREEZE 已完成（保留历史）

1. **审计** [`obsidianlink/evaluation/portal.py`](obsidianlink/evaluation/portal.py) 与 [`obsidianlink/evaluation/frame_geometry.py`](obsidianlink/evaluation/frame_geometry.py)：
   - 现有 frame 几何在 `frame_geometry.detect_portal_frame` 中以 `MIN_WIDTH=4 / MAX_WIDTH=23 / MIN_HEIGHT=5 / MAX_HEIGHT=23` 冻结；两个朝向 `plane_z`（X-Y 平面，Z 恒定）与 `plane_x`（Y-Z 平面，X 恒定）；
   - 4×5 frame 的 `2W + 2H - 4 = 14` 完整外周和 `2W + 2H - 8 = 10` 必需（不含角）cell 都被 `FrameCandidate` 显式表达，4 个角 cell 在 `allow_missing_corners=True` 时是可选的；
   - `interior_allowlist = {air, nether_portal, fire}`，`dirt` / `bedrock` / `grass` / `grass_block` / `other` / `missing` 全部判为阻挡；`missing` / `other` 触发 fail-closed；
   - `PortalEvaluator` 已有 latched `portal_built_by_episode` / `valid_portal_frame` / `portal_activated`、`latched_frame_identity` / `latched_activation_offsets`、`entered_via_episode_portal_by_agent` 和 `matched_frame_identity_by_agent`；激活与 Nether entry 严格绑定到本 episode 锁存的 frame identity；
   - **`is_episode_built` 已经能证明"门框由当前 episode 建造"**：`is_geometric_valid=True` 且 baseline 中没有 required cell 是 `obsidian`；`PortalA0EnvSpec` 仍在用 14 obsidian + 1 flint_and_steel + 2 dirt 的固定受控 A0 资源。
2. **新增 3 个 R6 Casting-S 任务实例**，统一放于 canonical `benchmark/instances/casting/single/`，遵循 B0 taxonomy 与 [TASK_REGISTRY.md](docs/architecture/TASK_REGISTRY.md) 目录规则：
   - `casting_s_c3_fixed.json`：提供水、熔岩和支撑方块，通过原版 block update 浇筑公开 4×5 full-ring 方案；Minecraft 最小合法计数 10 与本实例额外要求的 14-block full ring 分开冻结；
   - `casting_s_c4_fixed.json`：继承 C3，并公开冻结唯一计分点火目标 `[1,1,1]`、`use_item` 与 `flint_and_steel`；
   - `casting_s_c5_fixed.json`：继承 C4，并公开指定 `agent_1` 进入 `minecraft:the_nether`；机器合同冻结 episode-portal、frame-identity、pre-transition-position 与 transition-step 归因要求。
3. **新增 3 个 contract-only 实验配置**于 `configs/experiments/active/`，`backend="not_implemented"`、`planner="not_implemented"`、`evaluator="not_implemented"`、`status="contract_only"`、`max_real_runs=0`。
4. **更新 [catalog](benchmark/catalog/tasks.json)**：新增 3 个 `kind=benchmark` 条目（`implementation_status="contract_only"`、`benchmark_visible=true`、`live_run_allowed=false`），保持 `active_compatibility_id="casting_c3_fixed"`（C2 仍为实际实现的 active slice，C3/C4/C5 仅为合同冻结）。
5. **新增 3 个任务页** [C3](docs/tasks/casting/casting_s_c3_fixed.md) / [C4](docs/tasks/casting/casting_s_c4_fixed.md) / [C5](docs/tasks/casting/casting_s_c5_fixed.md)，明确公开任务规则与 evaluator-only 运行时真值边界、success / failure 边界、合同复用与未实现项。
6. **更新 [TASK_REGISTRY.md](docs/architecture/TASK_REGISTRY.md)**：目录规则与路径示例已与 R6 canonical 一致；calibration `route_a_a0_*` 仍保持历史路径与 `legacy_regression` 分类。
7. **更新 [TASK_TAXONOMY.md](docs/benchmark/TASK_TAXONOMY.md)**：C3/C4/C5 里程碑表与文档级命名规则保持与 B0 一致；明确说明 R6 合同已冻结但 evaluator/driver 未实现。
8. **更新 [README.md](README.md) / [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md) / [DATASET_CARD.md](DATASET_CARD.md) / [ROADMAP.md](ROADMAP.md)**：把"当前阶段 = R6 合同冻结"和"C3/C4/C5 仅为合同、未实现"显式标注，不声称 Casting 端到端已支持。
9. **新增离线合同测试** [tests/test_r6_casting_c3c4c5_contract.py](tests/test_r6_casting_c3c4c5_contract.py)：验证 Casting 资源/机制、公开 frame plan、C4 精确点火、C5 机器归因合同、公开/隐藏命名空间和现有 Agent 源码不读取 `scenario_parameters`，并验证 `route_a_a0_*` 仍为 calibration。R6 runtime 尚未实现，因此不能声称已验证未来 driver 的运行时隔离。

## R6 已复用 / 未实现

复用：

- `frame_geometry.FrameCandidate` / `detect_portal_frame` 字段语义作为合同字段语义基准；
- `obsidianlink.env.portal_spec.PORTAL_GRID_BLOCKS` 的 truth 编码与 `FrameCandidate` 几何语义；`PortalA0EnvSpec.PORTAL_INVENTORY` 的预置黑曜石仅属于历史 calibration，不复用于正式 Casting C3–C5；
- `EvaluationState` 全部字段语义（包括 latched booleans / frame identity / activation offsets / entered-via-portal 归因 / 6 个稳定 failure types）；
- `obsidianlink.env.portal_spec.PortalTransitionObservation` 提供的 `entered_via_portal` / `from_dimension` / `to_dimension` server-side truth 作为合同承认的上游 evaluator-only 真值。

未实现（下一子任务）：

- C3 deterministic driver 与 evaluator truth 采集编排；
- C4 ignition evaluator（把 `use_item(flint_and_steel)` 与 `[1,1,1]` 内框 cell 的 `nether_portal` 关联到 `latched_frame_identity`）；
- C5 Nether entry evaluator（绑定 `pre_transition_position_by_agent` 与 `latched_frame_identity` 的因果链）；
- 任意 C3/C4/C5 deterministic driver；
- 真实 MineRL 后端接通桶动作、公开选中物品、目标方块 truth、流体 truth 与维度切换 truth；
- 真实 MineRL、Gradle 与模型 API 调用。

已知 contract 局限（不在 R6 合同冻结阶段处理）：

- `PORTAL_GRID_MIN/MAX = (-3,-1,0)–(3,5,6)` 已覆盖本固定合同 x=`0..3`、y=`0..4`、z=`1`；真实 backend 接入时仍需验证 task-origin marker 与 truth-grid origin 的锚定关系；
- `obsidianlink/drivers/scripted_a0.py` 使用预置黑曜石和不同世界坐标，仍为 calibration，不冒充水/熔岩 Casting C3 合同；
- R6 runtime 尚未实现；当前隔离测试只能证明现有 Agent/Planner 代码不读取 `scenario_parameters`。实现 driver 时必须增加显式 public-context 构造与端到端泄漏测试。

## 已完成历史（保留）

### R1 — `casting_c1_fixed` 任务合同

固定 seed、Single-Agent、出生点、资源、单 target cell、里程碑、预算和禁止 evaluator/命令改世界规则已冻结。

### R2-CAPABILITY-MANIFEST — 后端能力清单

不可变 `BackendCapabilities`、稳定 capability IDs、FakeBackend 正反例、reset 前 fail-closed gate 和 JSON 快照已完成。真实 backend 缺关键能力时不得开始 casting episode。

### R3 — 单块 evaluator

`CastingEvaluationState`、`CastingEvaluationResult`、稳定 outcome、有限因果窗口和 evaluator-only truth 隔离已完成。

### R4-DETERMINISTIC-CASTING-DRIVER — 单块 driver

公共动作协议、封闭白名单、有限 plan/step/time/wait、确定性重放和 driver/evaluator 隔离已在 FakeBackend 完成。

### B0-BENCHMARK-SCOPE-FREEZE — 总范围冻结

三个 task family、两种 agent mode、C/R/A 能力层级、命名规范、难度维度、指标和数据证据协议已冻结。长期 scope 与 active implementation 分开。

### R5-CONTINUOUS-CASTING — 连续浇筑

`casting_c3_fixed` 的三 cell continuous evaluator、deterministic driver、per-cell 因果证据、部分完成和有限恢复已在 FakeBackend 完成，并映射为 Casting-S-C2。

### B1-TASK-CATALOG-FOUNDATION — 严格 Catalog

Task catalog、严格解析器、active entry 约束、calibration 分类与 C1/C2 可见性规则已在 catalog 落地；R6 合同冻结阶段继续把 C3/C4/C5 接入同一 catalog 体系。

## 当前限制

- R5 / R6 合同冻结都只在 FakeBackend 验证；真实 MineRL 浇筑与门框建造均未验证；
- 真实 backend 仍未完整接通桶动作、公开 selected item、目标方块 truth 和流体 truth；
- `casting_c3_fixed` 是 C2 连续浇筑切片，C2 success 不等于进入 Nether；
- R6 Casting-S-C3 frame evaluator 与 task-origin/truth-grid 数值锚定已离线完成，但 C3 driver、C4/C5 evaluator/driver 与真实 backend 接线均未完成；
- Ruined、Adaptive、Multi-Agent、真实 MineRL episode 集和 Benchmark 公开指标发布均未实现；
- 当前没有正式真实 Benchmark 数据；
- 禁止真实 MineRL、Gradle 和模型调用，除非用户针对每次操作单独授权；
- `vendor/minerl`、固定依赖和历史兼容 ID 在 R6 阶段均未改动。

## 测试要求

Task catalog 解析/路径/分类正反例、R5 evaluator 与 driver 专项测试、capability、benchmark file、CLI、R3/R4 回归、portal / frame geometry 旧测试必须保持通过。本轮新增的 R6 contract 测试也必须全部通过；任何合同整理不得削弱严格解析、预算、因果、兼容性或信息隔离合同。

本轮验证：`python -m obsidianlink --check` 与 `python scripts/check_environment.py` 均通过；全量 539 个离线测试（含 103 个 C3 frame evaluator 专项测试）全部通过；`git diff --check` 干净。本轮没有修改 `vendor/minerl`；该独立仓库中已有的其他工作区改动不属于本任务，保持原状。

## 下一任务

`R6-C3-DETERMINISTIC-DRIVER`：为 `casting_s_c3_fixed` 实现受限、确定性的 FakeBackend driver；不接通真实 MineRL；不运行 Gradle；不调用模型。
