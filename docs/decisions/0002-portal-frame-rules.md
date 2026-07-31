# ADR 0002：Phase 2 地狱门框几何与里程碑规则

- 状态：已实现并冻结；与 Minecraft 1.16.5 行为一致
- 日期：2026-07-30
- 适用阶段：Phase 2；本规则同时被 `BENCHMARK_SPEC.md` 引用
- 替代关系：取代 `obsidianlink/evaluation/portal.py` 中旧的
  `portal_built_by_episode = max_obsidian_added > 0` 启发式

## 背景

Phase 1 已经证明固定 MineRL/Malmo 1.0.2 后端可以回传 7×7×7 方块 grid、维度真值
和物品栏变化，并完成 Scripted-A0 端到端运行。但 `PortalEvaluator` 之前只是把
预先组合好的布尔值做 AND 运算，缺少以下能力：

- 门框几何、合法尺寸范围、缺角规则；
- 把门框归因到“本回合建造”的可追溯证据；
- 完整里程碑序列（包含 `build_site_selected` 与 `first_obsidian_placed`）；
- 失败分类（几何错、缺一、激活后离开 grid 等）。

任何在 VLM Agent 接入前必须先冻结这些规则。本决策只影响 `PortalEvaluator`
和 `MineRLEnvironmentBackend.get_evaluation_state` 的返回证据，不修改
EnvironmentBackend 协议、动作边界或固定依赖。

## 固定技术栈一致性

- Minecraft 1.16.5：传送门方块为 `minecraft:nether_portal`。
- 门框激活要求 flint_and_steel 击中有效门框内部的空气方块。
- 门框允许的最外尺寸为 23×23，最小尺寸为 4×5（不含 4×4）。
- 门框四角是可选的：缺角门框只要其余边框完整仍可激活。
- 内部必须没有被黑曜石阻挡；`fire` 出现在内部不影响激活与评分。
- 内部不得包含 `dirt` / `bedrock` / `grass` / `grass_block` / `other`
  / `missing` 等其他块；`missing` 触发 fail-closed 行为。

## 门框规则

### 1. 尺寸

- 外宽 W、外高 H（以 1 cell 为单位）。
- 合法范围：4 ≤ W ≤ 23；5 ≤ H ≤ 23。
- 最小可激活门框：4×5。
- 不接受 4×4 或更小的外框。
- 同一 episode 内只接受一个 frame candidate；grid 内同时存在多个有效门框时
  选择包含 episode 本回合新放置黑曜石最多者，相同则选择外周长最小者。

### 2. 朝向

只支持两种水平朝向：

- `plane_z`：门框位于 z = const 的 X-Y 平面内，外观宽度沿 X，外观高度沿 Y；
  内部为 x ∈ (x0, x0+W-1) × y ∈ (y0, y0+H-1) × z = const。
- `plane_x`：门框位于 x = const 的 Y-Z 平面内，外观宽度沿 Z，外观高度沿 Y；
  内部为 z ∈ (z0, z0+W-1) × y ∈ (y0, y0+H-1) × x = const。

`plane_z` 与 `plane_x` 的尺寸约束完全相同。垂直门框不在 Phase 2 范围内。

### 3. 黑曜石边框

- 完整外周（包含四个角）共 `2W + 2H - 4` 个 cell。
- 缺角规则下，必需的非角 cell 共 `2W + 2H - 8` 个：4 个角 cell 可缺。
- 边框的每个 cell 在当前 grid 中必须是 `obsidian`。
- Phase 2 默认 `allow_missing_corners=true`，因为这是 1.16.5 的合法形态。

### 4. 内部空间

- 内部宽度 = W-2，内部高度 = H-2（必为正整数，最小 2×3）。
- 内部每个 cell 在当前 grid 中必须是 `air`、`nether_portal` 或 `fire`。
- 任何 `obsidian`、`dirt`、`bedrock`、`grass`、`grass_block` 都是阻挡；
  `other` / `missing`（bridge 未映射或缺失）也视为阻挡，并使该候选
  失效（fail closed on missing）。
- 激活前内部可以为 `air` 或 `fire`；激活后允许出现 `nether_portal`。

### 5. 激活

- 只有在锁存的 episode-built frame identity 的 interior 出现
  `nether_portal` 时才视为本回合激活。
- 预存 portal / 外部因素生成的 `nether_portal` 不绑定到本回合。
- 激活状态被锁存：即使 latched frame 后续在 grid 中消失（典型情况：进入
  Nether 后 Overworld grid 失真），`portal_activated=True` 和
  `first_activation_step` 仍然有效。
- `latched_activation_offsets` 保存精确的 `nether_portal` 坐标。

### 6. 维度切换

- 当任意 agent 的 `dimension` 变为 `minecraft:the_nether` 时，
  `agents_in_nether` 永久加入该 agent。
- `first_nether_step_by_agent` 按 agent 锁存第一次进入下界时的 step。

### 7. 终止与失败

- 评测器不在 episode 进行中输出 failure。`EvaluationState.failure_type`
  / `failure_step` / `last_successful_milestone` 只在
  `episode_terminated=True` 时填充。
- 终止信号来源：MineRL `done=True`（backend 自动设置）、
  `backend.mark_terminated(step_id, reason)`（driver 主动结束）、或
  `EvaluationState.episode_terminated=True` 显式构造。
- 失败分类按以下优先级：
  1. `frame_not_built_by_episode`（attribution 失败）
  2. `frame_never_valid`（无 frame）
  3. `portal_never_activated`（已建未激活）
  4. `no_agent_entered_nether`（已激活无进入）
- `failure_step` 是 episode 终止 step（`terminated_step`），即不可恢复
  的首次确认点；不是“当前 step”。

### 8. 里程碑事件

- 所有 milestone 都是 `obsidianlink.logging.events.StructuredEvent` 顶层
  对象：`episode_id` / `step_id` / `event_type` / `timestamp` /
  `agent_id` / `payload`。
- `timestamp` 锁存于首次观察，禁止用 emission 时的 wall-clock 反填。
- `payload` 不得重复 `episode_id`（该字段在顶层）。
- `build_site_selected` 由 `partial frame candidate` 触发（非角
  obsidian 数量 ≥ 3 且 frame 尚未 geometric valid）。attribution 失败
  的几何合法 frame 不会触发 `build_site_selected`，因为它不表示“开始
  建造”。

## 门框由本回合建造（portal_built_by_episode）

不再使用“新增过一块黑曜石”代表门框建造。本规则要求 frame candidate 的所有
**非角** cell（共 `2W + 2H - 8` 个，缺角规则下）在当前 grid 中是
`obsidian`，并且在 baseline grid 中**不是** `obsidian`。

具体推导：

- 4×5 缺角合法门框需要 10 个必需黑曜石（`2·4 + 2·5 - 8`）。
- 4×5 完整外框共 14 个黑曜石（`2·4 + 2·5 - 4`）。
- 6×7 缺角合法门框需要 18 个必需黑曜石。
- 6×7 完整外框共 22 个黑曜石。
- 4 个角 cell 在缺角规则下可保留 baseline 原状态。
- baseline 中已是 obsidian 但当前 grid 仍是 obsidian 的 cell，**不计入**
  “本回合新增黑曜石”，防止“外部预先放置被复用”。
- 门框内部的 `nether_portal` / `fire` 不参与本判定。

## 退出条件

- `tests/test_frame_geometry.py` 覆盖所有正例与负例（含 dirt / bedrock /
  other / missing 内部阻挡）。
- `tests/test_evaluation.py`、`tests/test_minerl_backend.py` 同步反映
  新契约（latched frame identity、activation 绑定、in_progress 状态、
  StructuredEvent 顶层字段）。
- `python -m obsidianlink --check` 通过。
- 真实 MineRL 受控集成轨迹的回放证据（含每步 grid + 显式 termination
  signal）：当前 `scripts/replay_run_for_evaluator.py` 报告
  `status=insufficient_evidence`，Phase 2 不声称完成，直至该证据
  出现。
