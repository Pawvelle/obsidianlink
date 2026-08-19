# ObsidianLink Roadmap

> 完整研究与开发计划见 `docs/plans/`。本文是**唯一动态状态源**：Current Phase、Current Task、Completed、Next、Blocked 只在这里维护。不要把进度写回冻结计划、README 或 AGENTS.md。

## Current Phase

**Phase 3 — Single-Agent Portal Benchmark**

Phase 1 与 Phase 2 已关闭：

```text
Phase 1  Minimal Minecraft Agent Loop     ✅
Phase 2  Benchmark MVP                    ✅
           D1 Perception (Lava Presence)  ✅ representative
```

不要再增加 D1 / D2 / D3 diagnostic task。不要提前开发 D4 / D5 / D6。不要实现 Planner / Reflection / Multi-Agent。

## Current Task

**Formal L1 Controlled Construction**

端到端目标保持 method-agnostic：

```text
Construct / complete → activate → enter Nether
```

Bucket Casting 是第一版受控评测的主要 reference strategy，而非强制 solver。Agent 可使用 live Minecraft Wiki tool 查询任务相关的原版知识；Evaluator 仍只从 evaluator-only world truth 判断 portal activation 与 Nether transition。

不要把 scene 预建 portal frame 当作正式 L1。旧 L1 已移出 active path。

**2026-08-19 technical feasibility spike**（`obsidianlink/experiments/spike_l1_feasibility.py`，live `l1_spike_20260819_124538Z`）：

* bucket casting **可行**：真实 `use` lava/water 生成新的 obsidian，不是 DrawBlock 预建 frame
* 物品准备：**可行**，`InventoryAgentStart`（不改 L1 语义）
* 选物品：必须用 `hotbar.1-9`。`EquipAction` 会把 `equip none` 送给 MCP-Reborn 并 crash
* Oracle 走到：inventory → hotbar → pour lava → water convert → pickup → 至少 1 次 extra cast
* 未完成：10-block frame / flint ignition / Nether entry（extra cast 期间 connection timeout）
* Evaluator 不要用 `ObservationFromGrid`。可用：inventory delta、`location_stats`（gym info）、POV 帧、可选 `RewardForTouchingBlockType(nether_portal)`

该 spike 是 scripted/oracle mechanics feasibility，不是正式 L1 Agent 或 live L1 benchmark evidence。正式 L1 不用预建 frame，也不使用已知不可靠的 EquipAction。

**2026-08-19 L1 Controlled Environment v0.1**（`obsidianlink/env/l1_scene.py`，live `l1_env_smoke_20260819_175245Z`）：

* 施工面是超平坦 **草地**（`FLAT_WORLD`），不是黑曜石地板。DrawBlock 只画 4×4 lava pool
* 原因：Malmo DrawingDecorator 只能画 `lava` / `obsidian`；黑曜石地板会让 Agent 把地面当 portal 材料
* spawn `(0.5, 4.0, 0.5)`；inventory 与 `hotbar.1-9` 与上一版相同
* live `C2_grass_floor` 通过（`grass_frac≈0.76`）；无预建 portal，无 EquipAction
* 这是 environment smoke，不是 Oracle、Evaluator 或 ReactiveAgent L1 实验

## Completed

* Research direction frozen
* Research-First Master Plan frozen
* Development Plan frozen
* **2026-08-19 architecture reset**：删除被 Malmo workaround 污染的 L1 与过期 diagnostic 实现，重建最小 Environment / Agent / Runner
* **Phase 1 — Minimal Minecraft Agent Loop**
  * `MineRLEnvironment`: reset / observe / step / close
  * RGB + inventory + selected_item
  * bounded actions
  * Live 2026-08-19：`MineRLTreechop-v0` 16 steps，frame mean 117.5 → 115.3
* **Phase 2 — Benchmark MVP**
  * Task / Evaluator / BenchmarkRunner / Result
  * Agent-visible Observation 与 evaluator-only hidden_state 隔离
  * D1 以 env `target_truths` 为 ground truth；与 Task 标签冲突则为 evaluation_error
  * Runner 将 env / agent / evaluator 异常写成结构化 Result，不中断实验
  * `Task.allowed_actions` 外的动作被夹成 WAIT 并计入 invalid_actions
  * 代表性 diagnostic：D1 Lava Presence
  * Vision dispatch 必须把 `Observation.frame` 交给 vision model；fallback 写入 Result.evidence
  * Live 2026-08-19：Qwen3-VL-2B，`vision_completions=1`，`success=True`，GT 只在 hidden_state
* Live Minecraft Wiki Tool
* Tool-enabled ReactiveAgent
* **L1 Controlled Environment v0.1**（env + inventory + hotbar smoke；无 Oracle / 无 Agent）

## Next

实现 L1 Evaluator（不依赖 ObservationFromGrid），再做 Scripted / Oracle 机械验证；然后才是 tool-enabled ReactiveAgent L1 pilot。在此之前不要开始 L2 / Planner / Reflection。

## Blocked

* `MineRLNavigate-v0` 在本机 Malmo 0.37.0 上有 `NullPointerException`。Phase 1 使用 `MineRLTreechop-v0`。
* Malmo 0.37.0 已知限制（记录，不在本次用 workaround 改 Benchmark 定义）：
  * `<Placement>` / `/teleport` 可能被忽略
  * `<ChatCommands> /give` 可能不执行
  * `PlaceBlock` handler 可能 crash server
  * `ObservationFromGrid` 可能返回全 air，不能作为 evaluator truth
* DrawingDecorator：仅 `DrawBlock`，仅 `lava` / `obsidian`
* `EquipAction`：MineRL 1.0.2 MCP-Reborn `constructKeyboardState` 对 `equip none` / `equip <item>` 做 `Integer.parseInt`，episode 直接结束。L1 应使用 hotbar keys

## Historical L1

`obsidianlink/experiments/runs/l1_*` 是 debugging record，**invalid for L1 capability conclusion**。原因：

1. L1 semantics changed（Casting/Frame 被移入 scene）
2. evaluator world truth unreliable
3. Reactive run did not actually use vision
