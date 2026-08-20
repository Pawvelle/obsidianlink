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

**2026-08-20 L1 Mechanical Interaction Test**（`obsidianlink/experiments/run_l1_mechanics.py`，live `l1_mechanics_20260820_033330Z`）：

* 在正式 `MineRLL1Controlled-v0` 上，用未来 Agent 同款动作（MOVE / CAMERA / USE / ATTACK / HOTBAR / WAIT，外加 `sneak` 修饰）完成浇灌法关键 mechanics
* empty bucket 从 lava source 舀岩浆 → `lava_bucket`；`use` 放岩浆；`use` 放水；lava+water 生成 **至少 1 块新 obsidian**（非 DrawBlock / 非预建）
* cobblestone 经 hotbar + 原生 `use` 放置；iron pickaxe `attack` 拆除后 inventory `63 → 64`
* 未使用 EquipAction、ObservationFromGrid、PlaceBlock
* **NEW OBSIDIAN = TRUE**。这是机械可行性验证，不是 Oracle、Evaluator 或 Agent 能力结论

**2026-08-20 L1 Evaluator**（`obsidianlink/benchmark/l1_evaluator.py`，live `run_l1_evaluator_smoke.py`）：

* evaluator-only truth channels，live-verified on this MineRL 1.0.2 / MCP-Reborn / Malmo 0.37.0 stack：
  * `portal_activated` / `portal_contacted` ← gym step `reward` from `RewardForTouchingBlockType(nether_portal)`（新增到 `L1ControlledEnv.create_rewardables`；reward 只经 `MineRLEnvironment.hidden_state["reward"]`，从不写入 `Observation`）
  * `nether_entered` ← `biome_id == 8` (Nether) from `ObservationFromCurrentLocation`/`location_stats`，且必须发生在 `portal_activated` **之后**（防止 biome 噪声单独判定成功）
  * `portal_constructed` 保持 `"unknown"`：这台 stack 上没有便宜可靠、非 `ObservationFromGrid` 的 frame-complete 真值，不为它开发 block parser
* `success = nether_entered`，`nether_entered` 必须同时满足 activation 证据与 strict biome match；只有弱 biome 变化（未命中 8）不算 success
* live smoke 确认：`hidden_state` 含 `reward`/`biome_id`/`can_see_sky`/`light_level` 等字段，Agent-visible `Observation`（`frame`/`inventory`/`selected_item`）不变；空场景下 `evaluator.evaluate(...)` 正确 fail-closed（`success=False`, `reason=nether_entry_not_confirmed`）

**2026-08-20 Water Recovery Isolation**（`obsidianlink/experiments/run_water_recovery_isolation.py`，live `water_recovery_iso_20260820_105237Z` + `105355Z`）：

* 最小场景：fresh reset → 低头放 1 格 water source → fluid wait → 单次 `USE` 回收 → **20 tick 纯 WAIT**（无 USE / ATTACK / MOVE / HOTBAR / CAMERA；每 tick 记录 mapped `minerl.use`）
* 两次独立 episode 时间线相同：放水前 `bucket=1, water_bucket=1`；pour `USE` 当 tick（tick 3）稳定为 `bucket=2, water_bucket=0`；recover `USE` 当 tick（tick 13）出现 `water_bucket=1, bucket=1`；随后 20 WAIT 全程保持，`minerl.use=0`
* **没有 inventory rollback**。`water_bucket=1` 在这个 WAIT-only 窗口里不是 transient observation，也不是 delayed sync 后被权威状态打回 empty bucket
* 因此 **不要** 为这个 primitive 加 `N=3` consecutive confirmation，也不要把单帧 `water_bucket>=1` 一律当成假读数
* 这不是 Gate 1、不是完整地狱门。Gate 1 里曾经看到的 “CAMERA/WAIT 后 water_bucket 消失” **不能**用 “MineRL inventory 天生会回滚” 解释；剩余嫌疑是 recover 后的额外交互（多 tick `USE` burst、随后 CAMERA 瞄到残留流动水）或岩浆模具场景，本次 isolation 按协议没有测 CAMERA
* POV：回收后准星处水源消失，hotbar 变为 empty bucket + water bucket，与 inventory 一致

**2026-08-20 Gate 1 — one obsidian**（`run_l1_oracle.py`，live `l1_oracle_20260820_113909Z` + `114030Z`）：

* 最短序列：scoop lava → place lava → 邻格放水 → wait until obsidian。无模具、无 2 格底边、无 portal frame、无点火
* 桶交互全部 **单次 `USE`**（3 次 USE / episode）。约 65 steps，wall ≈ 28s，远低于 240s socket 超时
* 成功标准：`observed_new_obsidian`（inventory 链 + POV `obsidian_visual_rose`）。`L1Evaluator.success` 仍是 Nether entry，Gate 1 不为 True
* Run `113730Z` 是假阳性：对准岩浆格放水会把 lava source **替换成水**，`lava_frac` 下降但 `obsidian_frac` 几乎不变。已要求 `obsidian_visual_rose`，并改为邻格放水 + lava settle wait
* 修正后 2/2：`obsidian_frac` 0.0009 → 0.041 / 0.023，水保持放下（WAIT 时不再被空桶舀回）
* 限制：开阔地面岩浆仍会蔓延，这不是精确 portal 格；`exact_block_truth` 不可用

**2026-08-20 Full Scripted Oracle — 卡在 Gate 1**（`obsidianlink/experiments/run_l1_oracle.py` / `l1_oracle.py`，`obsidianlink/tasks/portal.py`）：

* Portal 参考几何：经典 cornerless 10-block frame（省略四角），`base_x=-1, base_y=4, z=3`，底边落在已验证的 y=3 grass 施工面正上方；offline tests 覆盖 frame/interior 形状、method-agnostic Task goal
* Live 发现 1 — **不加模具的浇灌不可控**：在开阔草地上倒岩浆会向四周多格蔓延，不会停留在单一目标格（有截图证据），之前 `l1_mechanics` 的“NEW OBSIDIAN=TRUE”只是启发式视觉判定，不是几何证明。修复：Oracle 在浇筑前先用 cobblestone 在目标两侧砌矮墙（mold）约束岩浆
* Live 发现 2（更严重，未解决）— **长 episode 会话可能挂起并触发 240s socket 超时**：`minerl/env/_multiagent.py` 硬编码 `SOCKTIME = 240s`。纯 `WAIT` 循环验证 93,200 步 / 340s 无异常（排除固定 wall-clock episode 上限）；但一旦 episode 内出现较多真实液体/方块放置动作（造好模具 + 舀岩浆 + 往返移动），会在约 270–280s 处遇到 `TimeoutError` → `RuntimeError: Attempted to step an environment server with done=True`，两次独立复现，失败点不完全固定在同一动作类型上（一次卡在放置模具的 `use`，一次卡在返回途中的 `move`），符合“服务器端因流体模拟负载累积而逐步变卡、最终某一步响应超过 240s”的特征，而不是我们代码的死循环（步数预算已大幅收紧后仍复现）
* Gate 1（浇筑 2 块底边 obsidian）**未在单次稳定 episode 内端到端确认**：模具搭建与至少一次舀岩浆已经真实成功，但受上述挂起风险影响，尚未拿到一次完整跑通 pour→water→obsidian 视觉确认的干净 run
* 未使用 EquipAction、PlaceBlock、ObservationFromGrid、DrawBlock portal、teleport、command、预建 frame、inventory 注入
* 结论：**Oracle SUCCESS = False**，停在 Gate 1（bottom row casting），不是几何/瞄准逻辑问题，而是 MCP-Reborn/Malmo 在长时间高频液体交互下的服务器端稳定性限制。下一步需要先定位/缓解这个挂起（例如减少单 episode 内的液体方块更新总量、拆分动作节奏、或确认是否有官方已知 issue），而不是继续堆 Gate 2-8 的建造逻辑

下一步：Gate 1 一格黑曜石已在短 episode 内 live 确认。不要自动继续完整 10 格 frame / 点火 / 入 Nether。精确几何仍需要模具，长 episode 240s 挂起风险仍在。不要提前开发 Planner / Reflection / L2 / ReactiveAgent L1 Pilot。

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
* **L1 Mechanical Interaction Test**（正式 L1 上 scripted 浇灌 mechanics；NEW OBSIDIAN = TRUE；无 Oracle / 无 Evaluator / 无 Agent）
* **L1 Evaluator**（evaluator-only `reward` + `biome_id` truth；live-verified fail-closed；无 ObservationFromGrid；无 Observation 泄漏）
* **Formal L1 Portal Task**（method-agnostic goal，`obsidianlink/tasks/portal.py`）与 cornerless 10-block 参考几何（offline-tested）
* **Water Recovery Isolation**（单次 `USE` 回收 + 20 WAIT；2/2 fresh episodes 无 rollback；非 Gate 1）
* **Gate 1 one obsidian**（短 scripted 浇灌；修正后 2/2 `observed_new_obsidian=True`；非 portal frame）

## Next

Gate 1 一格黑曜石已 live 确认。下一步若做精确 frame，需要模具，并继续避开长 episode 240s 挂起。在此之前不要开始 Gate 2-8 / L2 / Planner / Reflection / ReactiveAgent L1 Pilot。

## Blocked

* `MineRLNavigate-v0` 在本机 Malmo 0.37.0 上有 `NullPointerException`。Phase 1 使用 `MineRLTreechop-v0`。
* Malmo 0.37.0 已知限制（记录，不在本次用 workaround 改 Benchmark 定义）：
  * `<Placement>` / `/teleport` 可能被忽略
  * `<ChatCommands> /give` 可能不执行
  * `PlaceBlock` handler 可能 crash server
  * `ObservationFromGrid` 可能返回全 air，不能作为 evaluator truth
* DrawingDecorator：仅 `DrawBlock`，仅 `lava` / `obsidian`
* `EquipAction`：MineRL 1.0.2 MCP-Reborn `constructKeyboardState` 对 `equip none` / `equip <item>` 做 `Integer.parseInt`，episode 直接结束。L1 应使用 hotbar keys
* 流动水（flowing water）不能用空桶回收，只会推玩家。scripted 放置/挖掘圆石时准星必须打在方块上，不能打在水面。这不是 Benchmark 定义问题，也不要为此启用 PlaceBlock
* 在开阔地面（无模具）浇岩浆会自由蔓延到多个相邻格，不会停在单一目标格；构造精确几何（如 portal frame）前必须先用 cobblestone 砌矮墙围住目标格
* **2026-08-20**：长 episode（约 270-280s 内出现较多真实液体/方块放置动作）可能触发 `minerl/env/_multiagent.py` 硬编码的 240s socket 超时（`RuntimeError: Attempted to step an environment server with done=True`）。纯 `WAIT` 循环 93,200 步 / 340s 验证无此问题，说明不是固定 episode 时长上限，而更像服务器端因液体模拟负载累积导致某一步响应变慢。两次复现的具体失败动作类型不同（一次 `use`、一次 `move`），指向服务器端渐进变卡而非单一确定性触发点。这是 Full Scripted Oracle 卡在 Gate 1 的直接原因之一，需要专项调查（不要通过修改 L1 语义规避）

## Historical L1

`obsidianlink/experiments/runs/l1_*` 是 debugging record，**invalid for L1 capability conclusion**。原因：

1. L1 semantics changed（Casting/Frame 被移入 scene）
2. evaluator world truth unreliable
3. Reactive run did not actually use vision
