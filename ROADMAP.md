# ObsidianLink Roadmap

> 完整研究与开发计划见 `docs/plans/`。本文只记录**实现进度**，不改研究方向。

## Current Phase

**Phase 3 — Single-Agent Portal Benchmark（L1 Controlled Construction 实现完成，live MineRL 部分验证受 Malmo 0.37.0 限制）**

Phase 1 与 Phase 2 已关闭：

```text
Phase 1  Minimal Minecraft Agent Loop     ✅
Phase 2  Benchmark MVP                    ✅
           D1 Perception   = What is there?          ✅
           D2 Grounding    = Where is the specified target?  ✅
           D3 Manipulation = Given the grounded target, can the agent act?  ✅
```

Phase 2 正式 diagnostic 范围：

- D1 Perception：Lava Presence、Water Presence
- D2 Grounding：Direction Grounding、Spatial Region Grounding（均无 motor）
- D3 Manipulation：Camera Alignment、Target Approach

D1 / D2 / D3 的 live 实验都是 **pipeline / pilot**，用于验证 BenchmarkRunner、hidden world truth 与 failure attribution。**不是正式 capability conclusion。** 不要用这些 n=1 数字写能力结论。

不要再增加 D1 / D2 / D3 diagnostic task。不要提前开发 D4 / D5 / D6。不要把 motor 写回 D2。

## Current Task

**L1 Controlled Construction（code complete, live partial）。**

L1 是 Phase 3 的第一个 end-to-end Nether Portal Benchmark level。
Casting + Portal Frame Construction 由受控 scene 预生成（Malmo 0.37.0 限制
见下）；agent 的工作只是 Ignition + Nether Entry。`success = nether_entered`（ypos < 80）。

不要再增加 L1 diagnostic。不要提前开发 L2 / L3 / L4 / D4 / D5 / D6。不要为
Reactive Agent 失败擅自继续实现 Planner / Reflection。

## Phase 2 收尾摘要

统一 Runner 已能对真实 Qwen3-VL Agent 跑 D1 / D2 / D3，并写出结构化 Result。Evaluator 使用 hidden GT 或执行后的 Minecraft 状态，而不是模型文字声明。

| 正式 task | 测什么 | live n=1（pilot） |
| --- | --- | --- |
| D1-01 Lava Presence | 画面里有没有岩浆 | 2B / 4B 正负均对 |
| D1-02 Water Presence | 画面里有没有水 | 2B 正例漏检；4B 正负均对 |
| D2-01 Direction Grounding | 岩浆在左 / 中 / 右 | 2B 3/3；4B 3/3 |
| D2-02 Spatial Region Grounding | 岩浆在 3×3 哪一格 | 2B 3/9；4B 6/9 |
| D3-01 Camera Alignment | camera 把岩浆转到画面中央 | 2B 0/3 orientation_error；4B 3/3 |
| D3-02 Target Approach | 前进到交互距离并停止 | 2B 1/1；4B overshoot |

历史 exploratory D2（camera-yaw 居中、walk-and-stop 写进 Grounding）继续保留在 `obsidianlink/experiments/runs/`，不要与上表正式 D2 / D3 混淆。

Scene-validity 帧：

- D1 / D2-01：`obsidianlink/experiments/runs/d2_01_scene_validity/`
- D2-02：`obsidianlink/experiments/runs/d2_02_region_validity/`
- D3-01：`obsidianlink/experiments/runs/d3_01_scene_validity/`
- D3-02：`obsidianlink/experiments/runs/d3_02_scene_validity/`

历史 exploratory 帧：

- 旧 D2-01 camera alignment：`obsidianlink/experiments/runs/d2_01_direction_Qwen3-VL-*`（早期 8-step motor 版本）
- 旧 D2-02 target approach：`obsidianlink/experiments/runs/d2_02_scene_validity/` 与 `d2_02_approach_Qwen3-VL-*`

**Pilot 旧数据（保留，不作为 capability 结论）**：Phase 2A/2B inventory D1；Phase 2C 单块 lava；旧 64×64 D1 v2 抓帧；上述 exploratory motor-D2。

## Phase 2C lava presence — PILOT ONLY

旧场景：单块 lava 放在玩家前方 5 格。live JSON 仍在
`obsidianlink/experiments/runs/d1_presence_lava_*.json`。
当时记录的「lava 占顶部 ~30%、肉眼可清晰看到」已被否定：输入画面过小、目标位置差，连人工都难以稳定识别。2B/4B 的 0/3 **不作为 D1 capability 结论**。

旧 plumbing（hidden `Task.ground_truth`、`D1PresenceEvaluator` 的
`perception_error` / `output_protocol_error` 拆分、inventory pilot）
仍然沿用。

## Phase 2A sub-items status (2026-08-18)

1. **PerceptionReport 数据结构 + JSON 解析** —
   `obsidianlink.benchmark.perception.{PerceptionReport, parse_perception_report}`。
   解析对 missing/garbage 全部返回 `None`。
2. **Result 扩展 `evidence: Mapping[str, Any]`** — 主指标集
   (success/steps/model_calls/invalid_actions/elapsed_time) 完全不变；
   `evidence` 只放 evaluator 私有 diagnostic breadcrumb。
3. **Evaluator ABC 加 `report=None, observation=None` kwarg** —
   默认实现不变；D1 evaluator 用来接收 Runner 转发的 side-channel。
4. **ReactiveAgent 暴露 `last_report`** — `act()` 内同步解析
   `report` 字段并写入；Phase 1 行为不变（无 `report` 时
   `last_report=None`，action 解析路径完全独立）。
5. **BenchmarkRunner 转发 `report` + 末次 agent-visible observation** —
   关键的"agent 看什么 = evaluator ground truth"对齐：在
   `env.step()` 之前快照 `last_input_observation`，最后
   `evaluator.evaluate(report=last_report, observation=last_input_observation)`。
   Runner 在 `finally` 块关闭 env，agent 抛异常也不漏 JVM。
6. **D1 Task + Evaluator + Heuristic Model + Heuristic Agent** —
   `obsidianlink.tasks.diagnostic.{D1_INVENTORY_PERCEPTION,
   D1InventoryPerceptionEvaluator, D1InventoryPerceptionModel,
   D1InventoryPerceptionAgent}`。D1 评估用 inventory 精确匹配 +
   selected_item 精确匹配；evidence 给出 reason / inventory_match /
   selected_match 等 breadcrumb。
7. **main.py 新增 `OBSIDIANLINK_PHASE=2a` 入口** — 不破坏
   `OBSIDIANLINK_PHASE=1`（默认）和 `OBSIDIANLINK_OFFLINE=1` 的
   行为；Phase 2A 模式打印结构化 Result + 完整 evidence bag。

## Live smoke 运行方式

Phase 1 (默认)：

```bash
conda run -n mc-agent python main.py
# 或在本目录内：
PYTHONPATH=/Users/joey/Documents/Projects/ObsidianLink python main.py
```

Phase 2A — D1 vertical slice：

```bash
PYTHONPATH=/Users/joey/Documents/Projects/ObsidianLink \
  OBSIDIANLINK_PHASE=2a conda run -n mc-agent python main.py
```

无 Java / 无 MineRL 时可跑离线 stub smoke：

```bash
OBSIDIANLINK_OFFLINE=1 conda run -n mc-agent python main.py
```

## 测试

```bash
PYTHONPATH=. conda run -n mc-agent python -m pytest tests/
```

224 个离线单测覆盖 D1 / Phase 1 / D2-01 / D2-02 / D3-01 Camera Alignment / D3-02 Target Approach。

## Completed

* Research direction frozen
* Research-First Master Plan frozen
* Development Plan frozen
* 旧 v2 implementation 已从当前主线移除
* 已建立最小 Research-First 项目骨架
* 已建立最小 Environment / Agent / Task / Evaluator / Runner 接口
* 已建立 offline smoke-test 结构
* **Phase 1 — Minimal Minecraft Agent Loop (code + live verification complete)**
  * `obsidianlink.env.minerl.MineRLEnvironment` (bounded action set)
  * `obsidianlink.env.actions.Action` (MOVE/CAMERA/ATTACK/USE/PLACE/WAIT payload)
  * `obsidianlink.agents.heuristic_model.HeuristicModelClient`
  * `obsidianlink.agents.reactive.ReactiveAgent` (含 `parse_model_response`)
  * `obsidianlink.main` 接入 full Phase 1 loop (env + agent + model + 16 steps)
  * 62 离线单测全过
  * Live reset → 16 step loop → close 在 `MineRLTreechop-v0` 上跑通，
    RGB frame mean 119.2 → 116.2，证实 action 到达 Minecraft
* **Phase 2A — D1 Inventory Perception vertical slice (code + live verification complete)**
  * `obsidianlink.benchmark.perception.{PerceptionReport, parse_perception_report}`
  * `obsidianlink.benchmark.result.Result` 增加 `evidence: Mapping[str, Any]`
  * `obsidianlink.benchmark.evaluator.Evaluator` ABC 增 `report=, observation=`
  * `obsidianlink.benchmark.runner.BenchmarkRunner` 转发
    `agent.last_report` + 末次 agent-visible observation；env 关闭走 `finally`
  * `obsidianlink.agents.reactive.ReactiveAgent.last_report` side-channel
  * `obsidianlink.tasks.diagnostic.D1_INVENTORY_PERCEPTION` Task + Evaluator
    + Heuristic Model + Heuristic Agent
  * `obsidianlink.main` 新增 `OBSIDIANLINK_PHASE=2a` 入口
  * 100 离线单测全过（62 Phase 1 旧 + 29 D1 新增 + 8 vision dispatch
    + 1 兼容 stub 修正）
  * Offline vertical slice 端到端跑通：Task → Runner → Agent → Evaluator
    → Result，evidence bag 完整
  * **Live run 2026-08-18 21:41**：`OBSIDIANLINK_PHASE=2a` 在
    `MineRLTreechop-v0` 跑通，`env.reset()` 27.25s 冷启后 2 step
    loop 跑完，D1 evaluator 对 Treechop 初始空 inventory 打出
    `success=True, reason=ok`。**第一个真实结构化 Result**。
* **Phase 2B — wire Qwen3-VL into the D1 vertical slice (code + live verification complete)**
  * `obsidianlink.agents.model_client.VisionModelClient` opt-in side
    protocol（@runtime_checkable）+ `call_model` 派发函数；现有
    ModelClient contract 零改动。
  * `obsidianlink.agents.qwen_vl_client.QwenVLModelClient`：本地
    Qwen3-VL 客户端，懒加载、device 自适应（auto → mps/cpu）、
    兼容新旧 transformers API（`AutoModelForImageTextToText` /
    `AutoModelForVision2Seq`、`dtype` / `torch_dtype`）。
  * `D1InventoryPerceptionAgent` 改成真感知 agent：D1 prompt
    不再泄 inventory / selected_item，全部依赖模型对帧的视觉
    推断。Phase 2A 的"plumbing artefact" 自动派生行为已彻底移除。
  * `obsidianlink.experiments.smoke_qwen_vl_d1` 合成帧 smoke：
    Qwen3-VL-2B 走 MPS auto-detect float16 看到 64x64 空 hotbar
    帧，正确返回 `inventory={}, selected_item=None`。首次
    `agent.act()` 9.27s（含模型加载），text-only fallback 0.13s。
  * `obsidianlink.benchmark.evaluator.Evaluator` ABC 增
    `raw_response=` 通道；Runner 转发 agent.last_raw_response。
  * `obsidianlink.experiments.multi_episode_d1` 多 episode 跑实验
    脚本：`--model-path` / `--num-episodes`，保存完整 evidence
    + 聚合统计到 `experiments/runs/`。
  * **D1 model-scale control（N=5+5，2026-08-18 22:22 / 22:29）**：
    Qwen3-VL-2B **0/5** success vs Qwen3-VL-4B **4/5** success。
    2B 失败模式 = schema 错（截断 / 未引号 / 多语言漂移） +
    wild hallucination（9 个编造物品，含中文）；4B 失败模式 =
    mild hallucination（1/5，看到 dirt 方块误认为 hotbar 物品）。
    100 离线单测全过（增量部分为 raw_response 通道 + new
    dispatch tests 稳定 + D1 agent 行为更新 + Phase 2A plumbing
    artefact 移除）。

* **D1 Perception Pilot — complete (2026-08-19)**
  * D1-01 Lava Presence：640×360 受控正负场景，hidden GT，2B/4B 各 1 episode，链路通；两模型正负均判对。
  * D1-02 Water Presence：同一院落；DrawBlock 不能画水，正例 env-side 倒水桶。2B 正例漏检、负例对；4B 正负均对。n=1，不作 capability 结论。
  * 不再增加 Obsidian / Iron / Log 等 D1 task。D1 Perception Pilot 关闭。

* **D2-01 Direction Grounding — complete (2026-08-19, redesigned)**
  * 定义：给定语义目标与第一人称 RGB，判断目标相对画面的水平方向（left / center / right）。无 camera / movement。
  * 复用 D1-01 岩浆正例院落，三个 spawn yaw（left +35° / center 0° / right −35°），POV 640×360，`max_steps=1`，WAIT only。
  * Hidden GT 由 scene 的 spawn-yaw 映射产生，挂在 `Task.ground_truth`，不进 Observation / prompt。
  * 输出 `DirectionGroundingReport`：`{"target","direction"}`。Evaluator：`ok` / `grounding_error` / `output_protocol_error`。
  * MineRL camera `[pitch, yaw]` 适配器修正保留（D3 需要；D2 不再执行 camera）。
  * Scene-validity（既有）：left/center/right 岩浆质心 nx ≈ 0.21 / 0.50 / 0.79。
  * Live n=1（pipeline / pilot，非 capability）：Qwen3-VL-2B 3/3 ok；Qwen3-VL-4B 3/3 ok。未因结果改 prompt。

* **D2-02 Spatial Region Grounding — complete (2026-08-19)**
  * 定义：给定语义目标与第一人称 RGB，判断目标落在 3×3 画面区域中的哪一格。无 camera / movement。
  * 同一岩浆院落；9 个 spawn pose（yaw ±35°/0° × pitch 45°/25°/8°），POV 640×360，`max_steps=1`，WAIT only。
  * Hidden GT 由 scene 的 (yaw, pitch) → region 映射产生，挂在 `Task.ground_truth`，不进 Observation / prompt。
  * 输出 `SpatialRegionGroundingReport`：`{"target","region"}`。Evaluator：`ok` / `grounding_error` / `output_protocol_error`。
  * Scene-validity：9/9 岩浆质心落入对应 3×3 bin。
  * Live n=1（pipeline / pilot，非 capability）：Qwen3-VL-2B 3/9 ok（protocol 4、grounding 2）；Qwen3-VL-4B 6/9 ok（protocol 1、grounding 2）。未因结果改 prompt。

* **Historical / exploratory D2 (not a formal D2 result, 2026-08-19)**
  * 旧 D2-01 把 direction classification 与 camera yaw 居中绑在一起（`max_steps=8`，`orientation_error`）。Live n=1：2B 3/3 ok；4B 1/3。现已作为正式 **D3-01 Camera Alignment** 重做。
  * 旧 D2-02 把 walk-and-stop 写进 Grounding（`approach_error` / `overshoot_error`）。Live n=1：2B WAIT 原地；4B 走过头。现已作为正式 **D3-02 Target Approach** 重做。
  * 实验 JSON / 帧保留在 `obsidianlink/experiments/runs/`，不作 D2 capability 结论。

* **D3-01 Camera Alignment — complete (2026-08-19)**
  * 定义：目标已可见并可 grounding。Agent 用 camera yaw 把岩浆转到画面中央并停止。无 movement / attack / use / place。
  * 复用 D2-01 岩浆院落与 spawn yaw（left +35° / center 0° / right −35°），新 env id `MineRLD301*`，POV 640×360，`max_steps=8`。
  * 闭环：RGB → Agent → camera/wait → Minecraft → 新 RGB。MOVE 被夹成 WAIT。
  * Evaluator 读执行后的 hidden yaw（MineRL location monitor / gym info），目标 yaw = 0，容差 ±12°。不根据模型文字声明判成功。
  * Failure modes：`ok` / `orientation_error` / `output_protocol_error` / `missing_world_truth`。
  * Scene-validity：left/center/right 质心 nx ≈ 0.21 / 0.50 / 0.79；`+20` camera 后 hidden yaw 0 → 20。
  * Live n=1（pipeline / pilot，非 capability）：Qwen3-VL-2B 0/3（全部 `orientation_error`，protocol 正常）；Qwen3-VL-4B 3/3 ok。未因结果改 prompt。

* **D3-02 Target Approach — complete (2026-08-19)**
  * 定义：目标已可见且基本居中。Agent 用前进走到交互距离并停止。无 camera / strafe / attack / use / place。不测 D2 grounding（无 far/near 字段）。
  * 同一岩浆院落，yaw=0，spawn z=−1.5（warmup 后约 4.6 格）。新 env id `MineRLD302Approach-v0`，POV 640×360，`max_steps=20`。
  * 闭环：RGB → Agent → move/wait → Minecraft → 新 RGB。CAMERA / 后退被夹成 WAIT。
  * Evaluator 读执行后的 hidden xyz，计算到岩浆 AABB 的水平距离。成功带 0.6–2.0 格。过远 `approach_error`，过近 `overshoot_error`。不根据模型文字声明判成功。
  * Scene-validity：起点岩浆居中且偏远（nx ≈ 0.48，d ≈ 4.65）；脚本化前进第 14–19 步进入成功带，第 20 步 overshoot（0.52）。
  * Live n=1（pipeline / pilot，非 capability）：Qwen3-VL-2B 1/1 ok（最终 d ≈ 0.96）；Qwen3-VL-4B 0/1 `overshoot_error`（d ≈ 0.53，一路前进未停）。未因结果改 prompt。

## Next

**Phase 3 — L1 Controlled Construction 完成后，等待下一步指令。**

不要再增加 D1 / D2 / D3 diagnostic task。不要提前开发 D4 / D5 / D6 / L2 / L3 / L4。不增加导航 / 规划 / 检测框架 / Planner / Reflection / Multi-Agent。

## Blocked

* `MineRLNavigate-v0` 在本机 Malmo 0.37.0 (`mcprec-6.13.jar:583`,
  `EnvServer.setGameSetttings`) 上有 `NullPointerException`，跟
  `NavigationDecorator` / `RewardForTouchingBlockType` mission XML 分支
  相关。这是 Malmo 服务端 bug，**不是 ObsidianLink 代码问题**，不在本项目
  范围内修。Phase 1 暂时用 `MineRLTreechop-v0` 绕过；Phase 2 起需不需要
  切回 Navigate 由实验设计决定。

* **L1 live env 受 Malmo 0.37.0 多项限制（2026-08-19）**：
  1. `<Placement>` MissionHandler 的 `/teleport` 命令在 1.16.5 客户端被解析为
     `Ambiguity between arguments [teleport, ...]` 并被忽略，agent 经常 spawn
     在世界默认 spawn（~500 块内随机），远离 scene。
  2. `<ChatCommands>` ` /give` 解析但不被执行。`MinecraftItems` whitelist
     不含 `obsidian`（`obsidian` 在 `MinecraftBlocks` 里），所以
     `SimpleInventoryAgentStart` 不能给 agent obsidian。
  3. `PlaceBlock` handler 在 `mcprec-6.13` 上 crash Malmo server
     （"Connection timed out"），需要自定义 `L1PlaceCommands` 绕开。
  4. `ObservationFromGrid` 在 401×401 plate 上（8 MB XML）持续返回全部
     `minecraft:air`——POV 明明能看见 obsidian frame。Malmo 服务端 bug。

  L1 live end-to-end 状态：
  * 401×401 obsidian plate workaround 让 spawn 落在 plate 上的概率
    显著提高（l1_grid_y99.py: pos=(0.5, 101.0, 2.0)）。
  * Scene pre-draws the 14-block obsidian frame（Casting + Portal Frame
    Construction 由受控 scene 完成，是 Malmo 限制下的**最小可靠 workaround**）。
  * Oracle + Runner + Evaluator 跑通（267/267 离线单测 + offline stub end-to-end
    `test_runner_with_stub_env_and_scripted_oracle_succeeds`）。
  * Oracle live n=1 跑通 env reset、POV 显示 frame、inventory 有 flint_and_steel，
    但 grid 看不到 obsidian（Malmo bug）+ USE action 没点燃 portal（frame_complete
    evaluator 报 0/14）→ `success=False, reason=portal_frame_incomplete,
    final_ypos=101.0`。
  * L1 的 code / scene / task / oracle / evaluator 全部完成并离线验证；
    live 端的 grid observation 限制是 Malmo 服务端问题，**不是 ObsidianLink
    代码问题**，不在本项目范围内修。
