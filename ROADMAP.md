# ObsidianLink Roadmap

> 完整研究与开发计划见 `docs/plans/`。

## Current Phase

**D1 Perception Pilot — 完成。下一步：D2 Grounding。**

D1 v2 原则：单帧、单目标、二分类、受控场景、可靠 hidden ground truth、肉眼清晰、POV 640×360。

**D1-01 Lava Presence**：正负场景 + 2B/4B 各 1 episode，评测链路通。

**D1-02 Water Presence**：正负场景 + 2B/4B 各 1 episode，评测链路通。EnvServer 不能 DrawBlock water，正例在 env-side 把水桶倒在与 lava-negative 相同的黑曜石院落地板上。

不再增加 Obsidian / Iron / Log 等 D1 task。不调 prompt。不在 D1 上做大规模统计。

**Pilot 旧数据（保留，不作为 capability 结论）**:

- Phase 2A/2B inventory D1
- Phase 2C lava presence
- 旧 64×64 D1 v2 抓帧

## Current Task

D1 Pilot 已完成。下一步进入 **D2 Grounding**。

Lava 帧：`obsidianlink/experiments/runs/d1_01_scene_validity/`
Water 帧：`obsidianlink/experiments/runs/d1_02_scene_validity/`
Water 结果：`obsidianlink/experiments/runs/d1_02_water_Qwen3-VL-*_1ep_*.json`

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

158 个离线单测覆盖：Action payload、bounded action translation（含 Treechop /
Navigate 两套 action space）、HeuristicModelClient cycle、ReactiveAgent 解析
（含 JSON 错误兜底 + 新的 `last_report` 提取路径）、inventory 摘要新旧两种
shape、offline env smoke、**Phase 2A 新增**：PerceptionReport 构造 / 解析
edge case、D1 evaluator 4 种 failure 模式、ReactiveAgent 暴露 last_report、
D1 agent 从 observation 派生 report、Runner success/failure/exception 路径、
Runner 转发 agent-visible observation 而非 post-step 观察。

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
  * 不再增加 Obsidian / Iron / Log 等 D1 task。下一步是 D2 Grounding。

## Next

**D2 Grounding.** D1 Perception Pilot 已关闭，不再扩 D1 物体类、不调 prompt、不做 D1 大规模统计。

## Blocked

* `MineRLNavigate-v0` 在本机 Malmo 0.37.0 (`mcprec-6.13.jar:583`,
  `EnvServer.setGameSetttings`) 上有 `NullPointerException`，跟
  `NavigationDecorator` / `RewardForTouchingBlockType` mission XML 分支
  相关。这是 Malmo 服务端 bug，**不是 ObsidianLink 代码问题**，不在本项目
  范围内修。Phase 1 暂时用 `MineRLTreechop-v0` 绕过；Phase 2 起需不需要
  切回 Navigate 由实验设计决定。
