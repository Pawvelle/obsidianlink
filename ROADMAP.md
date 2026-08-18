# ObsidianLink Roadmap

> 完整研究与开发计划见 `docs/plans/`。

## Current Phase

**Phase 1 — Minimal Minecraft Agent Loop** *(code + live verification complete)*

目标：建立 Python、Minecraft、observation 与 action 之间的第一条真实闭环。
dev plan 的 7 个 sub-item（env 启动、RGB obs、inventory、bounded actions、
ModelClient、ReactiveAgent、真实 loop）均已实现并在 live Minecraft 上验证。

## Current Task

无 — Phase 1 七项 sub-item 全部完成。

下一条独立任务（Phase 2 — Benchmark MVP）按 dev plan 在 live Phase 1 loop
稳定后才启动；当前节点可以接 Phase 2，也可以停在 Phase 1 收尾。

## Phase 1 sub-items status (2026-08-18)

1. **Minecraft / MineRL 环境启动与 reset** — `obsidianlink.env.minerl.MineRLEnvironment`
   实现；`env.reset()` 28.0s 冷启后返回真实 observation。
2. **RGB observation** — `Observation.frame = ndarray shape=(64, 64, 3) dtype=uint8`。
3. **inventory / selected item observation** — adapter 适配新旧两种 MineRL
   inventory shape（`{name: count}` 和 `{name: {quantity: N}}`），并提供
   `selected_item` hint。`MineRLTreechop-v0` 自身 obs 不含 inventory，所以
   smoke 里 inventory 是空的，是 env 行为不是 adapter bug。
4. **bounded action set（MOVE / CAMERA / ATTACK / USE / PLACE / WAIT）** —
   `Action` 加了 `dx / dz / yaw / pitch / target / slot` payload 字段；
   adapter 在首次 `step()` 时 introspect 实际 action space，只发送 env 真有的
   key，所以 Treechop（无 `place`）和 Navigate（有 `place`）同一段代码通吃。
5. **ModelClient** — `obsidianlink.agents.heuristic_model.HeuristicModelClient`，
   contract 跟未来真 LLM 一样（`str -> str`），返回 JSON 描述 action。
6. **ReactiveAgent** — `obsidianlink.agents.reactive.ReactiveAgent`：
   `Observation -> _build_prompt -> model.complete -> parse_model_response ->
   Action`。model 输出非 JSON / 未知 action 类型一律兜底 WAIT，env loop 不会断。
7. **真实 `Obs -> Agent -> Action -> Minecraft` loop** — `obsidianlink.main`
   默认跑 16 步 live loop，frame mean 从 119.2 走到 116.2，actions 确实在
   Minecraft 里产生了视角变化（`rgb frame changed across loop: True`）。

## Live smoke 运行方式

```bash
conda run -n mc-agent python main.py
# 或在本目录内：
PYTHONPATH=/Users/joey/Documents/Projects/ObsidianLink python main.py
```

无 Java / 无 MineRL 时可跑离线 stub smoke：

```bash
OBSIDIANLINK_OFFLINE=1 conda run -n mc-agent python main.py
```

## 测试

```bash
PYTHONPATH=. conda run -n mc-agent python -m pytest tests/
```

62 个离线单测覆盖：Action payload、bounded action translation（含 Treechop /
Navigate 两套 action space）、HeuristicModelClient cycle、ReactiveAgent 解析
（含 JSON 错误兜底）、inventory 摘要新旧两种 shape、offline env smoke。

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

## Next

Phase 2 — Benchmark MVP：

1. Task（首条诊断任务，按 dev plan 是 D1 / D2 / D3 之一，少量代表性）；
2. BenchmarkRunner（在 Phase 0 的 stub runner 上接 Live env）；
3. Evaluator（独立判断 success，不能进入 agent-visible obs）；
4. Result + 最少 evidence / metrics；
5. 跑通"对一个 Diagnostic task 执行一个真实 Agent 并自动生成结构化结果"。

## Blocked

* `MineRLNavigate-v0` 在本机 Malmo 0.37.0 (`mcprec-6.13.jar:583`,
  `EnvServer.setGameSetttings`) 上有 `NullPointerException`，跟
  `NavigationDecorator` / `RewardForTouchingBlockType` mission XML 分支
  相关。这是 Malmo 服务端 bug，**不是 ObsidianLink 代码问题**，不在本项目
  范围内修。Phase 1 暂时用 `MineRLTreechop-v0` 绕过；Phase 2 起需不需要
  切回 Navigate 由实验设计决定。
