# ObsidianLink Roadmap

> 完整研究与开发计划见 `docs/plans/`。

## Current Phase

**Phase 1 — Minimal Minecraft Agent Loop**

目标：建立 Python、Minecraft、observation 与 action 之间的第一条真实闭环。

## Current Task

**Step 1 — Minimal Real Environment Adapter**

实现最小环境接入，使其能够：

* 启动并 reset Minecraft / MineRL environment；
* 获取一帧真实 RGB observation；
* 执行一次 bounded action；
* 获取下一帧 observation；
* 干净地 close environment。

本步骤不要实现 Benchmark tasks、LLM Agent、planner、evaluator 或 Multi-Agent 功能。

## Completed

* Research direction frozen
* Research-First Master Plan frozen
* Development Plan frozen
* 旧 v2 implementation 已从当前主线移除
* 已建立最小 Research-First 项目骨架
* 已建立最小 Environment / Agent / Task / Evaluator / Runner 接口
* 已建立 offline smoke-test 结构

## Next

真实 environment adapter 跑通之后：

1. 将 RGB observation 接入 Agent-visible 的 `Observation`；
2. 支持第一次 Agent loop 所需的最小 bounded action set；
3. 接入一个真实 `ModelClient`；
4. 跑通第一条真实 `Observation -> Agent -> Action -> Minecraft` loop。

## Blocked

None.
