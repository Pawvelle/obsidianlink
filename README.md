# ObsidianLink

一个通过 Nether Portal construction 研究 long-horizon AI Agent 的 Minecraft Benchmark。

## 研究

ObsidianLink 用一个统一核心任务评测 Agent：使用 Minecraft 原版机制构造、激活并进入 Nether Portal（默认：bucket casting / water-lava）。

研究方向：

- Diagnostic
- End-to-End Portal Construction
- Single-Agent
- Multi-Agent
- Generalization & Recovery

该 Benchmark 与具体 solver 无关。它不是 Minecraft 自动化脚本，也不绑定任何特定模型厂商。

## 当前状态

**D1 Perception Pilot 已完成。下一步：D2 Grounding。**

正式 D1 v2：640×360 受控场景、hidden ground truth、positive/negative。
D1-01 Lava Presence 与 D1-02 Water Presence 均已 live 验证。
旧 inventory D1 与旧 64×64 lava 抓帧仅作 historical pilot，不作为 capability 结论。
不再增加 Obsidian / Iron / Log 等 D1 task。
