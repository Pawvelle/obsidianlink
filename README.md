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

**D2 Grounding 已实现：D2-01 Direction + D2-02 Spatial Region（均无 motor）。**

Diagnostic 固定拆分：

```text
D1 Perception   = What is there?
D2 Grounding    = Where is the specified target?
D3 Manipulation = Given the grounded target, can the agent act?
```

D2：

* **D2-01 Direction Grounding**（已实现）：left / center / right，hidden GT，`max_steps=1`，WAIT only。
* **D2-02 Spatial Region Grounding**（已实现）：3×3 区域，仍不做任何 Minecraft 动作。

正式 D1 v2：640×360 受控场景、hidden ground truth、positive/negative。
D1-01 Lava Presence 与 D1-02 Water Presence 均已 live 验证。

早期把 camera yaw 居中 / 走向目标写进 D2 的实现是错误设计，属于 **historical / exploratory pilot**，不是正式 D2 capability result。那些 motor loop 属于未来 D3（Camera Alignment / Target Approach），本轮不实现 D3。

旧 inventory D1 与旧 64×64 lava 抓帧仅作 historical pilot，不作为 capability 结论。
不再增加 Obsidian / Iron / Log 等 D1 task。
