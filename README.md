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

**D3 Manipulation MVP 已完成：D3-01 Camera Alignment + D3-02 Target Approach。**

Diagnostic 固定拆分：

```text
D1 Perception   = What is there?
D2 Grounding    = Where is the specified target?
D3 Manipulation = Given the grounded target, can the agent act?
```

D2（均无 motor，已关闭）：

* **D2-01 Direction Grounding**：left / center / right，hidden GT，`max_steps=1`，WAIT only。
* **D2-02 Spatial Region Grounding**：3×3 区域，仍不做任何 Minecraft 动作。

D3：

* **D3-01 Camera Alignment**（已实现）：目标已可见。Agent 只发 camera / wait，用视觉反馈把岩浆转到画面中央。成功由执行后的 hidden yaw 判定（±12°），不是模型文字声明。
* **D3-02 Target Approach**（已实现）：目标已可见且基本居中。Agent 只发 move / wait，向前走到交互距离后停止。成功由执行后到岩浆 AABB 的 hidden 距离判定（0.6–2.0 格），不是模型文字声明。

正式 D1 v2：640×360 受控场景、hidden ground truth、positive/negative。
D1-01 Lava Presence 与 D1-02 Water Presence 均已 live 验证。

早期把 camera yaw 居中 / 走向目标写进 D2 的实现是错误设计，属于 **historical / exploratory pilot**，不是正式 D2 capability result。Camera 居中已作为正式 **D3-01** 重做；walk-and-stop 已作为正式 **D3-02** 重做。本轮不实现 attack / placement / item use，也不开始 L1。

旧 inventory D1 与旧 64×64 lava 抓帧仅作 historical pilot，不作为 capability 结论。
不再增加 Obsidian / Iron / Log 等 D1 task。
