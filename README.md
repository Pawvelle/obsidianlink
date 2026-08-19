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

**Phase 1 — Minimal Minecraft Agent Loop ✅**

**Phase 2 — Benchmark MVP ✅**

```text
D1 Perception   = What is there?
D2 Grounding    = Where is the specified target?
D3 Manipulation = Given the grounded target, can the agent act?
```

Phase 2 正式范围（已关闭，不再扩 diagnostic）：

- **D1 Perception ✅**
  - Lava Presence
  - Water Presence
- **D2 Grounding ✅**（视觉空间 only，无 motor）
  - Direction Grounding
  - Spatial Region Grounding
- **D3 Manipulation ✅**
  - Camera Alignment
  - Target Approach

D1 / D2 / D3 的 live 实验目前都是 **pilot**：用来验证 benchmark pipeline 与 failure attribution，**不作为正式 capability conclusion**。

历史 exploratory D2（把 camera 居中 / walk-and-stop 写进 Grounding）的实验数据继续保留，但不是正式 D2 result。正式对应物是 D3-01 / D3-02。

不要再增加 D1 / D2 / D3 diagnostic task。不要提前开发 D4 / D5 / D6。不要把 motor 写回 D2。

## 下一阶段

**Phase 3 — Single-Agent Portal Benchmark**

下一任务：**L1 Controlled Construction**（尚未开始）。

完整研究与开发计划见 `docs/plans/`。当前进度见 `ROADMAP.md`。
