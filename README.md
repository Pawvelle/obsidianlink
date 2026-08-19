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

**Phase 3 — L1 Controlled Construction（code complete，live partial）**

- L1 scene / task / oracle / evaluator / runner 全部实现
- 267/267 离线单测全过
- Live MineRL 受 Malmo 0.37.0 多项限制：
  - `<Placement>` 不生效，agent 常 spawn 在随机世界点
  - `<ChatCommands> /give` 不执行
  - `PlaceBlock` handler crash server
  - `ObservationFromGrid` 在 8MB XML mission 上持续返回 air
- L1 scene 选用 401×401 obsidian plate workaround + scene 预生成 14-block
  obsidian frame（Malmo 限制下的最小可靠 workaround；Casting + Portal
  Frame Construction 由 scene 完成，agent 只做 Ignition + Nether Entry）
- Oracle live n=1 跑通 env reset + 看到 frame + inventory flint_and_steel，
  但 grid 看不到 obsidian → `success=False, reason=portal_frame_incomplete`

详细进度见 `ROADMAP.md`。受限项见 `ROADMAP.md` 的 Blocked 段。

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

不要再增加 D1 / D2 / D3 diagnostic task。不要提前开发 D4 / D5 / D6。不要把 motor 写回 D2。L1 已完成（code），等下一步指令再决定是否推进 L2 / L3 / L4 或 D4 / D5 / D6。

## 下一阶段

**Phase 3 — Single-Agent Portal Benchmark**

L1 Controlled Construction 已完成（code）。等用户下一步指令决定是否推进
L2 / L3 / L4 或 D4 / D5 / D6。

完整研究与开发计划见 `docs/plans/`。当前进度见 `ROADMAP.md`。
