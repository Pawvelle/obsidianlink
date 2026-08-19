# ObsidianLink

一个通过 Nether Portal construction 研究 long-horizon AI Agent 的 Minecraft Benchmark。

## 研究

ObsidianLink 用一个统一核心任务评测 Agent：使用 Minecraft 原版机制构造、激活并进入 Nether Portal。

研究方向：

- Diagnostic
- End-to-End Portal Construction
- Single-Agent
- Multi-Agent
- Generalization & Recovery

该 Benchmark 与具体 solver 无关。它不是 Minecraft 自动化脚本，也不绑定任何特定模型厂商。

研究方向以这两份计划为准：

- `docs/plans/ObsidianLink_Research_First_Master_Plan.md`
- `docs/plans/ObsidianLink_Development_Plan.md`

## 当前状态

**Phase 1 — Minimal Minecraft Agent Loop ✅**

**Phase 2 — Benchmark MVP ✅**（代表性 diagnostic：D1 Lava Presence）

Runner 将 env / agent / evaluator 异常归因到结构化 Result。D1 以 environment-side hidden truth 为准。`Task.allowed_actions` 外的动作不会被执行。

**Phase 3 — Single-Agent Portal Benchmark**

Current Task:

```text
Tool-enabled L1 architecture / L1 Controlled Construction
```

端到端目标：

```text
Construct / complete → activate → enter Nether
```

Bucket Casting 是第一版受控评测的主要 reference strategy，不是 Agent 必须采用的 solver。Agent 可以查询 Minecraft Wiki，获取与任务相关的 Minecraft 规则、配方和机制知识；Benchmark prompt 不提供 portal construction recipe。

不要把 pre-built portal frame 当作正式 L1。旧 L1 实现已从 active path 删除。历史实验 JSON 保留在 `obsidianlink/experiments/runs/`，并标记为 **historical / invalid for L1 capability conclusion**。

## 运行

Phase 1 live smoke（真实 MineRL）：

```bash
PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python main.py
```

Phase 2 representative diagnostic（D1 Lava Presence + Qwen3-VL）：

```bash
PYTHONPATH=. OBSIDIANLINK_PHASE=2 \
  /opt/anaconda3/bin/conda run -n mc-agent python main.py
```

离线 stub：

```bash
OBSIDIANLINK_OFFLINE=1 /opt/anaconda3/bin/conda run -n mc-agent python main.py
```

测试：

```bash
PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python -m pytest tests/
```

详细进度见 `ROADMAP.md`。
