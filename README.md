# ObsidianLink

一个通过 Nether Portal construction 研究 long-horizon AI Agent 的 Minecraft Benchmark。

## 研究

ObsidianLink 用一个统一核心任务评测 Agent：使用 Minecraft 原版机制构造、激活并进入 Nether Portal。

Formal End-to-End objective：

```text
Construct / complete a Nether Portal
→ Activate it
→ Enter the Nether
```

Success 与具体 solver 无关，由独立 evaluator 依据当前 episode 的真实 Minecraft world truth 判断。Bucket Casting 是第一版受控评测的 **primary reference strategy**，不是 mandatory solver。

Agent 可通过 live Minecraft Wiki 查询任务相关的原版规则、配方和机制；Benchmark prompt 不提供 portal construction recipe。

研究方向：

- Diagnostic
- End-to-End Portal Construction
- Single-Agent
- Multi-Agent
- Generalization & Recovery

该 Benchmark 不是 Minecraft 自动化脚本，也不绑定任何特定模型厂商。

冻结研究与工程规范：

- `docs/plans/ObsidianLink_Research_First_Master_Plan.md`
- `docs/plans/ObsidianLink_Development_Plan.md`

> Current implementation status is maintained in `ROADMAP.md`.

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
