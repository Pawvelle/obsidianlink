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

GeneralAgent 真实 Minecraft smoke（自然语言 → Planner → Skill → MineRL → inventory Observation）：

```bash
PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python \
    obsidianlink/experiments/run_general_agent.py \
    --task "Mine 1 obsidian block"
```

随机自然森林中的 `collect_wood` 诊断可加 `--natural-world`；该路径当前仍不稳定。

LLM smoke（不是 Nether Portal 评测；只验证 MiniMax → JSON action → Minecraft `env.step`）：

```bash
export MINIMAX_API_KEY=your_key
PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python \
    obsidianlink/experiments/run_llm_smoke.py --agent llm --max-steps 8
```

L1 Portal Benchmark（正式 `MineRLL1Controlled-v0` + `L1_PORTAL_TASK` + `L1Evaluator` + `LLMAgent`；默认 500 steps）：

```bash
export MINIMAX_API_KEY=your_key
PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python \
    obsidianlink/experiments/run_l1_llm_agent.py
```

Vision Baseline v3（把 `Observation.frame` RGB 发给 MiniMax-M3）：

```bash
export MINIMAX_API_KEY=your_key
PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python \
    obsidianlink/experiments/run_l1_llm_agent.py --vision
```

`MINIMAX_API_KEY` 只从环境变量读取，不要写入代码。默认 endpoint 为 `https://api.minimaxi.com/v1/chat/completions`。当前实现进度见 `ROADMAP.md`。
