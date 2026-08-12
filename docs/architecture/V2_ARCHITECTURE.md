# v2 Architecture

```text
Real Minecraft Environment
        ↓
Benchmark Kernel
(Task / Observation / Action / Runner / Evaluator / Metrics / Evidence)
        ↓
Agent / Baseline Layer
```

Environment adapter 产生 Agent-visible observation 和独立 evaluator-only state。Runner 只协调生命周期与身份，不把 truth 交给 Agent。Evaluator 只根据 frozen truth/evidence fail closed 判定。Metrics 消费 evaluator verdict，不消费 Agent 文本。Agent/baseline 只能返回经过验证的 MacroAction。

旧 driver/evaluator 仍留在原模块以保持 import compatibility，但 v2 `obsidianlink.benchmark` 与 `obsidianlink.tasks` 不 import 它们。Scripted policy 是 oracle/calibration/regression，不定义 benchmark。
