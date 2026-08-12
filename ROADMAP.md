# ObsidianLink v2.0 Nether Portal Construction Roadmap

每个阶段只有在 exit criteria 满足后才能进入下一阶段。每次真实 MineRL/Minecraft、Gradle 和付费 API 操作仍需单独授权。

本文的 P0–P8 只表示工程 Roadmap phase。Environment Validation check 使用 E0–E12，Diagnostic level 使用 D1–D6，End-to-End Portal Construction difficulty 使用 L1–L4。

## P0 — Legacy Freeze / v2 Scope Freeze（完成）

**Goal**：冻结统一 Portal Construction 研究定位，建立 solver-independent 主线并隔离 v1 复杂度。

**Deliverables**：v2 README/spec/taxonomy/status；verification vocabulary；legacy 文档归档；旧 catalog entries quarantine；最小 benchmark/task/agent/multi-agent/P1 validation interfaces；v2 contract tests。

**Exit criteria**：active docs/catalog 一致；旧 C1–C5 不再 benchmark visible；v2 modules 不 import deterministic drivers；离线回归通过。

**Non-goals**：真实 MineRL、Gradle、L1–L4 task implementation、模型 baseline、Multi-Agent gameplay。

## P1 — Real Environment Validation（NEXT）

**Goal**：证明 MineRL action 能可靠到达 Minecraft server，并由 evaluator-only truth 观察原版机制结果。

**Deliverables**：E0–E12 controlled validation cases、版本冻结、证据 bundle、失败 taxonomy、重复运行协议；E10 最小 water/lava calibration。

**Exit criteria**：完整 checklist 稳定重复成功、`truth_missing=0`、无人工干预；建议至少 20 fresh episodes，最终次数在实验合同冻结时确定。

**Non-goals**：正式 end-to-end Agent、排行榜、复杂 recovery、旧 36-step driver 的 live patch。

## P2 — Benchmark Kernel

**Goal**：实现与 solver 解耦的 Task/Observation/Action/Runner/Evaluator/Metrics/Evidence/Splits 内核。

**Deliverables**：v2 task schema、registry、统一 runner 生命周期、evaluator versioning、evidence storage、split contract、replay。

**Exit criteria**：fail-closed kernel integration tests；Agent-visible/evaluator-only 渠道隔离；P1 verified backend 可接入而不依赖某个 policy。

**Non-goals**：批量 task generation、最优 Agent、Multi-Agent 策略。

## P3 — Diagnostic Suite

**Goal**：实现 D1 Perception 至 D6 Recovery 的最小、可审计诊断任务。

**Deliverables**：冻结 task instances、每类 evaluator、scripted oracle/calibration、baseline contract、分层报告。

**Exit criteria**：每个 D-level 有 unit 与真实 integration evidence；diagnostic success 不混入 end-to-end Success Rate。

**Non-goals**：L1–L4 全量长程实验、开放世界 leaderboard。

## P4 — End-to-End Portal Construction

**Goal**：实现 L1 Controlled、L2 Resource Interaction、L3 Resource Acquisition、L4 Open-World Construction。

**Deliverables**：冻结场景/预算/evaluator/entry attribution；全部 level 最终要求 Nether entry。

**Exit criteria**：真实环境可重复运行、episode portal attribution 完整、正式 task/schema/split freeze。

**Non-goals**：把 block/frame/ignition 当作 end-to-end level；依赖单一 deterministic solver 定义任务。

## P5 — Generalization & Recovery

**Goal**：在 seed、spawn、yaw、资源距离/分布、地形、障碍与执行失败下评测 closed-loop recovery。

**Deliverables**：variation profiles、failure injection/observation contracts、state mismatch 与 replan evidence、recovery metrics。

**Exit criteria**：变化与 truth 不泄漏；recovery 由重新观察和后续策略变化证明；跨 seed 报告完整。

**Non-goals**：恢复旧 Casting-vs-Ruined route selection taxonomy。

## P6 — Single-Agent Baselines

**Goal**：在冻结 benchmark 上比较 scripted oracle、reactive 和 planner/model baselines。

**Deliverables**：统一预算、安全 parser、async planner、model/version locks、baseline reports。

**Exit criteria**：相同 task/evaluator/evidence 协议；过期决策丢弃；能力声明符合 verification level。

**Non-goals**：把 scripted oracle 结果当作 Agent capability。

## P7 — Multi-Agent

**Goal**：实现显式通信和私有状态隔离下的协作 Portal Construction。

**Deliverables**：fixed_role、autonomous_role_assignment、Natural Multi-Agent、Compute-Matched Multi-Agent、协调与贡献指标。

**Exit criteria**：跨 Agent 信息只走协议；每个 Agent 身份与贡献可审计；团队成功仍要求可归因 Nether entry。

**Non-goals**：隐式共享 inventory/observation/memory；未匹配计算预算的笼统性能结论。

## P8 — Dataset & Paper Freeze

**Goal**：冻结版本、数据、实验和论文可复现性。

**Deliverables**：train/dev/test release、dataset card、benchmark version、baseline tables、confidence intervals、audit bundle、paper claims matrix。

**Exit criteria**：每条公开 claim 可映射到 `unit_verified`、`integration_verified` 或 `benchmark_evaluated` 证据；无 split leakage；正式归档可重放。

**Non-goals**：在未完成 benchmark experiments 时发布能力结论。
