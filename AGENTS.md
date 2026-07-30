# ObsidianLink Development Constraints

ObsidianLink is a reproducible Minecraft benchmark for single-agent and
two-agent Nether portal construction. Read `README.md`, `ROADMAP.md`, and
`BENCHMARK_SPEC.md` before project work.

1. Work on the active roadmap phase only. Do not add Route B, multi-agent, or
   large benchmark infrastructure before its prerequisite exit criteria pass.
2. Do not independently change the pinned MineRL, Minecraft, Qwen, Python,
   JDK, Gym, NumPy, or model revisions.
3. Model output must pass strict structured parsing, an action allowlist,
   type validation, and numeric clamping. Never execute model-generated code,
   shell commands, raw Minecraft commands, or unbounded input events.
4. Keep environment stepping decoupled from local and remote model inference.
   The environment owner must not wait for planner I/O.
5. Keep agent-visible observations separate from evaluator-only environment
   truth. Evaluator state must never leak into a planner prompt or memory.
6. Every observation, action, message, evaluation event, and log record must
   carry `episode_id`, `agent_id` where applicable, and `step_id`.
7. Prefer evaluator-first vertical slices: prove a task manually or with a
   deterministic driver before using a VLM policy.
8. `vendor/minerl` is an independent nested Git repository. The outer
   repository must not commit, delete, rewrite, or repair its history.
9. MineRL Gradle builds execute third-party code and require explicit user
   approval before each build.
10. Write generated results under `runs/`. Never commit API keys, local model
    weights, raw secrets, or hidden model reasoning.
11. A phase is complete only when its tests, controlled integration evidence,
    automatic evaluation, and required manual review all pass.
12. After structural or functional changes, run the relevant tests and report
    the result and remaining limitations.
