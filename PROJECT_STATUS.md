# ObsidianLink v2.0 项目状态

更新时间：2026-08-18

## Scope decision

v2 已完成 scope freeze、architecture reset 与 legacy quarantine：

- 唯一研究主题为 Nether Portal Construction；
- Diagnostic、End-to-End、Generalization & Recovery 是评测维度；
- Single-Agent / Multi-Agent 是正交 execution modes；
- Benchmark kernel 与 Agent/baseline 解耦；
- 正式 end-to-end success 必须是当前 episode portal 的可归因 Nether entry；
- 旧 deterministic progression 只作为 scripted oracle、calibration 或 regression fixture。

旧完整状态记录位于 [docs/legacy/v1/PROJECT_STATUS_V1.md](docs/legacy/v1/PROJECT_STATUS_V1.md)。该文件是历史证据，不是 active scope。

## P0.1 architecture cleanup（offline complete）

- Roadmap phase 保持 P0–P8，Environment Validation 保持 E0–E12，Diagnostic level 保持 D1–D6；
- End-to-End difficulty 已统一为 L1–L4；所有 L-level 仍以 attributed Nether entry 为最终成功；
- 历史 `TaskInstance` 是 v1 compatibility surface（`LegacyTaskInstance` alias）；v2 taxonomy 使用 `TaskIdentity`；
- architecture guards 覆盖 benchmark/tasks 对 deterministic drivers、legacy evaluators、model-specific agents 与 legacy TaskInstance 的依赖。

## 当前唯一 active task

`P1-REAL-MINERL-ENVIRONMENT-VALIDATION`

P1 real MineRL environment validation remains the only active phase.

目标：建立并真实验证最小、可重复、可审计的 MineRL/Minecraft 实验仪器。冻结清单：E0 reset/close、E1 RGB、E2 inventory、E3 selected item、E4 camera、E5 movement、E6 block placement、E7 bucket、E8 block truth、E9 fluid truth、E10 vanilla water-lava -> obsidian、E11 portal activation、E12 dimension transition。

E10 是 calibration，不是正式 benchmark task。Evaluator 必须同时观察 water-pour cell 出现 water/source，以及相邻 lava source 变为 obsidian。

## P1 E0–E12 active status

E0–E12 contracts / offline runtimes / MineRL adapters are `unit_verified`. Each now has at least one reviewed real success. None is `integration_verified`; `p1_validation_manifest()` stays `not_run`. The ordered E0–E12 suite orchestrator, OS-level process-release inspection, and explicit already-built JAR mapping remain `unit_verified`. Historical full-suite live pilots: `p1-e0-e12-suite-20260817-001` stopped after E1 (`process_release_not_proven`); `p1-e0-e12-suite-20260817-002` stopped after E4 (`truth_missing`); `p1-e0-e12-suite-20260817-003` stopped after E8 (`truth_missing` / E8 `truth_block_unknown`). Fourth authorized live pilot `p1-e0-e12-suite-20260817-004` completed all 15 steps with suite verdict `hard_gate_success`. This is one real complete P1 suite success; it does not set `integration_verified`. Stability campaign `p1-stability-20260817`: 0/20 full-suite successes, all 20 stopped at E0 `reset_failed`. Repeatability required to leave P1 is still open. P2 Benchmark Kernel v2 = FROZEN; remaining P2 steps not started.

- E0 lifecycle: `runs/p1_e0_reset_close/e0-live-20260813-130313`; E1 RGB: `runs/p1_e1_rgb_observation/e1-live-20260813-162733`.
- E2 inventory: `runs/p1_e2_inventory_observation/e2-live-20260813-232125`; E3 selected item: `runs/p1_e3_selected_item_observation/e3-live-20260814-001`; E4 camera: `runs/p1_e4_camera_control/e4-live-20260814-001`.
- E5 movement: historical attempt `p1-e5-live-001` failed during reset with JVM `SIGSEGV` / Malmo EOF; reviewed success `p1-e5-live-002` produced `movement_ok`. The failure was infrastructure, not movement capability.
- E6 placement: `p1-e6-live-001`, `placement_ok`, target `air -> dirt` from evaluator-only grid truth.
- E7 bucket use: `p1-e7-water-live-001` and `p1-e7-lava-live-001`, both `bucket_ok`; WATER + LAVA coverage complete.
- E8 block truth: historical attempt `p1-e8-live-001` failed before validation with the E5 `liblwjgl_stb` / Sound engine / `STBVorbis` fingerprint; `p1-e8-live-002` produced `block_truth_ok` with `truth_missing_count=0`. Targeted diagnostic `e8-live-20260817-001` also produced `block_truth_ok`; it does not rewrite suite Pilot #3 and is not `integration_verified`.
- E9 fluid truth: historical WATER attempt `p1-e9-water-live-001` failed before validation with the same native fingerprint; `p1-e9-water-live-002` and `p1-e9-lava-live-001` produced `fluid_truth_ok`, preserving source/flowing distinction with `truth_missing_count=0`.
- Detailed calibration fields remain in the referenced run evidence and [P1 Environment Validation](docs/architecture/P1_ENVIRONMENT_VALIDATION.md); startup failure diagnosis is preserved in [P1 startup reliability root cause](docs/architecture/P1_STARTUP_RELIABILITY_ROOT_CAUSE.md).
- E10: `unit_verified`；geometry VERIFIED；real conversion reviewed success YES (`p1-e10-live-001`, `obsidian_conversion_ok`)；`integration_verified`: NO.
- E11: `unit_verified`；geometry VERIFIED；real reviewed success YES (`p1-e11-completion-barrier-20260817-004`, `portal_activation_ok`, portal=6/6, retry=0, `tested_action_count=1`, `truth_missing_count=0`)；`integration_verified`: NO.
- E12: `unit_verified`；authorized fixture JAR SHA-256 `f459c36b…`；real reviewed success YES (`p1-e12-dimension-transition-20260817-001`, `dimension_transition_ok`)；before=`minecraft:overworld`，after=`minecraft:the_nether`，retry=0，`tested_action_count=1`，`truth_missing_count=0`；`integration_verified`: NO.

None of E0–E12 is `integration_verified`. Object-level `E0CleanupStatus.process_release_proven` remains false by design. Pilot #4 suite OS-level `process_release_proven=true` for every step. Typed evaluator truth never enters Agent-visible surfaces. Imports, `--check`, and unit tests do not start MineRL.

## 本次 v2 refactor 已完成

- `unit_verified`：v2 taxonomy、verification vocabulary、P1 manifest、E0–E12 contracts/offline runtimes/MineRL bridges、solver-independent kernel interfaces、catalog quarantine 与文档一致性合同；
- legacy infrastructure 保持原 import，可继续运行离线 regression；
- active catalog 不包含旧 C1–C5 正式 benchmark entries；
- `python -m obsidianlink --check` 与 `scripts/check_environment.py` 使用 v2/P1 语义，不把离线结果提升为真实 integration。标准本地运行时为 `environment.yml` 的 `mc-agent`。

## 尚未验证

- E0–E12 各有一次审查过的真实 success，全部 `integration_verified`: NO；`p1-e0-e12-suite-20260817-001` 在 E1 因 OS `process_release_not_proven` 停止；`p1-e0-e12-suite-20260817-002` 在 E4 因 `truth_missing` 停止（E4 `camera_ok` 但记录未带 `truth_missing_count`）；`p1-e0-e12-suite-20260817-003` 在 E8 因 `truth_missing` 停止（E8 `truth_block_unknown`，`truth_missing_count` 为空）；`p1-e0-e12-suite-20260817-004` 是一次真实完整 E0→E12 成功，仍不设 `integration_verified`；stability campaign `p1-stability-20260817` 为 0/20 完整 suite success，20/20 在 E0 `reset_failed`；
- L1–L4、Diagnostic instances、Generalization/Recovery 与 Multi-Agent gameplay 尚未实现；没有 `benchmark_evaluated` 结果。
- P2 Benchmark Kernel v2 已 FROZEN（offline `unit_verified`）；尚未在 `integration_verified` 或 `benchmark_evaluated` 等级上验证；`p1_validation_manifest()` 仍为 `not_run`。

## P1 hard gate

Pilot #4 的 suite 字段为 `verdict=hard_gate_success`、`p1_hard_gate_passed=true`。这只证明当前冻结版本获得一次真实完整 P1 suite 成功运行，不把 E0–E12 提升为 `integration_verified`，也不关闭进入 P2 所需的重复性要求。Stability campaign `p1-stability-20260817` 在同一冻结 commit 上得到 0/20 完整 suite success。进入完整 P2 task instance / Multi-Agent / L1–L4 推进前必须完成真实环境 validation suite 的稳定重复成功、`truth_missing=0`、无人工干预。仓库尚未冻结数值通过阈值（`repeatability_threshold_not_frozen`）。规划建议至少 20 个 fresh episodes。P2 Benchmark Kernel v2 已 offline FROZEN；该步骤不依赖 P1 hard gate，也未把任何 E0–E12 标为 `integration_verified`。完整 P2 推进仍以 P1 重复性要求为准。每次真实 MineRL/Minecraft 运行与每次 Gradle 构建仍需用户单独授权。

## 下一精确任务

Startup reliability post-audio validation remains 20/20 observed first-attempt success (`max_reset_attempts=1`); the finite sample does not prove absolute reliability.

Authorized P1 full-suite live pilot `p1-e0-e12-suite-20260817-001` remains historical failed evidence: it stopped after E1 with `process_release_not_proven`. It is not rewritten.

Second authorized live pilot `p1-e0-e12-suite-20260817-002` ran once, with no retry and no Gradle. Canonical JAR `684c20ec…` was verified for E0–E4 (`already_active=true`). E0–E3 succeeded (`lifecycle_ok` / `rgb_ok` / `inventory_ok` / `selected_item_ok`) with OS `process_release_proven=true` and no residual PIDs. E1 retained a full `java -Xmx4G ... -jar ...` identity (the prior `(java)` overwrite did not recur). E4 succeeded scientifically (`camera_ok`, `real_execution_performed=true`, OS `process_release_proven=true`) but `requires_server_truth=true` while the E4 record omitted `truth_missing_count`, so the suite treated it as `truth_missing` and stopped. E5–E12 were not launched. Verdict `truth_missing`. `p1_hard_gate_passed=false`. No human intervention. Evidence: `runs/p1_validation_suite/p1-e0-e12-suite-20260817-002/`. This does not set `integration_verified`.

Third authorized live pilot `p1-e0-e12-suite-20260817-003` ran once, with no retry and no Gradle. Canonical JAR `684c20ec…` was verified for E0–E8 (`already_active=true` after the first step). E0–E7 succeeded: `lifecycle_ok` / `rgb_ok` / `inventory_ok` / `selected_item_ok` / `camera_ok` / `movement_ok` / `placement_ok` / water `bucket_ok` / lava `bucket_ok`. E4–E7 used case-specific evaluator outcomes; they did not require `truth_missing_count`. Every completed step had OS `process_release_proven=true` and no residual PIDs. E8 launched, `real_execution_performed=true`, but failed scientifically with `truth_block_unknown` (`ValueError: unknown block truth`) before stimulus (`tested_action_count=0`, before/after block truth unset, `truth_missing_count=null`). The suite therefore stopped with verdict `truth_missing`. E9–E12 were not launched. `p1_hard_gate_passed=false`. No human intervention. Evidence: `runs/p1_validation_suite/p1-e0-e12-suite-20260817-003/`. This does not set `integration_verified`. Historical pilots `20260817-001` and `20260817-002` remain failed evidence and are not rewritten.

Authorized targeted E8 diagnostic `e8-live-20260817-001` produced `block_truth_ok` (`truth_missing_count=0`, before all air, after target dirt, controls unchanged, `tested_action_count=1`, `anchor_source=expected_spawn_fallback`, `grid_anchor_world=(0, 4, 0)`, position `(0.5, 4.0, 0.5)`, dimension `minecraft:overworld`). `unknown_block_diagnostics` was null because unknown-block did not occur. Canonical JAR `684c20ec…` was already active; Gradle was not invoked. Object-level cleanup `process_release_proven=false` as designed; OS process table had no residual Java/MineRL PIDs after close. This does not rewrite Pilot #3 and does not set `integration_verified`.

Fourth authorized live pilot `p1-e0-e12-suite-20260817-004` ran once, with no retry and no Gradle. E0–E10 used already-active canonical `684c20ec…`. E11 switched to completion-barrier `6b5705e4…`. E12 switched to portal-fixture `f459c36b…`. All 15 steps executed and succeeded: `lifecycle_ok` / `rgb_ok` / `inventory_ok` / `selected_item_ok` / `camera_ok` / `movement_ok` / `placement_ok` / water `bucket_ok` / lava `bucket_ok` / `block_truth_ok` / water `fluid_truth_ok` / lava `fluid_truth_ok` / `obsidian_conversion_ok` / `portal_activation_ok` / `dimension_transition_ok`. E8 did not reproduce `truth_block_unknown` (`tested_action_count=1`, before air/air/air, after dirt/air/air, `truth_missing_count=0`, `unknown_block_diagnostics=null`, `anchor_source=expected_spawn_fallback`). E8–E12 all had `truth_missing_count=0`. Every step had OS `process_release_proven=true` and empty residual PIDs. `real_execution_performed=true`. `human_intervention=false`. Suite verdict `hard_gate_success`; this run’s `p1_hard_gate_passed=true`. `integration_verified=false`; `p1_validation_manifest()` remains `not_run`. Evidence: `runs/p1_validation_suite/p1-e0-e12-suite-20260817-004/`. Historical pilots `20260817-001`, `20260817-002`, and `20260817-003` remain failed evidence and are not rewritten.

Authorized P1 full-suite stability campaign `p1-stability-20260817` ran 20 new fresh suites on frozen git `28583f8`, with Pilot #4 excluded. No retry, no Gradle, no production-code change, no human intervention. All 20 stopped at E0 with `reset_failed` / `validation_failed`. The identical error was `MineRL reset failed after 1 attempts: a bytes-like object is required, not 'NoneType'`. Canonical runtime SHA `684c20ec…` verified on every E0. Executed-step OS `process_release_proven=true` and residual PIDs empty on all 20; suite-level `process_release_proven` is false because the 15-step contract was incomplete. E1–E12 were never launched. E8 `truth_block_unknown` did not recur because E8 was not reached. Empirical full-suite success rate 0/20. Longest success streak 0. Repeatability threshold is not frozen (`repeatability_threshold_not_frozen`). This does not set `integration_verified` and does not start P2. Evidence: `runs/p1_stability_campaign/p1-stability-20260817/` and `runs/p1_validation_suite/p1-stability-20260817-001` through `-020`. Historical Pilot #1–#4 are not rewritten.

当前完整 P1 suite 仍只有 Pilot #4 一次真实完整成功。本次 20-run campaign 未产生任何完整 suite success。Next: 不要开始 D1 / P3 / 真实 Agent / MineRL attach；不要把 E0–E12 标为 `integration_verified`；不要发明通过阈值。若要调查 E0 `NoneType` reset 或授权新的 campaign，必须另开任务，不得并入本次样本。该 campaign 当时未开始 P2 task instances；P2 Benchmark Kernel 现已独立 offline FROZEN。下一工程任务回到 P1 suite repeatability Hard Gate。

## P2 — Benchmark Kernel v2

**P2 Benchmark Kernel v2 = FROZEN** (2026-08-18 freeze audit). Remaining P2 Roadmap work (D1–D6 / L1–L4 / Multi-Agent / real-backend attach) is not started and is not the next engineering task. The kernel freeze is offline-only; no LLM, no MineRL/Minecraft, no Gradle, no paid API. P3 is not started.

### P2 step 1 — minimal kernel freeze (commit `a5131dc`)

Step 1 reused every pre-existing v2 primitive and added only the missing aggregate and a single driver:

- `TaskIdentity` (frozen taxonomy: suite / mode / level / layout / family);
- `Evaluator` Protocol + `EvaluatorVerdict` (fail-closed);
- `EvidenceRecord` + `EvidenceIdentity` + `EvidenceChannel` (`AGENT_VISIBLE` / `EVALUATOR_ONLY`);
- `MetricRecord` + `MetricName` (finite-numeric, optional numerator / denominator);
- `BenchmarkRunner` Protocol + `RunnerResult` (status ∈ {completed, blocked, failed});
- `VerificationLevel` (`unit_verified` / `integration_verified` / `benchmark_evaluated`);
- `BenchmarkSplit` (train / dev / test);
- `Observation` / `MacroAction` / `BackendStep` from `obsidianlink.core.types` (referenced via duck typing; the kernel module does not import them, so the v2 architecture guard `obsidianlink.benchmark` ≠ `obsidianlink.core.types` stays intact);
- `Agent` Protocol from `obsidianlink.agents.base` (referenced by duck typing).

New types in this step:

- `obsidianlink.benchmark.run_record.BenchmarkRunRecord` — frozen aggregate dataclass: `task`, `runner_status`, `verdict`, `evidence`, `metrics`, `verification_level`, `schema_version` (`p2.benchmark.run_record.v1`). Success verdict requires `runner_status == 'completed'`.
- `obsidianlink.benchmark.run_record.run_benchmark` — single function that drives `task -> (backend, agent, evaluator) -> BenchmarkRunRecord` without any LLM call. Writes observation and action evidence on `AGENT_VISIBLE`, and runner_error / close_error / verdict on `EVALUATOR_ONLY`. Projects observations to a whitelisted agent-visible surface so backend evaluator-only data never reaches the agent.
- `obsidianlink.benchmark.run_record.write_run_record` / `load_run_record` — JSON round-trip helpers keyed by `schema_version`.

No new abstract base class, registry, plugin system, dependency-injection container, or framework. `obsidianlink.benchmark` is still solver-independent and model-independent.

### P2 step 1 — no-LLM / no-MineRL stub flow (offline unit-verified)

`tests/test_p2_benchmark_kernel.py` (8 cases, all passing) proves the kernel can:

1. load a minimal `TaskIdentity` (Diagnostic / Single / D1 / Controlled);
2. drive a stub `Agent` (`act(observation) -> MacroAction.wait()`) against an in-memory stub `Backend`;
3. accept a stub `Evaluator` (`evaluate(state) -> EvaluatorVerdict(success=True, outcome='kernel_smoke_ok', evidence_complete=True)`);
4. record the episode as a `BenchmarkRunRecord` (task, verdict, evidence, metrics, runner_status, verification_level);
5. round-trip the record through `write_run_record` → `load_run_record` (schema `p2.benchmark.run_record.v1`);
6. attach `MetricRecord` entries via an optional `metrics_hook` (e.g. `MetricName.ENVIRONMENT_STEPS`);
7. keep evaluator-only payload off the agent-visible surface (`'hidden'` in `initial_evaluator_state` does not appear in any `AGENT_VISIBLE` evidence payload; backend's hidden `frame` slot is dropped by the projection);
8. fail-closed on contract violation (non-`MacroAction` from the agent → `runner_status='failed'`, runner_error recorded on `EVALUATOR_ONLY`).

P1 historical evidence is preserved unchanged: 4 pilots + Full Suite #5 + stability campaign `p1-stability-20260817` 0/20 + campaign-path recheck `campaign-path-recheck-20260817-001` + single-Full-Suite recheck `p1-e0-e12-suite-20260817-005` are all untouched in `runs/`. None of these are reclassified as `integration_verified`. None of the P1 E0–E12 contracts, runtime switching, process-release observer, evaluator truth rules, or hard-gate logic were modified. P1 hard gate is still open (stability repeatability not frozen).

### P2 freeze audit (2026-08-18) — FROZEN

Audit used current code as the only source of truth. No D1/L1/Multi-Agent work, no MineRL, no Gradle, no LLM/API.

- **Git SHA:** `84c6c23` (`docs(p2): freeze audit mark Benchmark Kernel v2 FROZEN`). Includes the `BenchmarkRunRecord.as_dict()` `verdict.identity.agent_id` round-trip completeness fix on top of committed kernel `77c6c4e`.
- **Schema version:** `p2.benchmark.run_record.v1`
- **Core interfaces:** `TaskIdentity` (`suite` / `mode` / `level` / `layout` / `family`); `Evaluator` Protocol + `EvaluatorVerdict`; `EvidenceRecord` + `EvidenceIdentity` + `EvidenceChannel` (`AGENT_VISIBLE` / `EVALUATOR_ONLY`); `MetricRecord` + `MetricName`; `BenchmarkRunner` Protocol + `RunnerResult`; `VerificationLevel`; `BenchmarkSplit`; `BenchmarkRunRecord`; `run_benchmark` / `write_run_record` / `load_run_record`
- **Tests:** `tests/test_p2_benchmark_kernel.py` (11 cases, all passing) plus `tests/test_v2_architecture.py` (kernel import / taxonomy / channel guards)
- **Information boundary:** `run_benchmark` writes observation/action on `AGENT_VISIBLE` and runner_error / close_error / verdict on `EVALUATOR_ONLY`; observation evidence is a whitelist projection; `initial_evaluator_state` is forwarded only to the evaluator; `obsidianlink.benchmark` does not import `obsidianlink.core.types`, P1 validation types, MineRL, or agents
- **Fail-closed guarantees:** `EvaluatorVerdict.success` requires `evidence_complete`; `BenchmarkRunRecord` rejects `success` unless `runner_status == 'completed'`; unsupported `schema_version` refused on load; agent contract violation → `runner_status='failed'` + `runner_error`; `close()` failure downgrades `completed` → `failed` and records `close_error`; dishonest success after close failure is rejected by record construction; `terminated` / `truncated` stop the step loop without another `backend.step()`
- **Not claimed:** `integration_verified`, `benchmark_evaluated`, P1 hard-gate close, E0–E12 reclassification, D1–D6 / L1–L4 / Multi-Agent instances, replay helper, or real-backend attach

### P2 remaining steps (not started; not next)

Do not start D1, P3, real Agent, or MineRL attach until P1 repeatability Hard Gate is closed.

- Freeze one minimal D1 perception task instance and register it in the catalog.
- Drive that D1 instance through the same `run_benchmark` with a stub oracle and with a real Agent (no MineRL yet).
- Connect the kernel to a real MineRL adapter (`MineRLEnvironmentBackend`) and re-run the D1 instance under `VerificationLevel.INTEGRATION_VERIFIED` (requires P1 hard gate).
- Add a replay helper that consumes a `BenchmarkRunRecord` and reconstructs the agent-visible trace.

Next engineering task: return to P1 and close the suite repeatability Hard Gate. E10 remains calibration (`unit_verified`, reviewed real success YES, `integration_verified` NO); the open gate is full E0–E12 suite stability, not a new E10 feature.
