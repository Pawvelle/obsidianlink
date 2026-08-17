# ObsidianLink v2.0 项目状态

更新时间：2026-08-17

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

E0–E12 contracts / offline runtimes / MineRL adapters are `unit_verified`. Each now has at least one reviewed real success. None is `integration_verified`; `p1_validation_manifest()` stays `not_run`. The ordered E0–E12 suite orchestrator, OS-level process-release inspection, and explicit already-built JAR mapping remain `unit_verified`. Historical full-suite live pilots: `p1-e0-e12-suite-20260817-001` stopped after E1 (`process_release_not_proven`); `p1-e0-e12-suite-20260817-002` stopped after E4 (`truth_missing`); `p1-e0-e12-suite-20260817-003` stopped after E8 (`truth_missing` / E8 `truth_block_unknown`). P1 Hard Gate: NOT PASSED.

- E0 lifecycle: `runs/p1_e0_reset_close/e0-live-20260813-130313`; E1 RGB: `runs/p1_e1_rgb_observation/e1-live-20260813-162733`.
- E2 inventory: `runs/p1_e2_inventory_observation/e2-live-20260813-232125`; E3 selected item: `runs/p1_e3_selected_item_observation/e3-live-20260814-001`; E4 camera: `runs/p1_e4_camera_control/e4-live-20260814-001`.
- E5 movement: historical attempt `p1-e5-live-001` failed during reset with JVM `SIGSEGV` / Malmo EOF; reviewed success `p1-e5-live-002` produced `movement_ok`. The failure was infrastructure, not movement capability.
- E6 placement: `p1-e6-live-001`, `placement_ok`, target `air -> dirt` from evaluator-only grid truth.
- E7 bucket use: `p1-e7-water-live-001` and `p1-e7-lava-live-001`, both `bucket_ok`; WATER + LAVA coverage complete.
- E8 block truth: historical attempt `p1-e8-live-001` failed before validation with the E5 `liblwjgl_stb` / Sound engine / `STBVorbis` fingerprint; `p1-e8-live-002` produced `block_truth_ok` with `truth_missing_count=0`.
- E9 fluid truth: historical WATER attempt `p1-e9-water-live-001` failed before validation with the same native fingerprint; `p1-e9-water-live-002` and `p1-e9-lava-live-001` produced `fluid_truth_ok`, preserving source/flowing distinction with `truth_missing_count=0`.
- Detailed calibration fields remain in the referenced run evidence and [P1 Environment Validation](docs/architecture/P1_ENVIRONMENT_VALIDATION.md); startup failure diagnosis is preserved in [P1 startup reliability root cause](docs/architecture/P1_STARTUP_RELIABILITY_ROOT_CAUSE.md).
- E10: `unit_verified`；geometry VERIFIED；real conversion reviewed success YES (`p1-e10-live-001`, `obsidian_conversion_ok`)；`integration_verified`: NO.
- E11: `unit_verified`；geometry VERIFIED；real reviewed success YES (`p1-e11-completion-barrier-20260817-004`, `portal_activation_ok`, portal=6/6, retry=0, `tested_action_count=1`, `truth_missing_count=0`)；`integration_verified`: NO.
- E12: `unit_verified`；authorized fixture JAR SHA-256 `f459c36b…`；real reviewed success YES (`p1-e12-dimension-transition-20260817-001`, `dimension_transition_ok`)；before=`minecraft:overworld`，after=`minecraft:the_nether`，retry=0，`tested_action_count=1`，`truth_missing_count=0`；`integration_verified`: NO.

None of E0–E12 is `integration_verified`; `process_release_proven=false`. Typed evaluator truth never enters Agent-visible surfaces. Imports, `--check`, and unit tests do not start MineRL.

## 本次 v2 refactor 已完成

- `unit_verified`：v2 taxonomy、verification vocabulary、P1 manifest、E0–E12 contracts/offline runtimes/MineRL bridges、solver-independent kernel interfaces、catalog quarantine 与文档一致性合同；
- legacy infrastructure 保持原 import，可继续运行离线 regression；
- active catalog 不包含旧 C1–C5 正式 benchmark entries；
- `python -m obsidianlink --check` 与 `scripts/check_environment.py` 使用 v2/P1 语义，不把离线结果提升为真实 integration。标准本地运行时为 `environment.yml` 的 `mc-agent`。

## 尚未验证

- E0–E12 各有一次审查过的真实 success，全部 `integration_verified`: NO；`p1-e0-e12-suite-20260817-001` 在 E1 因 OS `process_release_not_proven` 停止；`p1-e0-e12-suite-20260817-002` 在 E4 因 `truth_missing` 停止（E4 `camera_ok` 但记录未带 `truth_missing_count`）；`p1-e0-e12-suite-20260817-003` 在 E8 因 `truth_missing` 停止（E8 `truth_block_unknown`，`truth_missing_count` 为空）；
- L1–L4、Diagnostic instances、Generalization/Recovery 与 Multi-Agent gameplay 尚未实现；没有 `benchmark_evaluated` 结果。

## P1 hard gate

P1 Hard Gate 尚未通过。一次 E12 成功不是 Hard Gate。进入 P2 前必须完成真实环境 validation suite 的稳定重复成功、`truth_missing=0`、无人工干预。规划建议至少 20 个 fresh episodes。每次真实 MineRL/Minecraft 运行与每次 Gradle 构建仍需用户单独授权。

## 下一精确任务

Startup reliability post-audio validation remains 20/20 observed first-attempt success (`max_reset_attempts=1`); the finite sample does not prove absolute reliability.

Authorized P1 full-suite live pilot `p1-e0-e12-suite-20260817-001` remains historical failed evidence: it stopped after E1 with `process_release_not_proven`. It is not rewritten.

Second authorized live pilot `p1-e0-e12-suite-20260817-002` ran once, with no retry and no Gradle. Canonical JAR `684c20ec…` was verified for E0–E4 (`already_active=true`). E0–E3 succeeded (`lifecycle_ok` / `rgb_ok` / `inventory_ok` / `selected_item_ok`) with OS `process_release_proven=true` and no residual PIDs. E1 retained a full `java -Xmx4G ... -jar ...` identity (the prior `(java)` overwrite did not recur). E4 succeeded scientifically (`camera_ok`, `real_execution_performed=true`, OS `process_release_proven=true`) but `requires_server_truth=true` while the E4 record omitted `truth_missing_count`, so the suite treated it as `truth_missing` and stopped. E5–E12 were not launched. Verdict `truth_missing`. `p1_hard_gate_passed=false`. No human intervention. Evidence: `runs/p1_validation_suite/p1-e0-e12-suite-20260817-002/`. This does not set `integration_verified`.

Third authorized live pilot `p1-e0-e12-suite-20260817-003` ran once, with no retry and no Gradle. Canonical JAR `684c20ec…` was verified for E0–E8 (`already_active=true` after the first step). E0–E7 succeeded: `lifecycle_ok` / `rgb_ok` / `inventory_ok` / `selected_item_ok` / `camera_ok` / `movement_ok` / `placement_ok` / water `bucket_ok` / lava `bucket_ok`. E4–E7 used case-specific evaluator outcomes; they did not require `truth_missing_count`. Every completed step had OS `process_release_proven=true` and no residual PIDs. E8 launched, `real_execution_performed=true`, but failed scientifically with `truth_block_unknown` (`ValueError: unknown block truth`) before stimulus (`tested_action_count=0`, before/after block truth unset, `truth_missing_count=null`). The suite therefore stopped with verdict `truth_missing`. E9–E12 were not launched. `p1_hard_gate_passed=false`. No human intervention. Evidence: `runs/p1_validation_suite/p1-e0-e12-suite-20260817-003/`. This does not set `integration_verified`. Historical pilots `20260817-001` and `20260817-002` remain failed evidence and are not rewritten.

Next: first real blocker is E8 before-truth `truth_block_unknown`; do not treat this pilot as Hard Gate success. A new authorized investigation or Full Suite Pilot #4 may be requested. Do not start P2. P1 Hard Gate: NOT PASSED. P2: NOT STARTED.
