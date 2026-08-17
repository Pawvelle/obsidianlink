# P1 Real MineRL Environment Validation

P1 验证实验仪器，不评测 Agent。所有 case 都是 calibration/integration checks，不进入正式 Success Rate。

## Checklist

| ID | Validation target |
|---|---|
| E0 | reset / close lifecycle |
| E1 | RGB observation |
| E2 | inventory observation |
| E3 | selected item |
| E4 | camera control |
| E5 | movement |
| E6 | block placement |
| E7 | bucket usage |
| E8 | server-side block truth |
| E9 | water/lava fluid truth |
| E10 | vanilla water-lava -> obsidian |
| E11 | portal activation |
| E12 | dimension transition |

每个 case 保存版本、episode/step/agent identity、Agent action、server truth、verdict、异常与 close 状态。客户端图像只能用于诊断，不能替代 E8–E12 server truth。

## E10 controlled calibration

E10 验证 MineRL `use_item(water_bucket)` → Minecraft 1.16.5 vanilla water/lava 更新 → evaluator-only server truth 观察到目标格变为 `obsidian`。它不是 Agent benchmark、portal construction、E11 点火或 E12 维度切换。

冻结几何（spawn-relative / `ObservationFromGrid(atSpawn=true)`）：

- spawn `(0, 4, 0)`，yaw `0.0`，pitch `60.0`（复用已验证的 E7 姿态）
- water-pour cell `(0, 4, 1)` / grid `(0, 0, 1)`；before 必须是 `air` / no fluid
- target lava-source cell `(0, 4, 2)` / grid `(0, 0, 2)`；禁止预置目标 obsidian
- control cells `(0, 5, 1)`、`(0, 5, 2)`
- E10-only Mission XML `DrawBlock`：world `(0, 4, 2)` = `lava`；默认 `PortalA0EnvSpec` 仍无 DrawingDecorator
- 平台支撑 `(0, 3, 2)` 来自已部署 `prepareControlledBuildArea` 的 grass_block，不是 XML、不是 control
- 恰好 1 次 stimulus：`use_item(water_bucket)`，`duration_ticks=1`
- bounded observation window：最多 5 个 `wait` tick
- 成功必须同时证明：after water-pour cell = water/source，after target = obsidian，controls unchanged，`truth_missing_count=0`
- 成功 outcome：`obsidian_conversion_ok`
- 水桶已 accepted 但 after water cell 不是 water/source：`water_placement_not_observed`
- water 已出现但 lava 未变 obsidian：`conversion_not_observed`

```text
MineRL Action
  -> Minecraft Server
  -> Vanilla Mechanics
  -> Evaluator-only World Truth
  -> Verdict
```

E10 controlled initial geometry：Mission XML `DrawingDecorator` **implemented / offline verified**，runtime application **deployed / geometry-smoke verified**. `prepareControlledBuildArea` still fills `feetY` and above with air, so DrawBlock is applied **after** that platform step and awaited after skip-first-frames. Allowlist is lava source only; obsidian DrawBlock fails closed. Evidence: [P1 E10 runtime geometry](P1_E10_RUNTIME_GEOMETRY.md), `runs/p1_e10_obsidian_conversion/e10-geometry-20260816-002`, outcome `e10_geometry_ready`. This is not `obsidian_conversion_ok` and does not set `integration_verified`.

E10 vanilla causality：water placement + lava-source conversion **both required**。FakeBackend success 不能证明真实 Minecraft 能力。

One reviewed real conversion: `p1-e10-live-001`, outcome `obsidian_conversion_ok`. Before: water `(0, 4, 1)` = air/none, target `(0, 4, 2)` = lava/source. Stimulus: exactly one accepted `use_item(water_bucket)`. After: water = water/source, target = obsidian, controls unchanged, `truth_missing_count=0`. Compact evidence: `runs/history/p1-e10-live-20260816-001/`. This is not portal activation and does not set `integration_verified`.

E10 通过不表示正式 portal task 成功。当前 verification：contract/offline runtime/MineRL bridge = `unit_verified`；E10 geometry real verified = **YES**；E10 real conversion reviewed success = **YES**；`integration_verified` = **NO**。

## E11 controlled calibration

E11 只验证：完整合法 obsidian portal frame + 一次真实 `use_item(flint_and_steel)` + Minecraft 1.16.5 vanilla portal mechanics → interior 出现 `nether_portal` → evaluator-only server truth → `portal_activation_ok`。

它不是 Agent task、不是 E10 再造 obsidian、不是玩家进入 portal、不是 E12 dimension change。预置 frame 是 **calibration fixture**，不能宣称 Agent built the portal，也不能当作 end-to-end portal construction success。

冻结几何（1.16.5 MCP-Reborn `PortalSize`：interior width 2–21、height 3–21；最小 interior 2×3，外框 4×5 完整 14 格 obsidian；axis X，平面 z=1）：

- spawn `(0, 4, 0)`，yaw `0.0`，pitch `60.0`
- frame：x=-1..2，y=3..7，z=1（14 obsidian）
- interior：`(0|1, 4|5|6, 1)` 共 6 格，before 必须是 air，禁止预置 portal/fire
- ignition cell `(0, 4, 1)`
- controls `(0, 8, 1)`、`(0, 4, 3)`
- inventory：仅 `flint_and_steel:1`
- 恰好 1 次 stimulus：`use_item(flint_and_steel)`，`duration_ticks=1`
- bounded observation window：最多 3 个 wait tick（工程缓冲，不是科学阈值）
- 成功必须：6 格 interior 均为 portal block（`nether_portal`，兼容 Malmo `portal` 别名），controls unchanged，dimension 仍为 overworld，`truth_missing_count=0`
- fire 单独出现不是成功（`portal_activation_not_observed`）
- 成功 outcome：`portal_activation_ok`

Mission XML 可以预置 obsidian frame（`allow_obsidian_frame_fixture=True`）。Deployed DrawingDecorator allowlist is lava + obsidian; portal/fire remain forbidden. Geometry evidence: [P1 E11 runtime geometry](P1_E11_RUNTIME_GEOMETRY.md), `runs/p1_e11_portal_activation/e11-geometry-20260816-001`, outcome `e11_geometry_ready`. This is not `portal_activation_ok` and does not set `integration_verified`.

FakeBackend success 不能证明真实 Minecraft activation。E11 `unit_verified` 不把 `integration_verified` 设为 true。

One real E11 activation: `p1-e11-live-001`, outcome `portal_activation_not_observed`. Before: 14/14 obsidian, 6/6 interior air, portal=0, fire=0, ignition `(0, 4, 1)` = air. Stimulus: exactly one accepted `use_item(flint_and_steel)`. After: ignition cell = `fire`, 0/6 `nether_portal`, controls unchanged, `truth_missing_count=0`. Fire ≠ success. Compact evidence: `runs/history/p1-e11-live-20260816-001/`. Offline PortalSize replica of that snapshot: Axis.X **PASS** (width 2, height 3, bottomLeft `(1, 4, 1)`); Axis.Z invalid as expected. Instrumented diagnostic clone `p1-e11-diag-001` (not a formal benchmark result): Case F / `ROOT_CAUSE_NARROWED`. Write-path clone `p1-e11-diag-002`: `placePortalBlocks` writes `ServerWorld` (`isRemote=false`) from the Render thread; 6/6 `setBlockState` accepted with immediate `nether_portal`; `updatePostPlacement` did not run; evaluator after still 0/6. Evidence: [P1 E11 diagnostic runtime](P1_E11_DIAGNOSTIC_RUNTIME.md), `runs/history/p1-e11-diagnostic-20260817-002/`. Authorized marshal live #2 (`p1-e11-live-002`): wait before `addAction` timed out; `tested_action_count=0`. Compact: `runs/history/p1-e11-live-20260817-001/`. Nonblocking marshal live #3 (`p1-e11-live-003`): queued `server.execute` after `addAction` then still timed out before `waitForNextObservation`; `processRightClickBlock` never ran; `tested_action_count=0`. Compact: `runs/history/p1-e11-live-20260817-002/`. Await-after-tick live #4 (`p1-e11-live-004`): awaited after `waitForNextObservation` (`stepClient:772`); still no `processRightClickBlock`; `tested_action_count=0`. Compact: `runs/history/p1-e11-live-20260817-003/`. Details: [P1 E11 server-thread marshal](P1_E11_SERVER_THREAD_MARSHAL.md). This is not portal activation capability and does not set `integration_verified`.

Canonical live `p1-e11-canonical-runtime-20260817-002` rebuilt from frozen source and excluded `mcp_patch.diff`, all old E11 marshal/paused-executor/diagnostic patches, and E12. Production semantic diff is clean: only `IntegratedServer.class` plus `version.properties`; EnvServer, SoundEngine, PortalSize, FlintAndSteelItem, Entity, and ServerPlayerEntity are byte-identical. The environment-mode integrated server remained unpaused (`Saving and pausing game` absent). One fresh process, one reset, exactly one accepted `use_item(flint_and_steel)`, and zero retry produced client-visible fire but 0/6 `nether_portal`; `truth_missing_count=0`, outcome `portal_activation_not_observed`. Compact evidence: `runs/history/p1-e11-canonical-20260817-002/`.

## E12 controlled calibration

E12 只验证：已激活的 Nether portal fixture + 一次有界 `move(forward=1)` + vanilla portal wait → evaluator-only `portal_dimension` 从 `minecraft:overworld` 变为 `minecraft:the_nether` → `dimension_transition_ok`。

它不是 portal construction、不是 E11 ignition、不是 Agent task、不是 end-to-end success。预置 active portal 是 **calibration fixture**。Nether 之后不再用 Overworld portal grid 作为 after-truth。

冻结几何复用 E11 4×5 / interior 2×3 框，但 interior 在 reset 时已是 Malmo `portal` DrawBlock（Java `Blocks.NETHER_PORTAL`；ObservationFromGrid 仍报 `nether_portal`）：

- spawn `(0, 4, 0)`，yaw `0.0`，pitch `0.0`（平视走进 portal 平面 z=1）
- frame：14 obsidian；interior：6 portal；controls `(0, 8, 1)`、`(0, 4, 3)` 为 air
- inventory：惰性 `dirt:1`（规范禁止空背包）
- 恰好 1 次 stimulus：`move`，`duration_ticks=8`，`forward=1`，无 strafe/sprint/jump
- bounded observation window：最多 100 个 wait tick（vanilla ~80 tick + MineRL 缓冲）
- 成功必须：before dimension=`minecraft:overworld`，after dimension=`minecraft:the_nether`，before portal=6/6，frame=14/14，`truth_missing_count=0`
- 成功 outcome：`dimension_transition_ok`

Mission XML 使用 `allow_active_portal_fixture=True`。Canonical DrawingDecorator 仍拒绝 portal；`patches/minerl/e12-drawing-decorator-portal.patch` 不进入 `CANONICAL_PATCHES`。Authorized E12 fixture JAR SHA-256 `f459c36b7aaacd7e5f98ff9bbe001f1d54e77b73740537c24d5c5540290d36f4` maps Malmo `portal` to `Blocks.NETHER_PORTAL` and still rejects fire/end portal. Vanilla PortalSize / NetherPortalBlock / transition classes are unchanged.

One reviewed real E12 transition: `p1-e12-dimension-transition-20260817-001`, outcome `dimension_transition_ok`. Before: dimension=`minecraft:overworld`, 14/14 obsidian, 6/6 `nether_portal`. Stimulus: exactly one accepted `move(forward=1, duration_ticks=8)`, retry=0. After: dimension=`minecraft:the_nether`, `truth_missing_count=0`. Evidence: `runs/p1_e12_dimension_transition/e12-live-20260817-001/`. This is not Agent construction, not E11 ignition, and does not set `integration_verified`.

FakeBackend / offline stub success 不能证明真实 Minecraft 维度切换。E12 `unit_verified` 不把 `integration_verified` 设为 true。

## Startup reliability hardening

- P1 startup reliability hardening: **COMPLETE**.
- Post-audio fresh-process validation: **20 / 20 observed success**.
- `max_reset_attempts`: **1**; each attempt used a fresh Python process.
- Startup infrastructure failures observed post-mitigation: **0**.
- Native crash recurrence: **0**; Malmo EOF recurrence: **0**.
- Evidence: `runs/history/p1-startup-reliability-post-audio-20260816/`.

The historical E5/E8/E9 `liblwjgl_stb` / Sound engine / `STBVorbis` SIGSEGV evidence remains valid. Attempt-014 remains historically unresolved and was not reproduced post-mitigation. A 20 / 20 finite sample does not prove absolute reliability.

## Authorization and hard gate

E0–E12 contract / adapter / offline runtime / live bridge 为 `unit_verified`，且各有 reviewed real success；均不是 `integration_verified`，`process_release_proven=false`，`p1_validation_manifest()` 仍将 E0–E12 标为 `not_run`。E11 runtime geometry deployed / real verified = **YES**；real E11 reviewed success = **YES** (`p1-e11-completion-barrier-20260817-004`, `portal_activation_ok`, `tested_action_count=1`, retry=0, portal=6/6, `truth_missing_count=0`)；`integration_verified` = **NO**。E12 real reviewed success = **YES** (`p1-e12-dimension-transition-20260817-001`, `dimension_transition_ok`, before=`minecraft:overworld`, after=`minecraft:the_nether`, `tested_action_count=1`, retry=0, `truth_missing_count=0`)；`integration_verified` = **NO**。一次 E12 成功不是 Hard Gate。E8/E9/E10/E11/E12 evaluator-only truth 与 Agent-visible observation 保持隔离。P1 Hard Gate 为 **NOT PASSED**，P2 为 **NOT STARTED**。

当前最小未验证点：P1 Hard Gate 剩余要求（稳定重复成功 / process release）。每次真实运行与每次 Gradle 构建仍需用户单独明确授权。

进入 P2 前要求完整 suite 稳定重复成功、`truth_missing=0`、无人工干预。P1 Hard Gate 尚未通过。建议至少 20 个 fresh episodes，最终次数、seed、timeout 与失败处理后续冻结。

## Full E0–E12 suite and process-release gate

The suite is orchestration only. It reuses `P1_VALIDATION_CASES`, the existing authorized E0–E12 runners, `E0CleanupStatus`, and the startup-reliability PID-tree inspection. It does not wrap each case in a new class or duplicate case logic.

Ordered steps: E0–E6, E7 water, E7 lava, E8, E9 water, E9 lava, E10–E12.

Aggregate verdicts, first blocking issue in order:

- `validation_failed`
- `truth_missing`
- `cleanup_failed`
- `process_release_not_proven`
- `hard_gate_success`

OS-level process release is separate from `env.close()`. `E0CleanupStatus.process_release_proven` remains false for object-level cleanup. `ProcessReleaseStatus.process_release_proven` is true only when a MineRL/Minecraft/JVM child was observed in the OS process table, the case subprocess exited, and those tracked PIDs are gone.

Encoded Hard Gate conditions for one complete suite run:

1. every required step present in that order;
2. every step succeeded;
3. `truth_missing_count == 0` where `requires_server_truth`;
4. no explicit cleanup failure;
5. OS `process_release_proven` for every step;
6. `real_execution_performed`;
7. no human intervention.

A passing Hard Gate still does not set `integration_verified` or change `p1_validation_manifest()` from `not_run`. Offline `--check` / `--preflight-only` cannot pass the Hard Gate.

Offline-safe check:

```bash
conda run -n mc-agent python -m obsidianlink.env.integration.p1_suite --check
```

Future authorized single pilot (do not run without a separate live MineRL authorization):

```bash
conda run -n mc-agent python -m obsidianlink.env.integration.p1_suite \
  --execution-mode authorized_live_p1_suite \
  --authorized-live-run p1_e0_e12_validation_suite \
  --output-dir runs/p1_validation_suite/<unique-pilot-id>
```

Use `--preflight-only` with those flags to validate authorization without launching MineRL.
