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

One real E11 activation: `p1-e11-live-001`, outcome `portal_activation_not_observed`. Before: 14/14 obsidian, 6/6 interior air, portal=0, fire=0, ignition `(0, 4, 1)` = air. Stimulus: exactly one accepted `use_item(flint_and_steel)`. After: ignition cell = `fire`, 0/6 `nether_portal`, controls unchanged, `truth_missing_count=0`. Fire ≠ success. Compact evidence: `runs/history/p1-e11-live-20260816-001/`. Offline PortalSize replica of that snapshot: Axis.X **PASS** (width 2, height 3, bottomLeft `(1, 4, 1)`); Axis.Z invalid as expected. The 14-cell frame is not a vanilla Axis.X geometry failure. Callback / `canLightPortal` / `BlockTags.FIRE` remain unproven without JVM logs. Diagnosis: [P1 E11 live failure diagnosis](P1_E11_LIVE_FAILURE_DIAGNOSIS.md). Prepared logging-only patch is **not** applied or deployed. This is not portal activation capability and does not set `integration_verified`. Do not retry the uninstrumented run.

## Startup reliability hardening

- P1 startup reliability hardening: **COMPLETE**.
- Post-audio fresh-process validation: **20 / 20 observed success**.
- `max_reset_attempts`: **1**; each attempt used a fresh Python process.
- Startup infrastructure failures observed post-mitigation: **0**.
- Native crash recurrence: **0**; Malmo EOF recurrence: **0**.
- Evidence: `runs/history/p1-startup-reliability-post-audio-20260816/`.

The historical E5/E8/E9 `liblwjgl_stb` / Sound engine / `STBVorbis` SIGSEGV evidence remains valid. Attempt-014 remains historically unresolved and was not reproduced post-mitigation. A 20 / 20 finite sample does not prove absolute reliability.

## Authorization and hard gate

E0–E10 contract / adapter / offline runtime / live bridge 为 `unit_verified`，且各有 reviewed real success；均不是 `integration_verified`，`process_release_proven=false`，`p1_validation_manifest()` 仍将 E0–E12 标为 `not_run`。E11 contract / adapter / offline runtime / live gate 为 `unit_verified`；E11 runtime geometry deployed / real verified = **YES**；E11 real activation attempted = **YES** (`p1-e11-live-001`, `portal_activation_not_observed`)；offline PortalSize Axis.X on that snapshot = **PASS**；callback UNPROVEN；`integration_verified` = **NO**。E8/E9/E10/E11 evaluator-only truth 与 Agent-visible observation 保持隔离。E12 为 **NOT STARTED**，P1 Hard Gate 为 **NOT PASSED**，P2 为 **NOT STARTED**。

下一项经授权的 P1 工作是 **authorize E11 diagnostic runtime instrumentation**（logging-only `e11-portal-activation-diagnostic.patch`）。不要 retry 无 instrumentation 的 E11，不要 Gradle，不要开始 E12。每次真实运行仍需用户单独明确授权。

进入 P2 前要求完整 suite 稳定重复成功、`truth_missing=0`、无人工干预。P1 Hard Gate 尚未通过。建议至少 20 个 fresh episodes，最终次数、seed、timeout 与失败处理后续冻结。
