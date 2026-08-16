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

E10 通过不表示正式 portal task 成功。当前 verification：contract/offline runtime/MineRL bridge = `unit_verified`；E10 geometry real verified = **YES**；Real E10 conversion calibration = **NOT RUN**；`integration_verified` = **NO**。

## Startup reliability hardening

- P1 startup reliability hardening: **COMPLETE**.
- Post-audio fresh-process validation: **20 / 20 observed success**.
- `max_reset_attempts`: **1**; each attempt used a fresh Python process.
- Startup infrastructure failures observed post-mitigation: **0**.
- Native crash recurrence: **0**; Malmo EOF recurrence: **0**.
- Evidence: `runs/history/p1-startup-reliability-post-audio-20260816/`.

The historical E5/E8/E9 `liblwjgl_stb` / Sound engine / `STBVorbis` SIGSEGV evidence remains valid. Attempt-014 remains historically unresolved and was not reproduced post-mitigation. A 20 / 20 finite sample does not prove absolute reliability.

## Authorization and hard gate

E0–E9 contract / adapter / offline runtime / live bridge 为 `unit_verified`，且各有 reviewed real success；均不是 `integration_verified`，`process_release_proven=false`，`p1_validation_manifest()` 仍将 E0–E12 标为 `not_run`。E8/E9/E10 evaluator-only truth 与 Agent-visible observation 保持隔离。E10 离线实现为 `unit_verified`，geometry real verified 为 **YES**，真实 conversion calibration 为 **NOT RUN**。E11、E12 均为 **NOT STARTED**，P1 Hard Gate 为 **NOT PASSED**，P2 为 **NOT STARTED**。

下一项经授权的 P1 validation target 是 **一次独立真实 E10 water-bucket → obsidian MineRL calibration**。这不授权或启动该运行；每次真实 MineRL/Minecraft 运行仍需用户单独明确授权。

进入 P2 前要求完整 suite 稳定重复成功、`truth_missing=0`、无人工干预。P1 Hard Gate 尚未通过。建议至少 20 个 fresh episodes，最终次数、seed、timeout 与失败处理后续冻结。
