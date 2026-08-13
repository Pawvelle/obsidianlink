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

环境可预置合法 support/trench，但不能预置目标黑曜石。Deterministic calibration script 只执行最小 lava/water interaction。Evaluator 必须观察目标位置的 server-side transition（例如 `air/lava/... -> obsidian`），并把动作、world update 和 verdict 绑定到同一 episode/step 因果链。

```text
MineRL Action
  -> Minecraft Server
  -> Vanilla Mechanics
  -> Evaluator-only World Truth
  -> Verdict
```

E10 通过不表示正式 portal task 成功。

## Authorization and hard gate

E0 contract 已完成；E0 离线 runtime 为 `unit_verified`；E0 MineRL integration bridge 已实现。一次授权真实 E0 lifecycle run 已成功并审查，但 E0 不是 `integration_verified`：`process_release_proven` 仍为 false，仓库没有自动 verification promotion。E1 RGB contract / adapter / offline runtime / live bridge 已实现并离线 `unit_verified`；真实 E1 尚未执行。`p1_validation_manifest()` 仍将 E0–E12 标为 `not_run`。E2–E12 尚未实现。E0 入口为 `python -m obsidianlink.env.integration.e0_run`（`--execution-mode authorized_live_e0 --authorized-live-run e0_reset_close`）。E1 入口为 `python -m obsidianlink.env.integration.e1_run`（`--execution-mode authorized_live_e1 --authorized-live-run e1_rgb_observation`）。每次真实 MineRL/Minecraft 与每次 Gradle 构建需用户单独授权。

进入 P2 前要求完整 suite 稳定重复成功、`truth_missing=0`、无人工干预。P1 Hard Gate 尚未通过。建议至少 20 个 fresh episodes，最终次数、seed、timeout 与失败处理后续冻结。
