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

E0–E5 contract / adapter / offline runtime / live bridge 为 `unit_verified`。E0–E5 各有已审查的真实成功证据，但都不是 `integration_verified`：`process_release_proven` 仍为 false，live runtime / `--check` / `p1_validation_manifest()` 不得自动 promotion，P1 Hard Gate 要求完整 suite 稳定重复成功。E6 contract / adapter / offline runtime / live bridge 为 `unit_verified`，并有一次已审查的真实 `placement_ok`（`p1-e6-live-001`，world `(0, 4, 1)` / atSpawn grid `(0, 0, 1)`）；E6 不是 `integration_verified`。E7 contract / adapter / offline runtime / live bridge 为 `unit_verified`；WATER 有一次已审查的真实 `bucket_ok`（`p1-e7-water-live-001`）；LAVA 有一次已审查的真实 `bucket_ok`（`p1-e7-lava-live-001`，world `(0, 4, 1)` / atSpawn grid `(0, 0, 1)`）；E7 不是 `integration_verified`。E8 contract / adapter / offline runtime / live bridge 为 `unit_verified`。E8 建立 evaluator-only `ServerTruthSnapshot` 的 block portion 与 generalized target-region block truth。一次真实 attempt `p1-e8-live-001` / `runs/p1_e8_block_truth/e8-live-20260815-001` 在 reset 因 JVM SIGSEGV（`liblwjgl_stb.dylib` / Sound engine / `STBVorbis`）失败；已审查为与 E5 attempt #1 SAME_FINGERPRINT 的 infrastructure crash。第二次独立 attempt `p1-e8-live-002` / `runs/p1_e8_block_truth/e8-live-20260815-002` 为已审查的真实 `block_truth_ok`：before 三格 `air`，一次 `place_block(dirt)`，after target `dirt` 且 control 仍为 `air`，`truth_missing_count=0`，dimension `minecraft:overworld`。E8 reviewed real success: YES，仍不是 `integration_verified`。E9 contract / adapter / offline runtime / live bridge 为 `unit_verified`。E9 扩展同一 evaluator-only snapshot 的 fluid portion：`ServerFluidTruth` 保留 ObservationFromGrid 的 `water`/`flowing_water`/`lava`/`flowing_lava`，不把 source 与 flowing 折叠成 E7 的粗类。校准为封闭 WATER/LAVA variants，各一次 `use_item`，probe 为 world `(0, 4, 1)` / `(0, 5, 1)` / `(0, 5, 0)`。E9 真实 MineRL 尚未运行，没有 reviewed real success，不是 `integration_verified`。不要开始 E10。`p1_validation_manifest()` 仍将 E0–E12 标为 `not_run`。E10–E12 尚未实现。E0 入口为 `python -m obsidianlink.env.integration.e0_run`。E1 入口为 `python -m obsidianlink.env.integration.e1_run`。E6 入口为 `python -m obsidianlink.env.integration.e6_run`。E7 入口为 `python -m obsidianlink.env.integration.e7_run`。E8 入口为 `python -m obsidianlink.env.integration.e8_run`。E9 入口为 `python -m obsidianlink.env.integration.e9_run`。每次真实 MineRL/Minecraft 与每次 Gradle 构建需用户单独授权。一次 E9 live authorization 只对应一个 variant、一个 fresh episode、一次 `use_item` stimulus。

进入 P2 前要求完整 suite 稳定重复成功、`truth_missing=0`、无人工干预。P1 Hard Gate 尚未通过。建议至少 20 个 fresh episodes，最终次数、seed、timeout 与失败处理后续冻结。
