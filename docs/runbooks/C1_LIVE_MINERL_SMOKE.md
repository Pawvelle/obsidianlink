# Legacy v1 — C1 Live MineRL Smoke Validation Contract

> **v2 status:** legacy/calibration only. This runbook is preserved for regression
> and audit compatibility; it is not the active P1 validation suite and cannot
> support a v2 benchmark capability claim.

阶段：`R6-C1-PLAYER-RELATIVE-TRUTH-GRID-ANCHOR-OFFLINE-FIX`（离线完成）；八次 `R6-C1-LIVE-MINERL-SMOKE-AUTHORIZED-RUN` 均已结束且**未成功**

本文档冻结 **C1** 真实 MineRL 烟雾验证合同，并记录 offline runner wiring、三次授权 live 尝试，以及离线 aim/place 与 inventory settle 修复。合同冻结、offline wiring、失败的 live 尝试或离线修复 **不等于**真实 casting 能力已验证，也不得把 `live_run_allowed` 改为 `true`。

## 冻结身份

| 字段 | 冻结值 |
|---|---|
| family | `casting` |
| mode | `single` |
| level | `C1` |
| layout | `fixed` |
| compatibility task | `casting_c1_fixed` |
| designated agent | `agent_1` |
| target cell | `[2, 4, 3]` |

任务页：[casting_c1_fixed](../tasks/casting/casting_c1_fixed.md)。

## 最小目标

- 只验证一个目标 cell；
- 必须使用原版水、熔岩和 Minecraft block update 生成黑曜石；
- 不允许预置或直接放置 `obsidian`；
- evaluator 必须独立验证 target block、water/lava observation、transition step 和合法动作因果；
- driver 完成或文本声称成功不能构成成功；
- truth 缺失、坐标不一致、身份不一致或因果不足时必须 fail closed。

## 授权边界

- 每次真实 MineRL/Minecraft 运行都需用户单独批准；
- 每次 Gradle 构建都需用户单独批准；
- 修复后若要再次真实运行，必须重新批准；
- 不得跳过 C1 smoke 直接进入 C5 live 或 R7。

## Offline runner wiring（`R6-C1-LIVE-MINERL-SMOKE-RUNNER-WIRING`，已完成）

入口：

- 核心：`obsidianlink.runners.casting_c1_live_smoke.run_casting_c1_live_smoke`
- CLI：`scripts/run_c1_live_smoke.py --mode offline_stub --output-dir <绝对路径>`

执行模式闭集：仅 `offline_stub`。必须使用 `build_offline_stub_env_factory()` 返回的受控 `OfflineC1StubEnvFactory`；任意 callable 和外部 backend 注入均被拒绝。拒绝 live 请求；不得写入正式 `runs/`。

## Authorized live 入口（已实现；一次运行失败）

入口：

- 核心：`obsidianlink.runners.casting_c1_live.run_casting_c1_authorized_live`
- CLI：`scripts/run_c1_live.py --mode authorized_live_c1 --authorized-live-run casting_c1_fixed`

要求：

- 不接受 caller-supplied factory/backend；
- production factory 为仓库固定 `_default_env_factory`；
- `max_reset_attempts=1`；每进程最多一次真实 env/episode；
- 输出必须是尚不存在的 `runs/casting_c1_fixed/<run_id>/`；
- catalog `live_run_allowed` 保持 `false`（授权仅记入 `authorization.json`）；
- 禁止 Gradle / 模型 API。

### 2026-08-12 首次授权运行结果

目录：`runs/casting_c1_fixed/20260812-110318/`

| 项 | 值 |
|---|---|
| env factory calls | 1 |
| episodes | 1 |
| driver | completed (24；旧 plan) |
| evaluator | truth_missing / false |
| water/lava/obsidian | 未捕获 |
| evidence_complete | true |
| close_status | closed |
| Gradle | 未运行 |
| overall_success | **false** |

根因假设：冻结 C1 deterministic plan 无瞄准；真实世界中 `place_block`/`use_item` 未消耗库存、未改变目标 cell。

## Offline aim/place 修复（`R6-C1-LIVE-AIM-AND-PLACE-OFFLINE-FIX`，已完成）

离线已完成：

- C1 plan 加入有界相对 `look` delta，分别瞄准两个支撑面、熔岩目标面和相邻水源面；圆石独立 `equip` + wait，并严格确认具体目标物品减少 1；
- FakeBackend 覆盖 `not_aimed` / `too_far` / `no_valid_face` / `no_world_effect`，库存或网格未变则 fail closed；
- placement diagnostics 仅 evaluator-only，不进入 Observation。

这 **不等于** live 已验证。再次真实运行必须重新授权。真实水/熔岩/黑曜石 casting **尚未验证**。

## 后续授权运行与库存 settle 修复（`R6-C1-INVENTORY-SETTLE-CONFIRMATION-OFFLINE-FIX`，已完成）

- 第二次证据：`runs/casting_c1_fixed/20260812-153108/`。第一块圆石生效，第二块因瞄准面被遮挡而未放置；随后离线改为两格竖直支撑面。
- 第三次证据：`runs/casting_c1_fixed/20260812-154111/`。最终画面确认两块圆石均已放置，但 MineRL 在第二块 action 当拍仍返回旧库存，driver 因此 fail closed；水、熔岩和黑曜石步骤尚未执行。
- 离线修复只允许紧随 relevant action 的一个既有 settle wait tick 确认目标物品恰好减少 1；下一拍仍未满足时继续 fail closed，不接受无限等待或宽松库存变化。

下一次真实 C1 smoke 仍须用户单独授权。

### 第四次授权运行（2026-08-12）

- 证据：`runs/casting_c1_fixed/20260812-162216/`。
- final frame 显示两格高圆石柱，但 step 9 与唯一 settle step 10 的 observation 都报告圆石 7。
- driver 在 step 10 fail closed；水、熔岩和黑曜石动作均未执行，evaluator 为 `truth_missing`。
- 已离线实现严格有界的 4-tick inventory settle window：仅连续 no-op 可用于确认，只接受目标物品恰好减 1，超时或其他变化 fail closed。再次 live 仍需单独授权。

客户端 final frame 只能作为人工诊断材料，不能证明服务端已接受方块。当前尚未把支撑格 server-side block truth 接入 evaluator 证据。

### 第五次授权运行（2026-08-12）

- 证据：`runs/casting_c1_fixed/20260812-163842/`。
- 执行越过支撑 confirmation，桶动作后 MineRL 合法返回主手 `bucket`；旧 selected-item 白名单拒绝该值并 fail closed，evaluator 未运行。
- 离线修复允许 `bucket` 作为 observation 值，但不把它加入 translator 动作目标白名单。
- 再次 live 仍需单独授权。

### 第六次授权运行（2026-08-12）

- 证据：`runs/casting_c1_fixed/20260812-164731/`。
- JVM 在 reset mission 握手期间于 `liblwjgl_stb.dylib` / `STBVorbis` sound-engine 原生线程 `SIGSEGV`；episode 未开始，证据包 fail closed incomplete。
- 离线修复在临时目录复制同一个 vendored JAR/launcher，并用全静音 `options.txt` 避免音频解码；close 后恢复 MineRL runtime path 并删除临时目录。
- 不修改 `vendor/minerl`，不更换版本，不运行 Gradle；再次 live 仍需单独授权。

### 第七次授权运行（2026-08-12）

- 证据：`runs/casting_c1_fixed/20260812-165809/`。
- 临时静音 runtime 正常工作；driver 完成全部 36 步，两个桶动作均消耗成功，环境正常关闭。
- evaluator 仍为 `truth_missing`。客户端最终画面可见水与深色方块，但不能替代 server-side truth。
- 日志确认 MineRL 1.0.2 忽略绝对 `AgentStart Placement`，玩家真实出生坐标与 XML placement 相距很远；Java `atSpawn` grid 因此采样了错误区域。
- 离线修复对 Casting mission 省略绝对 placement，并把 C1 公开目标 `[2,4,3]` 映射到出生点相对 truth-grid `[2,0,3]`；Route A0 保持旧合同。
- 再次 live 仍需单独授权。

### 第八次授权运行（2026-08-12）

- 证据：`runs/casting_c1_fixed/20260812-171620/`；沙箱内预启动失败证据为 `20260812-171556/`，没有进入 episode。
- 沙箱外环境一次、episode 一次、正常关闭；driver 完成 36 步，两个桶和两块圆石均消耗成功。
- final frame 显示熔岩、流水和支撑块，但没有黑曜石；evaluator 为 `truth_missing`。
- MineRL bridge 在 `atSpawn=true` 且没有 placement 时使用共享世界出生点；玩家仍可能受默认 `spawnRadius` 偏移，因此上一轮锚定修复不完整。
- 离线修复只为 C1 输出 `atSpawn=false`，以实际、未移动的玩家位置锚定 grid；Route A0 与 C2–C5 保持原 XML 行为。
- 再次 live 仍需单独授权。

## 证据要求

结果写入 `runs/casting_c1_fixed/<run_id>/`，至少包含标准 10 文件，另加：

```text
runtime_preflight.json
authorization.json
process_lifecycle.jsonl
```
