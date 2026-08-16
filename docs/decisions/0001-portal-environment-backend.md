# ADR 0001：地狱门环境后端采用最小 MineRL 桥接扩展

- 状态：已实现并通过桥接能力验证；完整地狱门进入验证待后续确定性 driver
- 日期：2026-07-30
- 适用阶段：Phase 1、Phase 2

## 背景

ObsidianLink A0 需要一个可重复的受控场景，并让评测器读取以下环境真值：

- 当前回合的方块变化；
- 门框区域的方块类型；
- 传送门方块是否生成；
- 角色当前维度；
- 固定出生点、朝向和初始资源是否生效。

Python 侧已经实现 `PortalA0EnvSpec`、`MineRLEnvironmentBackend` 和受限动作翻译，
但真实运行证明 MineRL 1.0.2 的当前 `EnvServer` 没有完整执行 Mission XML。

## 已验证事实

原始阻塞验证目录：

`runs/history/phase1-portal-env-smoke/20260730-201203-real/`

通过的能力：

- 固定 Python 3.10.20 与 OpenJDK 8 可以启动 MineRL；
- `reset/step/close` 正常完成，14/14 tick 未提前终止；
- 640x360 RGB POV 正常；
- 初始物品栏包含 10 个黑曜石和 1 个打火石；
- 看向、热键装备、使用和攻击命令通过受限 action space；
- 放置动作使黑曜石从 10 降到 9；
- `use_item.obsidian=1`、`use_item.flint_and_steel=1`。

未通过的能力：

- 请求出生点为 `(0, 64, 0)`，真实位置为
  `(-893.5, 63.0, -501.5)`；
- XML 中声明了 `FlatWorldGenerator`，真实世界仍是普通种子世界；
- XML 中声明了 `ObservationFromGrid`，Python 收到的 JSON 没有
  `portal_build_region`；
- JSON 中没有维度字段；
- 因此当前后端不能可靠判定门框几何、激活和进入下界。

## 根因

根因位于嵌套仓库 `vendor/minerl` 的 MineRL Java 桥接层，而不是
`PortalA0EnvSpec`：

1. `EnvServer.createNewWorld()` 固定调用默认
   `DimensionGeneratorSettings.fromDynamicRegistries(...)`，没有解释
   Mission XML 中的 `FlatWorldGenerator` 和 `DrawingDecorator`。
2. `EnvServer.setAgentPosition()` 只修改客户端玩家位置，服务端随后把玩家同步回自然
   出生点。
3. `EnvServer.getInfo()` 只转发 inventory、full stats 和 equipped items；
   `ObservationFromGrid` 不在转发路径中。
4. `getInfo()` 没有输出玩家当前 dimension。

这意味着继续在 Python handler 中调整 XML、坐标或方块名称不会解决问题。

## 决策

保留当前 Python 接口和 Benchmark 定义，下一步在 `vendor/minerl` 的独立分支中做
一个最小、通用、可测试的桥接扩展。不在上层通过画面分类或动作计数伪造环境真值。

已实现的最小 Java 改造范围：

1. 当 EnvSpec 请求 `FlatWorldGenerator` 时，在固定出生点生成 25x25 的确定性平整
   平台并清空上方空间；
2. 在 integrated server player 上应用位置和朝向；
3. 对 Mission XML 中声明的 `ObservationFromGrid/Grid` 逐项读取方块状态，并按
   grid name 写入 info JSON；
4. 在 info JSON 中加入稳定的 dimension 标识；
5. 应用 A0 依赖的时间、天气和生物生成初始条件；
6. 不改变现有 MineRL observation/action 协议的其他字段。

没有使用全局 superflat dimension generator。该旧 Forge/MCP fork 的原生 flat
generator 在两种构造方式下都会产生 `invalid biome id -1` 并卡在 spawn-area
准备阶段；局部确定性平台避免修改维度注册表，并覆盖 A0 所需活动区域。

`vendor/minerl` 继续作为独立 Git 仓库管理。可复现补丁保存在
[`patches/minerl/obsidianlink-envserver.patch`](../../patches/minerl/obsidianlink-envserver.patch)。
外层仓库不提交或重写嵌套仓库历史。

## 2026-07-30 实现证据

- `./gradlew compileJava`：通过；
- `./gradlew shadowJar`：通过；
- 运行时 jar SHA-256：
  `0bf3daaa884c3a31d94b2dc0f7f2fc9f0f5b73dc0019e94fb2dba37d7a8398b9`；
- 38 个 Python 单元测试通过；
- 真实 bridge smoke：
  `runs/history/phase1-portal-env-smoke/20260730-203612-real/`；
- 真实动作与 grid 增量：
  `runs/history/phase1-portal-env-smoke/20260730-203826-real/`。

最终动作运行完成 14/14 tick，固定位置为 `(0.5, 4.0, 0.5)`，343 个 grid cell
全部成功回传。放置后黑曜石从 10 减为 9，Evaluator 读取到
`obsidian_added=1`、一个 `obsidian` 和一个 `fire`，dimension 为
`minecraft:overworld`。

Apple Silicon 上仍存在 MineRL 历史性的间歇 `liblwjgl_stb.dylib` Sound engine
SIGSEGV；失败发生在 Mission 世界创建之前，限定重试可以成功，且每次失败后均无
残留 Minecraft 进程。该原生音频问题不属于本次 EnvServer 补丁范围。

后续 P1 startup hardening 通过 property-gated client-audio mitigation 绕开该
SoundEngine/STB 路径。2026-08-16 的独立 post-mitigation validation 在
`max_reset_attempts=1` 下观察到 20/20 fresh-process success，native crash 与
Malmo EOF recurrence 均为 0；这不删除上述历史证据，也不构成绝对可靠性证明。
详见 `docs/architecture/P1_STARTUP_RELIABILITY_ROOT_CAUSE.md` 与
`runs/history/p1-startup-reliability-post-audio-20260816/`。

## 最终端到端验证

确定性 Scripted-A0 已完成：

1. 生存模式下使用 14 块黑曜石构造完整 4x5 门框；
2. 打火石激活后，bridge 锁存到 `portal_activated=true`；
3. 角色在门内等待 84 tick 后，dimension 从 overworld 变为
   `minecraft:the_nether`；
4. 正常 close 后无残留 Minecraft 或 Gradle 进程。

通过运行位于
`runs/history/phase1-scripted-a0/20260730-214356/`。该运行共完成 251 step，
`max_obsidian_added=14`、`use_item.flint_and_steel=1`，且未提前终止。

最终 observation 已位于 Nether，因此当前 grid 不再包含主世界门框；
`portal_activated_latched=true` 保存了维度切换前的激活真值。Phase 1 据此完成，
下一阶段是独立实现 Portal Evaluator 的几何和负例规则。

## 回退方案

如果最小桥接扩展无法稳定通过上述验证，则停止继续扩展 MineRL 单机桥接，改为评估
独立 Minecraft 服务器后端。上层 `EnvironmentBackend`、动作协议、Evaluator 接口
和 Benchmark 文件保持不变。
