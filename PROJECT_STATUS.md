# 当前状态

更新时间：2026-08-12

## 当前唯一目标

任务：`R6-C1-LIVE-MINERL-SMOKE-RUNNER-WIRING`（已完成，offline）

本轮在不启动真实 MineRL/Minecraft 的前提下，完成 C1 smoke runner wiring：冻结 TaskInstance + C1 deterministic driver + production `MineRLEnvironmentBackend`(注入式 stub `env_factory`) + 独立 `CastingEvaluator` + episode finalization + 完整 evidence bundle。`live_run_allowed` 保持 `false`；不得把 offline stub success 写成 live 验证。

`R6-C5-LIVE-MINERL-BACKEND-WIRING` 与 `R6-C1-LIVE-MINERL-SMOKE-VALIDATION-CONTRACT-FREEZE` 均已完成。下一步必须是用户单独授权的一次 **C1** 真实 MineRL smoke run；不得直接跳到 C5 或 R7。若真实运行需要 Gradle，须另行单独批准。

当前 active implementation 仍是 Casting-S-C2 / fixed 的兼容 ID `casting_c3_fixed`。旧 ID 中的 `c3` 表示三个 cell，不表示 B0 taxonomy 的 C3；文档级兼容名称为 `casting_s_c2_fixed`。`casting_c1_fixed` 保留为 Casting-S-C1 回归合同，并作为 C1 live smoke 的兼容任务身份。

R6 的 Casting-S-C3 / C4 / C5 任务合同**已经冻结**（catalog 可见、taxonomy 正确、scenario_parameters 显式、`live_run_allowed=false`）。R6-C3 / C4 / C5 evaluator + deterministic driver 已在 FakeBackend 上完成离线证明；MineRL backend typed truth wiring 已在 stub raw observations 上完成离线证明。`casting_s_c5_fixed` 仍必须保持 `implementation_status="contract_only"`、`live_run_allowed=false`；C5 不冒充正式 live implementation。

## R6-C1-LIVE-MINERL-SMOKE-RUNNER-WIRING 完成（offline）

1. **入口**：[`obsidianlink/runners/casting_c1_live_smoke.py`](obsidianlink/runners/casting_c1_live_smoke.py) 与 [`scripts/run_c1_live_smoke.py`](scripts/run_c1_live_smoke.py)。
2. **执行模式**：闭集仅 `offline_stub`；只接受受控类型 `OfflineC1StubEnvFactory`，任意 callable 与外部 `backend=` 注入通道均被拒绝；`request_live` / `allow_live_run_override` / 未选 offline 模式一律 fail closed；CLI 无可用 live 命令。
3. **冻结身份**：family=`casting`、mode=`single`、level=`C1`、layout=`fixed`、compatibility task=`casting_c1_fixed`、canonical=`casting_s_c1_fixed`、agent=`agent_1`、target=`[2,4,3]`。
4. **Preflight**：在 env factory 调用前校验完整 TaskInstance 与 canonical `casting_c1_fixed` 精确等值（含 step/time/model-call budgets、精确库存和禁止额外 obsidian）、capability、plan 等值、catalog `live_run_allowed=false`；输出目录必须为绝对、尚不存在且位于正式 `runs/` 外。
5. **编排**：driver → `mark_terminated` → backend typed `get_casting_evaluation_state` → 独立 `CastingEvaluator`；driver completed 不映射为 success。
6. **Evidence**：在目标同一父目录 staging，完整 10 文件 bundle 与最终 close 状态写定后用原子 rename 发布；拒绝覆盖已有目录；public events / summary 不含 evaluator-only token；PNG 合法可读；失败只写 fail-closed summary。
7. **生命周期**：成功/失败/异常路径均有限次 close；close 失败被结构化记录。
8. **未验证**：真实 MineRL/Minecraft 水/熔岩/黑曜石变化、task-origin/grid 锚定、portal transition 仍未验证。

操作说明见 [`docs/runbooks/C1_LIVE_MINERL_SMOKE.md`](docs/runbooks/C1_LIVE_MINERL_SMOKE.md)。

## R6-C1-LIVE-MINERL-SMOKE-VALIDATION-CONTRACT-FREEZE（合同冻结）

### 冻结身份

后续 C1 live smoke validation 的身份冻结为：

| 字段 | 冻结值 |
|---|---|
| family | `casting` |
| mode | `single` |
| level | `C1` |
| layout | `fixed` |
| compatibility task | `casting_c1_fixed` |
| designated agent | `agent_1` |

### 最小目标（后续真实烟雾测试）

- 只验证一个目标 cell（`casting_c1_fixed` 冻结目标 `[2, 4, 3]`）；
- 必须使用原版水、熔岩和 Minecraft block update 生成黑曜石；
- 不允许预置或直接放置 `obsidian`；
- evaluator 必须独立验证 target block、water/lava observation、transition step 和合法动作因果；
- driver 完成或文本声称成功不能构成成功；
- truth 缺失、坐标不一致、身份不一致或因果不足时必须 fail closed。

### 授权边界

- 每次真实 MineRL/Minecraft 运行都需用户单独批准；
- 每次 Gradle 构建都需用户单独批准；
- 修复后若要再次真实运行，必须重新批准；
- 本轮合同冻结不能将任何任务的 `live_run_allowed` 改为 `true`；
- 不得把合同冻结写成真实 MineRL / 水 / 熔岩 / 黑曜石 / portal transition 能力已经验证。

### 证据要求（后续真实运行）

结果必须写入 `runs/`，至少包含：

```text
task_instance.json
experiment_config.json
capability_manifest.json
code_version.json
initial.png
final.png
events.jsonl
evaluator_events.jsonl
summary.json
manual_review.md
```

observation、action、evaluation 和 log 必须带 `episode_id`、`step_id`，适用时带 `agent_id`。Agent-visible 与 evaluator-only 数据必须分开；evaluator truth 不得进入 Observation、driver event、prompt 或 memory。

### 本轮交付边界

- 已完成：状态收敛、C1 smoke 合同冻结、离线 phase / catalog 不变量锁定；
- 后续 runner wiring 已在 `R6-C1-LIVE-MINERL-SMOKE-RUNNER-WIRING` 完成（offline stub）；
- 未执行：真实 MineRL/Minecraft、Gradle、付费模型 API、提交或推送；
- 仍未验证：真实环境中的水/熔岩/黑曜石变化、task-origin/grid 锚定、portal transition（尚未验证）。

操作说明见 [`docs/runbooks/C1_LIVE_MINERL_SMOKE.md`](docs/runbooks/C1_LIVE_MINERL_SMOKE.md)。

## R6-C5-LIVE-MINERL-BACKEND-WIRING 完成（offline）

R6-C5-LIVE-MINERL-BACKEND-WIRING 已通过严格离线测试完成 typed target-block / fluid truth 的 production MineRL backend wiring。`casting_c1_capabilities()` 现在将 `exposes_target_block_truth` 和 `exposes_fluid_truth` 报告为 `True`（offline-only），背后的 typed truth surface 由下列保障支撑：

1. **Schema-legal unified grid bridge** [`obsidianlink/env/portal_spec.py`](obsidianlink/env/portal_spec.py)：
   - 只使用 Malmo schema 支持的 `<ObservationFromGrid>`；`PORTAL_GRID_BLOCKS` 闭集同时包含 `water` / `flowing_water` / `lava` / `flowing_lava` 与门框方块，不声明不存在的 `<ObservationFromFluidGrid>`；
   - evaluator grid 范围为 `(-3,-1,0)–(4,5,6)`，覆盖 C2 的 x=4 目标 cell；offline 测试通过 stub `env_factory` 的 `portal_grid` 提供严格的方块/流体 truth，**不**启动 MineRL 客户端。

2. **Production backend typed truth surface** [`obsidianlink/env/minerl_backend.py`](obsidianlink/env/minerl_backend.py)：
   - `get_casting_evaluation_state(target_cell)` → C1 typed `CastingEvaluationState`，`get_continuous_casting_evaluation_state(target_cells)` → C2 `ContinuousCastingEvaluationState`，`get_frame_evaluation_state()` → C3 `FrozenFrameEvaluationState`，`get_ignition_evaluation_state()` → C4 `FrozenIgnitionEvaluationState`，`get_nether_entry_evaluation_state()` → C5 `FrozenNetherEntryEvaluationState`。这些是 production MineRL backend **唯一**对外的 evaluator-only 入口；
   - 每个 getter 严格从 `raw["portal_grid"]` / `raw["portal_transition"]` 读世界 truth，从 backend latched state 读经世界变化确认的 action step / identity，**永不**根据 driver 意图、action 参数或 Agent prompt 伪造 world truth；
   - cast credits 只在 macro translator 成功接受后追加；`action.target == "water_bucket" / "lava_bucket" / "flint_and_steel"` 才记入 credit history，翻译失败的 action **不**记入 credits（fail closed）。

3. **Causal action credit tracking**（`cast_credit_history` + `first_water_step_by_offset` / `first_lava_step_by_offset`）：
   - 每个 backend step 严格最多记一次 cast credit（即使 `duration_ticks=4` 也只记一次，因为 macro 是一次翻译）；`MAX_TRANSLATOR_DURATION_TICKS=40` 硬上限确保 credit 不会爆炸；
   - 只有当前 step 的合法 bucket action 与唯一 tracked cell 的预期流体变化同时出现，才为该 cell latch water/lava step；每个 cell 的水、熔岩、黑曜石转移独立归因，不借用全局最近 action；
   - pre-existing water / lava（baseline 已是水/熔岩）**不**算 causal credit；agent 不能在 episode 启动前免费拥有流体 truth。

4. **Capability honesty**（`casting_c1_capabilities()`）：
   - `exposes_target_block_truth=True` 和 `exposes_fluid_truth=True` 是 *static* 声明，**仅**当 typed truth surface 与对应测试通过之后才置 `True`；任何 `capabilities()` 子类覆写（如 `CapabilityManifestFailClosedTests`）会被 pre-episode gate 拦截；
   - 失败的 production manifest 仍由同一 gate fail closed 测试覆盖（`test_minerl_casting_reset_fails_before_env_creation` 使用 downgraded per-instance manifest）。

5. **Reset / step / close 清理**（`_fresh_latched_state()`）：
   - 每条 cast credit 都关联 `step_id`；`reset` 与 `close` 调用 `_fresh_latched_state()` 重新初始化 `cast_credit_history` / `first_obsidian_step_by_offset` / `first_water_step_by_offset` / `first_lava_step_by_offset` / `first_ignition_step` / `first_nether_portal_step`；
   - `step` 在每条 macro action 末尾调用 `_refresh_evaluation_milestones`，但 stale credit 不会跨 step 生存（credit 是 step 内的因果凭证，不进入下一 step 的 attribution）；新的 truth 永远从当前 raw observation + 当前 step 推算。

6. **Information isolation**（`Observation` schema 不变）：
   - `Observation` 仍是 9 字段（`episode_id` / `agent_id` / `step_id` / `timestamp` / `frame` / `visible_inventory` / `selected_item` / `messages` / `workflow_stage`）；`target_block_truth` / `fluid_truth` / `portal_grid` / `latched_frame_identity` / `matched_frame_identity` / `agents_in_nether` / `entered_via_episode_portal` / `pre_transition_position` / `nether_entry_evaluation` 全部缺席于 `Observation` / `BackendStep.info` / driver event；
   - `tests/test_r6_c5_live_minerl_backend_wiring.py` 的 `ObservationIsolationTests` 与 `DriverEventHygieneTests` 锁住 AST 锁，禁止任何 driver / evaluator 副作用进入 agent-visible 通道。

7. **新增专项离线测试** [`tests/test_r6_c5_live_minerl_backend_wiring.py`](tests/test_r6_c5_live_minerl_backend_wiring.py)（85 个离线用例）：
   - 翻译器 allowlist / 正路径 / fail-closed / bounded forward `move` / `duration_ticks > 40`；
   - backend selected item 直接从 bridge `equipped_items.mainhand.type` 读，**不**根据 `equip_item` 意图伪造；未知 / 缺失 / 非 `PORTAL_SELECTABLE_ITEMS` 的 value 全部 fail closed；
   - `get_*_evaluation_state` 全部在 stub `env_factory` 上跑：每条 typed state 的 `episode_id` / `step_id` / `agent_id` / `causality_window_steps` / `terminated_step` 都与 task 严格一致；
   - C1 / C2 / C3 / C4 / C5 都有 production-backend evaluator success 路径；同时覆盖 pre-existing truth、错误流体类型、被拒绝 action、无世界变化 action 不得归因、以及 evaluator-only 信息隔离。

8. **全量测试**：`1175` 个离线测试全部通过（`Ran 1175 tests in 170.485s → OK`）；`python -m obsidianlink --check` 输出 `phase: "r6_c5_live_minerl_backend_wiring_done"`；`python scripts/check_environment.py` 通过；`git diff --check` 干净。

9. **未验证限制**：
   - 真实 MineRL / Minecraft 中的 typed target-block / fluid / nether-transition truth 仍未验证；当前仅确认任务 XML 不再声明非法的 `ObservationFromFluidGrid`；
   - 真实服务端能否让 14 个目标 cell 在 800 step 内以可重复的顺序全部 `air → water+lava → obsidian` 仍未在真实 Minecraft 验证；
   - 真实服务端 `portal_transition` 字段在跨维度切档时是否同步返回，**未**验证；
   - 真实服务端在 A0 MineRL `Hotbar.4-6` 上能否稳定执行 `use_item(water_bucket)` / `use_item(lava_bucket)` / `place_block(cobblestone)` 仍**未**验证；
   - C2 网格已扩展到 x=4，C1–C5 production-backend evaluator success 已在 stub raw trajectory 中验证；这不等于真实 live 成功。C5 仍依赖尚未验证的 evaluator-only `portal_transition` bridge 字段。

10. **未修改**：
   - vendor/minerl：未触碰；
   - MineRL / Minecraft / Python / JDK / Gym / NumPy / Qwen / 模型版本：未升级或回退；
   - 没有启动真实 MineRL / Minecraft / Gradle / 付费模型 API / 提交 / 推送 / 创建 PR；
   - C5 仍保持 `implementation_status="contract_only"`、`live_run_allowed=false`；后续要进入 live run 仍需用户单独授权。
   - **状态收敛说明**：本节是最终完成记录。下方「部分完成（离线修正）」是 wiring 中途审查记录，其中 capability=`False` / “当前任务仍未完成”等描述已被本节取代，不得再当作当前状态。

## R6-C5-LIVE-MINERL-BACKEND-WIRING 部分完成（离线修正）— 历史中途记录（已被上方完成节取代）

> **过期标记（2026-08-12）**：本节保留 wiring 审查过程，但下列结论已过期，以「R6-C5-LIVE-MINERL-BACKEND-WIRING 完成（offline）」为准：
> - `exposes_target_block_truth` / `exposes_fluid_truth` 现已在 offline production manifest 中为 `True`（仅离线声明，不代表 live 已验证）；
> - typed C1–C5 truth surface 已接通并通过 stub raw observations 离线测试；
> - “当前任务仍未完成 / 只剩 typed truth 子项 / capability gate 保持关闭”不再成立；
> - 当前唯一目标已转为 `R6-C1-LIVE-MINERL-SMOKE-RUNNER-WIRING`（现已完成 offline）。

本轮修正了早期 wiring 审查发现的虚报和 fail-open 问题：

1. **MineRL 动作执行修正** [`obsidianlink/actions/minerl_translator.py`](obsidianlink/actions/minerl_translator.py) / [`obsidianlink/env/minerl_backend.py`](obsidianlink/env/minerl_backend.py)：
   - hotbar 映射改为根据冻结初始库存顺序构造；C5 水桶 / 熔岩桶 / 圆石 / 打火石对应 slot 1–4；
   - 闭集 `TRANSLATOR_EQUIPPABLE_ITEMS` 与 `TRANSLATOR_PLACEABLE_ITEMS` 明确锁定每种动作的目标允许集；
   - `equip_item` / `use_item` / `place_block` / `move` / `wait` 的每条分支都经过严格类型检查、有限数值限制和闭包空间验证；
   - translation 被拒绝时 backend 在 `env.step` 之前抛出受控错误，不再执行 no-op 并假装完成；
   - `duration_ticks` 在 backend 内以有限循环真实提交低层动作，宏级 `step_id` 仍严格 +1。

2. **Observation 扩展 selected_item 字段** [`obsidianlink/core/types.py`](obsidianlink/core/types.py)：
   - 公开 `Observation` dataclass 新增 `selected_item: str | None = None` 字段；`__post_init__` 严格验证非空字符串；
   - 字段顺序：episode_id / agent_id / step_id / timestamp / frame / visible_inventory / selected_item / messages / workflow_stage（共 9 个公开字段）；
   - C3 / C4 / C5 driver / evaluator 离线测试已同步更新到 9 字段 schema。

3. **MineRL selected-item 观察** [`obsidianlink/env/portal_spec.py`](obsidianlink/env/portal_spec.py)：
   - 删除没有 XML / bridge 数据源的伪 `PortalSelectedItemObservation`；
   - 改用 `HumanSurvival` 自带的 `EquippedItemObservation`，严格解析 `equipped_items.mainhand.type`；字段缺失或类型错误时 fail closed。

4. **capability 诚实性与 workflow 闭集** [`obsidianlink/env/minerl_backend.py`](obsidianlink/env/minerl_backend.py)：
   - `_selected_item_from_raw` 从 raw observation 的 `equipped_items.mainhand.type` 读取 bridge 值（空槽返回 `None`），永不根据 driver 请求的动作伪造成功；
   - `reset` / `step` 走同一份 `_public_observations` 路径，Observation 上的 `selected_item` 字段是 Agent-visible 公开数据；
   - `reset` 现在接受 C3 / C4 / C5 casting workflow（`route="lava_casting"`），保留 legacy Route A0 difficulty 1 校验；
   - `get_evaluation_state()` 仍只返回 legacy portal `EvaluationState`，不冒充 typed target-block / fluid / C5 truth；
   - `exposes_target_block_truth=False` / `exposes_fluid_truth=False`，真实 casting reset 在 env factory 前 fail closed；
   - 只接受 `route_a_a0` 和已冻结 C1–C5 workflow，未知 `lava_casting` workflow 不再绕过 gate。

5. **未完成边界**：
   - R6-C5 wiring 之前，MineRL backend 的 `casting_c1_capabilities()` 报告 7 个能力为 `False`（`select_water_bucket` / `select_lava_bucket` / `use_water_bucket` / `use_lava_bucket` / `selected_item` / `target_block_truth` / `fluid_truth`），导致 reset gate 提前 fail closed；
   - typed target-block / water / lava / fluid 因果证据尚未从 MineRL bridge 生成 C1–C5 evaluator state；
   - 任何后端只要 per-instance 覆盖 `capabilities()` 让其报告不完整，gate 依然 fail closed **before** the env factory is called and **before** any state mutation;
   - 没新增 `observation_only` 等其他分类；保留原始 `CAPABILITY_IDS` 8 项不动。

6. **专项离线测试** [`tests/test_r6_c5_live_minerl_backend_wiring.py`](tests/test_r6_c5_live_minerl_backend_wiring.py)，覆盖：
   - translator allowlist 与正路径（`equip_item`/`use_item`/`place_block`/`move`/`wait`/legacy A0）；
   - translator 失败闭合：未知 item / 空 / None target / 未知 action / 越界数值 / 非有限数 / bool-as-int / duration_ticks > 40；
   - selected_item 表面：bridge 提供值 / `empty` / 未知值 / 缺失键均严格处理，不根据 driver intent 伪造；
   - production capability manifest 保持 target-block / fluid 为 False，gate 在 env factory 之前 fail closed；
   - 完整 C5 evaluator outcome 闭集：success / external entry / missing entry / wrong source dim / wrong target dim / transition before activation / frame identity mismatch；
   - 明确使用测试专用 full-capability stub 验证 347-step driver 动作边界，不冒充 production truth wiring；
   - AST / 源码双门锁：C5 driver 源文件不引用 `selected_item` / `target_block_truth` / `fluid_truth` / `latched_frame_identity` / `matched_frame_identity` / `pre_transition_position` / `entered_via_episode_portal` / `agents_in_nether` / `latched_activation_offsets` / `nether_entry_evaluation` 任一属性；
   - driver 事件不携带任何 evaluator-only token（10 个 token 全部缺席）；
   - Observation 8→9 字段 schema 锁住，`target_block_truth` / `fluid_truth` / `portal_grid` / `latched_frame_identity` / `matched_frame_identity` / `agents_in_nether` / `entered_via_episode_portal` / `pre_transition_position` / `nether_entry_evaluation` 全部缺席于 Observation / driver event / public context；
   - reset / step / close 清空 backend latched state；
   - 动态 hotbar slot 1–4、`duration_ticks` 有限重复、translation reject 不调用 env.step、未知 workflow 提前拒绝。

7. **evaluator-only 信息隔离验证**：
   - production manifest 明确报告 typed truth 缺口，不再把 legacy portal grid 当作 C1–C5 typed truth；
   - 公共 `Observation` 仍只有 9 个字段，evaluator-only 字段全部缺席；
   - C5 driver 源文件 AST + `ast.Attribute` + `ast.Subscript` 扫描确认不引用任何 evaluator-only token；
   - legacy portal truth 仍与 public Observation 隔离；typed casting truth 未接通前不启动 episode。

8. **driver 与 evaluator 隔离**：
   - driver 不知道 selected_item、target block truth、fluid truth、frame identity、transition evidence 的存在；
   - evaluator 不依赖 driver 的 completed / failed 状态认定 success；
   - 任何 evaluator-only 数据不能进入 driver event / driver log / driver `as_dict()` snapshot。

9. **未验证限制**：
   - 真实 MineRL / Minecraft 浇筑、门框建造、点火、Nether entry 仍未验证；
   - 真实 MineRL 中 task-origin marker 与 evaluator truth-grid origin 的世界坐标锚定仍**未**在真实 Minecraft 中验证（`(-3,-1,0)–(3,5,6)` grid 数值范围已经覆盖固定 4×5 full-ring 方案）；
   - 真实 MineRL 桥接的 `selected_item` observable 需要在 MineRL 0.x 端 `HumanSurvival` 的 `Inventory.name_in_slot` 上验证；
   - 当前所有 wiring 仍只通过 stub `env_factory` + 注入式 raw observations 在离线环境下证明；
   - C5 仍保持 `implementation_status="contract_only"`、`live_run_allowed=false`，没有启动真实 MineRL、Gradle 或模型 API；
   - Ruined / Adaptive / Multi-Agent / R7 模型阶段仍未实现。

10. **（过期）当时未完成项**：本节撰写时 typed target-block / fluid truth 尚未接通。该项已在上方「R6-C5-LIVE-MINERL-BACKEND-WIRING 完成（offline）」中关闭；勿再据此判断当前任务。


## R6-C5-DETERMINISTIC-DRIVER 已完成（FakeBackend 离线证明）

1. **新增 C5 公开 Nether-entry 上下文边界** [`obsidianlink/core/casting_s_c5_nether_entry_context.py`](obsidianlink/core/casting_s_c5_nether_entry_context.py)：
   - `build_public_c5_nether_entry_driver_context_from_task(task)` 是整个 R6 C5 driver 家族中**唯一**允许读取 task `scenario_parameters` 的函数；它只读取 `public_task_spec.frame_plan.fixed_offsets`、 `public_task_spec.ignition_plan`（action=`use_item`、item=`flint_and_steel`、target_offset=`[1, 1, 1]`、target_policy=`exact`）和 `public_task_spec.nether_entry_goal`（designated_agent_ids=`["agent_1"]`、source_dimension=`minecraft:overworld`、target_dimension=`minecraft:the_nether`），以及 task limits / initial inventories，**忽略** `evaluator_contract` 与任何 evaluator-only 字段；
   - 返回严格冻结的 [`PublicC5NetherEntryDriverContext`](obsidianlink/drivers/casting_s_c5_nether_entry.py)：workflow / family / mode / level / layout / agent_id / 14 个有序 target offsets / 公开 ignition plan / 公开 nether_entry goal / 不可变 initial_inventory（MappingProxyType）/ task_step_limit / task_time_limit；任何未知 family / mode / level / layout、越界或重排 target offsets、bool 充当 int、缺失或重复 cell、ignition 字段与公开值不一致、designated agent / 源 / 目标维度不匹配、库存缺失或 bool 都 fail closed，context builder 不再用 `bool(...)` 掩盖原始类型错误。

2. **新增 C5 deterministic driver** [`obsidianlink/drivers/casting_s_c5_nether_entry.py`](obsidianlink/drivers/casting_s_c5_nether_entry.py)：
   - 显式接受 immutable `PublicC5NetherEntryDriverContext` 作为唯一 TaskInstance-shaped 输入；`run_casting_s_c5_nether_entry_driver` 不读取 scenario_parameters / evaluator_contract / `FrozenFrameIdentity` / `IgnitionActionEvidence` / `PortalActivationEvidence` / `FrozenIgnitionEvaluationState` / `FrozenNetherEntryEvaluationState` / `NetherEntryEvidence` / `FrozenNetherEntryEvaluator` 或任何 evaluator-only 字段；
   - 默认 plan：14 cell × 24 step = 336 step 的 C3 浇筑子计划 + 4 step 的 C4 ignition 子计划 + 7 step 的 C5 Nether-entry 子计划（4 approach moves + 1 alignment move + 1 portal-traversal move + 1 settle wait）= **347 step** default plan，落在 800 step 任务预算内；每 cell 2 个 `use_item(water|lava)` 视为 evaluator 的 relevant action，2 个 `place_block(cobblestone)` 支撑方块是 plan 内的机械动作但不进入 per-cell evidence；1 个 `use_item(flint_and_steel)` 是 ignition relevant action；portal-traversal `move` 是 nether-entry relevant action；
   - 运行入口对 caller-supplied plan 做整体等值校验：只能接受由公开 context 与恢复设置生成的完整确定性计划；仅 ignition、仅 entry、截断、重排或重复 entry traversal 的计划在 reset 前 fail closed，不能以 `completed` 绕过 14-cell 浇筑；
   - 动作白名单严格闭合：`equip_item` / `use_item` / `place_block` / `move` / `wait`；物品白名单 `water_bucket` / `lava_bucket` / `cobblestone` / `flint_and_steel`；`move` 参数固定为有限前进、无横移/冲刺/跳跃；永不放 `obsidian`、driver 不直接修改 dimension 或 portal truth；duration_ticks 1..40；
   - 预算硬上限：environment step 800、game time 720 秒、plan wait 320、plan length 700、per-action recovery 2、total recovery 32；
   - 恢复只响应 typed `RecoverableBackendError`，受 per-action 与 total 双重预算；其他异常 fail closed；
   - driver status 闭集 `completed` / `blocked` / `failed`；永不返回 `success` / `passed`（这些 verdict 仍由 `FrozenNetherEntryEvaluator` 独立判定）；
   - 结构化事件必须带 `episode_id` / `step_id` / `agent_id` / `cell_index` / `target_offset` / `label` / `phase` / `action_type` / `target` / `relevant_action` / `role` / `attempt`；结果对象 `CastingC5NetherEntryDriverResult` 不可变、类型严格、可序列化、暴露 `as_dict()`；
   - 新增 `nether_entry_step` / `nether_entry_target_offset` / `nether_entry_approach_step` 让 orchestrator 完全不读 evaluator truth 即可识别 C5 entry 步；`per_cell_relevant_action_records` 复用 C3 表面给 `FrozenFrameEvaluator`；`ignition_*` 字段复用 C4 表面给 `FrozenIgnitionEvaluator`；
   - 4 个新 PHASE 常量与 4 个新 ROLE 常量（`entry_approach` / `entry_align` / `entry_teleport` / `entry_settle`）严格分离 C3 浇筑、C4 ignition 与 C5 Nether-entry 子计划；
   - 模块 AST 检查确认：未 import `casting_nether_entry_evaluator` / `casting_ignition_evaluator` / `casting_frame_evaluator` / `agents` / `workflows` / `model` / `planner` 等；未通过 `ast.Attribute` / `ast.Subscript` 访问 `scenario_parameters` / `evaluator_contract` / `_nether_entry_evaluation_state` / `set_nether_entry_evaluation_state` / `get_nether_entry_evaluation_state` / `clear_nether_entry_evaluation_state` / `FrozenFrameIdentity` / `IgnitionActionEvidence` / `PortalActivationEvidence` / `FrozenIgnitionEvaluationState` / `FrozenNetherEntryEvaluationState` / `NetherEntryEvidence` / `FrozenNetherEntryEvaluator` / `agents_in_nether` / `entered_via_episode_portal` / `matched_frame_identity` / `latched_frame_identity` / `pre_transition_position` 等。

3. **不重命名旧 driver**：旧 `obsidianlink/drivers/casting_c3.py` 仍是 R5 / Casting-S-C2 的三 cell 连续浇筑 driver；`obsidianlink/drivers/casting_s_c3_frame.py` 仍是 R6-C3 14-cell driver；`obsidianlink/drivers/casting_s_c4_ignition.py` 仍是 R6-C4 14-cell + 4-step ignition driver；新模块明确命名为 `casting_s_c5_nether_entry.py`，反映 B0 taxonomy 的 C5（Nether entry）合同。四个 driver 互不共享类型。

4. **capability gate** [`obsidianlink/env/capabilities.py`](obsidianlink/env/capabilities.py) 的 `_GATED_WORKFLOWS` 已经包含 `casting_s_c5_fixed`（在 R6-C5-NETHER-ENTRY-EVALUATOR 阶段已加入）；C5 driver 使用同一份 gate，缺桶/选物品/selected_item/target_block_truth/fluid_truth 任一能力时 fail closed。

5. **新增 142 个专项离线测试** [`tests/test_r6_casting_c5_nether_entry_driver.py`](tests/test_r6_casting_c5_nether_entry_driver.py)，覆盖：
   - 公开上下文严格解析与不可变性（含 family / mode / level / layout / agent / 14 cell / grid 边界 / 重复 / 重排 / bool 充当 int / 库存缺失 / 库存 bool / 库存未知 item / step 预算下限 / ignition 字段与公开值不一致 / designated agent / source / target dimension）；
   - `build_casting_s_c5_nether_entry_action_plan` 固定 14-cell × 24 + 4 + 7 = 347 步、241 wait、28 cast relevant + 1 ignition relevant + 1 entry relevant、闭集 action / 参数 / 物品 / 阶段 / role；
   - Plan step validation（cast role 需要 cell_index、ignition / entry role 需要 cell_index=None、target/parameters validation、entry traversal step 验证等）；
   - 公开 ignition plan 验证（wrong action / wrong item / wrong target / wrong target_policy / ignition_required=False / required 缺失）；
   - 公开 nether_entry goal 验证（designated agent / source / target dimension drift / required=False / required 缺失）；
   - Driver result contract（frozen、status 闭集、nether_entry 字段必填、as_dict() detached、JSON 序列化、events 闭集）；
   - Driver 完整执行（status=completed、347 step、per_cell 2 records、nether_entry traversal 346 step、首个 approach move 341 step、ignition records 339 step、ignition equip 337 step、events 不携带任何 evaluator token）；
   - 800 step / 720 秒 / 320 wait / 700 plan / 32 total recovery / 2 per-step recovery 预算失败的 fail-closed 行为；
   - 重复 `RecoverableBackendError` 有限重试、per-step 与 total budget 各自独立耗尽、metadata 透传；
   - 非 `RecoverableBackendError` 异常（RuntimeError / OSError / TypeError）立即 fail closed；
   - backend 提前 terminated / truncated 的 blocked 行为；
   - 每个事件的 episode / step / agent / cell / target_offset / relevant_action / role 身份；
   - 相同输入的 action 序列、events 与 `as_dict()` 快照完全一致（确定性 replay）；
   - 闭集 status（`completed` / `blocked` / `failed`），驱动从不返回 `success` / `passed`；
   - capability gate 缺失能力时 reset 前 fail closed；
   - driver 拒绝错误 backend（无 reset/step）、错误 plan 类型、仅 entry 计划、仅 ignition 计划、仅 cast 计划、plan 超长、max_environment_steps 越界、max_game_time_seconds 越界、max_wait_steps 越界、total_recovery_budget 越界、重复 entry traversal 等；
   - AST + 源码双门锁：driver 源文件**不** import C5 / C4 / C3 / 任何 evaluator 表面、**不**调用 `set_nether_entry_evaluation_state` / `get_nether_entry_evaluation_state` / `clear_nether_entry_evaluation_state` / `_nether_entry_evaluation_state`、**不**以代码形式访问 `scenario_parameters` / `evaluator_contract` / `agents_in_nether` / `entered_via_episode_portal` / `matched_frame_identity` / `latched_frame_identity` / `pre_transition_position` 等；
   - Observation 8 字段 schema 锁住，事件 / final_observation 不携带 nether_entry_evaluation / latched_frame_identity / nether_portal / matched_frame_identity / agents_in_nether / pre_transition_position / entered_via_episode_portal / source_dimension / target_dimension / FrozenNetherEntry / NetherEntryEvidence 等 token；
   - 端到端 orchestrator（driver + `set_nether_entry_evaluation_state` + `FrozenNetherEntryEvaluator`）返回 `success` / `nether_entry_portal_unknown` / `nether_entry_not_via_episode_portal` / `wrong_entry_agent` / `wrong_source_dimension` / `wrong_target_dimension` / `transition_step_missing` / `pre_transition_position_missing` / `transition_before_activation` / `frame_identity_mismatch` / `step_budget_exceeded` / `ignition_not_completed` 等 outcome；4-step 因果窗口 delta = 4 内 success，5+ 步 activation_outside_window 闭合到 ignition_not_completed；
   - FakeBackend C1 / C2 / C3 / C4 / C5 槽位独立与互不污染（同一 backend 注入 C1 / C2 后 C3 / C4 / C5 仍空；close 清空所有 5 个槽位；`casting_s_c5_fixed` workflow 校验缺位 fail closed）；
   - C1 / C2 / C3 / C4 ignition / C5 Nether-entry evaluator 回归不受影响；
   - 离线 `--check` 与 `check_environment.py` 输出 `phase="r6_c5_deterministic_driver"`。

6. **C1 / C2 / C3 / C4 / C5 回归不受影响**：修复后 C5 driver 专项测试 **142 个**全过；完整测试集、离线检查与 diff 检查结果见本节最终验证记录。

7. **evaluator-only 信息隔离验证**：
   - 整个 driver 源文件（`obsidianlink/drivers/casting_s_c5_nether_entry.py`）的 AST 检查 + `ast.Attribute` / `ast.Subscript` 扫描确认：`set_nether_entry_evaluation_state` / `get_nether_entry_evaluation_state` / `clear_nether_entry_evaluation_state` / `_nether_entry_evaluation_state` / `_ignition_evaluation_state` / `_frame_evaluation_state` / `_continuous_casting_evaluation_state` / `_casting_evaluation_state` / `FrozenFrameIdentity` / `IgnitionActionEvidence` / `PortalActivationEvidence` / `FrozenIgnitionEvaluationState` / `FrozenIgnitionEvaluationResult` / `FrozenIgnitionEvaluator` / `FrozenNetherEntryEvaluationState` / `FrozenNetherEntryEvaluationResult` / `FrozenNetherEntryEvaluator` / `NetherEntryEvidence` / `build_c4_c3_frame_identity` / `evaluator_contract` / `agents_in_nether` / `entered_via_episode_portal` / `matched_frame_identity` / `latched_frame_identity` / `pre_transition_position` / `scenario_parameters` 全部 0 命中；driver 表面无访问 C5 / C4 / C3 / 任何 evaluator truth 接口的能力；
   - `Observation` 公开字段集（8 个字段）保持不变；C5 Nether-entry truth 通过 FakeBackend 显式 set/get 注入，不进入 `Observation`；test 验证 15 个 C5 关键字 token 全部缺席于 `Observation` 任意字符串字段与列表元素；
   - test orchestrator 在 `tests/test_r6_casting_c5_nether_entry_driver.py` 内**独立**通过 `set_nether_entry_evaluation_state` 注入 truth；driver 表面无访问 C5 truth 接口的能力。

8. **未验证限制**：
   - 真实 MineRL / Minecraft 浇筑、门框建造、点火、Nether entry 仍未验证；
   - 真实 backend 仍未完整接通 use_item 动作、公开 selected item、目标方块 truth、流体 truth、nether_portal 出现、pre-transition 位置、维度切换 truth；
   - 真实 task-origin marker 与 evaluator truth-grid origin 的世界坐标锚定、Ruined / Adaptive / Multi-Agent 仍**未**实现；
   - 当前仅在 FakeBackend 上完成离线证明；C5 driver 与真实 backend 的接线下游仍需后续阶段验证；
   - 真实 MineRL 中 Agent 初始朝向、portal 平面与固定前进轨迹的对齐尚未验证；当前有限 `move` 序列只完成 FakeBackend 协议证明；
   - 4 个新增 PHASE 常量与 4 个新 ROLE 常量（`entry_approach` / `entry_align` / `entry_teleport` / `entry_settle`）保留在 `PHASE_VALUES` / `ROLE_VALUES` 闭集内。

9. **下一任务只能根据本次真实完成范围谨慎填写**：本轮 R6-C5-DETERMINISTIC-DRIVER 已完成 C5 deterministic driver + 严格 public context 边界 + 完整计划 fail-closed 校验 + capability gate + 142 个专项测试，**没有**提前实现真实 MineRL、Gradle 或模型 API；下一任务已冻结为 `R6-C5-LIVE-MINERL-BACKEND-WIRING`，真实 backend 接入需用户单独授权。

## R6-C5-NETHER-ENTRY-EVALUATOR 已完成（FakeBackend 离线证明）

1. 新增 [`obsidianlink/evaluation/casting_nether_entry_evaluator.py`](obsidianlink/evaluation/casting_nether_entry_evaluator.py)：
   - `NetherEntryEvidence`、`FrozenNetherEntryEvaluationState`、`FrozenNetherEntryEvaluationResult` 均为 frozen、严格类型、可 JSON 序列化类型；unknown attribution 使用 `None` 与明确 `False` 分离；
   - `FrozenNetherEntryEvaluator` 是纯确定性 evaluator，内部复用 `FrozenIgnitionEvaluator` 重新验证 C4 success，不读取 observation、task scenario、driver、planner、Agent 或 workflow；
   - 指定 `agent_1` 必须从 `minecraft:overworld` 进入 `minecraft:the_nether`，transition 不得早于 activation，并要求切换前位置、`entered_via_episode_portal=True` 和与 C4 latched identity 完全一致的 typed `FrozenFrameIdentity`；
   - 合同指定的 `nether_entry_portal_unknown` / `nether_entry_not_via_episode_portal` 已实现；错误 Agent/维度、缺 transition step/position、transition 早于 activation、identity 缺失/不匹配、预算与异常终止均 fail closed。

2. FakeBackend 新增独立 `_nether_entry_evaluation_state` 及 set/get/clear API：只接受 `casting_s_c5_fixed`，严格校验 episode/step/agent，reset/step/close 自动清空；C5 truth 不进入 Observation，并与 C1–C4 truth 槽隔离。

3. `casting_s_c5_fixed` 加入 reset 前 capability gate；C5 experiment config 指向离线 evaluator，但继续保持 `status=contract_only`、`allow_live_run=false`、`backend=not_implemented`、`planner=not_implemented`。

4. 新增 9 个专项离线测试，覆盖 success、确定性 replay、JSON 快照、闭集 outcome、全部冻结归因失败路径、C4 failure 不可被进入声明覆盖、严格类型/不可变性、FakeBackend C5 truth 槽生命周期/身份门锁与 evaluator AST 信息隔离；全量 947 个离线测试通过。

5. 未实现/未验证：真实 MineRL/Minecraft 维度切换证据采集、正式 runner 接线、真实 task-origin/world-coordinate anchor、Ruined/Adaptive/Multi-Agent、Gradle 和模型 API。

6. 下一唯一任务是真实 backend 接入或下一阶段工程任务；不得提前接真实 MineRL 或模型。

## R6-C4-DETERMINISTIC-DRIVER 已完成（FakeBackend 离线证明）

1. **新增 C4 公开 19-cell 上下文边界** [`obsidianlink/core/casting_s_c4_ignition_context.py`](obsidianlink/core/casting_s_c4_ignition_context.py)：
   - `build_public_c4_ignition_driver_context_from_task(task)` 是整个 R6 C4 driver 家族中**唯一**允许读取 task `scenario_parameters` 的函数；它只读取 `public_task_spec.frame_plan.fixed_offsets` 与 `public_task_spec.ignition_plan`（action=`use_item`、item=`flint_and_steel`、target_offset=`[1, 1, 1]`、target_policy=`exact`），以及 task limits / initial inventories，**忽略** `evaluator_contract` 与任何 evaluator-only 字段；
   - 返回严格冻结的 [`PublicC4IgnitionDriverContext`](obsidianlink/drivers/casting_s_c4_ignition.py)：workflow / family / mode / level / layout / agent_id / 14 个有序 target offsets / 公开 ignition plan / 不可变 initial_inventory（MappingProxyType）/ task_step_limit / task_time_limit；任何未知 family / mode / level / layout、越界或重排 target offsets、bool 充当 int、缺失或重复 cell、ignition 字段与公开值不一致、`ignition_plan.required` 缺失或非 bool、库存缺失或 bool 都 fail closed，context builder 不再用 `bool(...)` 掩盖原始类型错误。

2. **新增 C4 deterministic driver** [`obsidianlink/drivers/casting_s_c4_ignition.py`](obsidianlink/drivers/casting_s_c4_ignition.py)：
   - 显式接受 immutable `PublicC4IgnitionDriverContext` 作为唯一 TaskInstance-shaped 输入；`run_casting_s_c4_ignition_driver` 不读取 scenario_parameters / evaluator_contract / `FrozenFrameIdentity` / `IgnitionActionEvidence` / `PortalActivationEvidence` / `FrozenIgnitionEvaluationState` 或任何 evaluator-only 字段；
   - 默认 plan：14 cell × 24 step = 336 step 的 C3 浇筑子计划 + 4 step 的 C4 ignition 子计划（equip + release + use + portal settle）= **340 step**，落在 700 step 任务预算内；每 cell 2 个 `use_item(water|lava)` 视为 evaluator 的 relevant action，2 个 `place_block(cobblestone)` 支撑方块是 plan 内的机械动作但不进入 per-cell evidence；1 个 `use_item(flint_and_steel)` 是 ignition relevant action；
   - 运行入口对 caller-supplied plan 做整体等值校验：只能接受由公开 context 与恢复设置生成的完整确定性计划；仅点火、截断、重排或重复点火计划在 reset 前 fail closed，不能以 `completed` 绕过 14-cell 浇筑；
   - 动作白名单严格闭合：`equip_item` / `use_item` / `place_block` / `wait`；目标白名单严格闭合：`water_bucket` / `lava_bucket` / `cobblestone` / `flint_and_steel`；永不放置 `obsidian`、永不下 Nether；ignition target 必须与公开 `[1, 1, 1]` 一致；
   - 预算硬上限：environment step 700、game time 640 秒、plan wait 320、plan length 700、per-action recovery 2、total recovery 32、per-step `duration_ticks` 1..40；
   - 恢复只响应 typed `RecoverableBackendError`，受 per-action 与 total 双重预算；其他异常 fail closed；
   - driver status 闭集 `completed` / `blocked` / `failed`；永不返回 `success` / `passed`（这些 verdict 仍由 `FrozenIgnitionEvaluator` 独立判定）；
   - 结构化事件必须带 `episode_id` / `step_id` / `agent_id` / `cell_index` / `target_offset` / `label` / `phase` / `action_type` / `target` / `relevant_action` / `role` / `attempt`；结果对象 `CastingC4IgnitionDriverResult` 不可变、类型严格、可序列化、暴露 `as_dict()`；
   - 新增 `ignition_relevant_action_step` / `ignition_target_offset` / `ignition_equip_step` 让 orchestrator 完全不读 evaluator truth 即可独立构造 `IgnitionActionEvidence`；新增 `per_cell_relevant_action_records` 复用 C3 表面给 C3 frame evaluator；
   - 5 个新 PHASE 常量（ignition_equip / ignition_use / ignition_portal_settle 等）与 4 个 ROLE 常量（cast / ignition_equip / ignition_use / ignition_settle）严格分离 C3 浇筑与 C4 点火子计划；
   - 模块 AST 检查确认：未 import `casting_ignition_evaluator` / `casting_frame_evaluator` / `agents` / `workflows` / `model` / `planner` 等；未通过 `ast.Attribute` 访问 `scenario_parameters` / `evaluator_contract` / `_ignition_evaluation_state` / `set_ignition_evaluation_state` / `FrozenFrameIdentity` 等；未调用 `set_ignition_evaluation_state` / `get_ignition_evaluation_state` / `clear_ignition_evaluation_state`。

3. **不重命名旧 driver**：旧 `obsidianlink/drivers/casting_c3.py` 仍是 R5 / Casting-S-C2 的三 cell 连续浇筑 driver；`obsidianlink/drivers/casting_s_c3_frame.py` 仍是 R6-C3 14-cell driver；新模块明确命名为 `casting_s_c4_ignition.py`，反映 B0 taxonomy 的 C4（点火）合同。三个 driver 互不共享类型。

4. **capability gate 升级** [`obsidianlink/env/capabilities.py`](obsidianlink/env/capabilities.py)：`_GATED_WORKFLOWS` 新增 `casting_s_c4_fixed`，确保 reset 前的 capability 检查覆盖 C4 合同；缺桶/选物品/selected_item/target_block_truth/fluid_truth 任一能力时 fail closed。

5. **新增 159 个专项离线测试** [`tests/test_r6_casting_c4_ignition_driver.py`](tests/test_r6_casting_c4_ignition_driver.py)，覆盖：
   - 公开上下文严格解析与不可变性（含 family / mode / level / layout / agent / 14 cell / grid 边界 / 重复 / 重排 / bool 充当 int / 库存缺失 / bool 数量 / 库存未知 item / step 预算下限 / ignition 字段与公开值不一致）；
   - `build_casting_s_c4_ignition_action_plan` 固定 14-cell × 24 + 4 = 340 步、240 wait、28 cast relevant + 1 ignition relevant、14 × 17 + 2 = 240 wait、闭集 action / 物品 / 阶段 / role；
   - Plan step validation（cast role 需要 cell_index、ignition role 需要 cell_index=None、target validation、ignition_use step 验证等）；
   - 公开 ignition plan 验证（wrong action / wrong item / wrong target / wrong target_policy / ignition_required=False / required 缺失 / 字符串或整数 required 被严格拒绝）；
   - Driver result contract（frozen、status 闭集、ignition 字段必填、as_dict() detached、JSON 序列化、events 闭集）；
   - Driver 完整执行（status=completed、340 step、per_cell 2 records、ignition records 339 step、ignition equip 337 step、events 不携带任何 evaluator token）；
   - 640 step / 640 秒 / 320 wait / 700 plan / 32 total recovery / 2 per-step recovery 预算失败的 fail-closed 行为；
   - 重复 `RecoverableBackendError` 有限重试、per-step 与 total budget 各自独立耗尽、metadata 透传；
   - 非 `RecoverableBackendError` 异常（RuntimeError / OSError / TypeError）立即 fail closed；
   - backend 提前 terminated / truncated 的 blocked 行为；
   - 每个事件的 episode / step / agent / cell / target_offset / relevant_action / role 身份；
   - 相同输入的 action 序列、events 与 `as_dict()` 快照完全一致（确定性 replay）；
   - 闭集 status（`completed` / `blocked` / `failed`），驱动从不返回 `success` / `passed`；
   - 闭集 PHASE_VALUES（9 个）与 ROLE_VALUES（4 个）锁住；
   - capability gate 缺失能力时 reset 前 fail closed；
   - driver 拒绝错误 backend（无 reset/step）、错误 plan 类型、仅点火计划、重复点火计划、plan 超长、max_environment_steps 越界、max_game_time_seconds 越界、max_wait_steps 越界、total_recovery_budget 越界等；
   - AST + 源码双门锁：driver 源文件**不** import ignition evaluator、**不**调用 `set_ignition_evaluation_state` / `get_ignition_evaluation_state` / `clear_ignition_evaluation_state` / `_ignition_evaluation_state`、**不**以代码形式访问 `scenario_parameters` / `evaluator_contract` / `FrozenFrameIdentity` / `IgnitionActionEvidence` / `PortalActivationEvidence` 等；
   - Observation 8 字段 schema 锁住，事件 / final_observation 不携带 ignition / latched_frame_identity / nether_portal 等 token；
   - 端到端 orchestrator：driver + `set_ignition_evaluation_state` + `FrozenIgnitionEvaluator` 返回 `success` / `activation_outside_window` / `activation_before_ignition` / `external_activation` / `wrong_ignition_agent` / `frame_not_built` / `truth_missing` / `step_budget_exceeded` / `ignition_action_missing` / `portal_activation_missing` / `wrong_ignition_action` / `wrong_ignition_item` / `wrong_ignition_target` 各 outcome；4-step 因果窗口 delta = 0/1/4 都 success、5/6 步 activation_outside_window、delta < 0 activation_before_ignition；
   - FakeBackend C1 / C2 / C3 / C4 truth 槽位独立与互不污染（同一 backend 注入 C1 与 C2 状态后 C3 / C4 仍空；close 清空所有 4 个槽位；`casting_s_c4_fixed` workflow 校验缺位 fail closed）；
   - C1 / C2 / C3 / C4 ignition evaluator 回归不受影响；
   - 离线 `--check` 与 `check_environment.py` 仍输出 `status: "ok"`。

6. **C1 / C2 / C3 / portal / 既有 R6 C3 frame evaluator / 既有 R6 C4 ignition evaluator 回归不受影响**：完整测试集 **938 个**全过（779 个旧测试 + 159 个新测试，`python -m unittest discover -s tests -p 'test_*.py'` → `Ran 938 tests in 55.746s` → `OK`），`python -m obsidianlink --check` 与 `python scripts/check_environment.py` 均通过且报告 `phase="r6_c4_deterministic_driver"`；`git diff --check` 干净。

7. **evaluator-only 信息隔离验证**：
   - 整个 driver 源文件（`obsidianlink/drivers/casting_s_c4_ignition.py`）的 AST 检查 + `ast.Attribute` / `ast.Subscript` 扫描确认：`set_ignition_evaluation_state` / `get_ignition_evaluation_state` / `clear_ignition_evaluation_state` / `_ignition_evaluation_state` / `_frame_evaluation_state` / `_continuous_casting_evaluation_state` / `_casting_evaluation_state` / `FrozenFrameIdentity` / `IgnitionActionEvidence` / `PortalActivationEvidence` / `FrozenIgnitionEvaluationState` / `FrozenIgnitionEvaluationResult` / `FrozenIgnitionEvaluator` / `build_c4_c3_frame_identity` / `evaluator_contract` / `scenario_parameters` 全部 0 命中；driver 表面无访问 ignition / frame truth 接口的能力；
   - `Observation` 公开字段集（8 个字段）保持不变；C4 ignition truth 通过 FakeBackend 显式 set/get 注入，不进入 `Observation`；test 验证 ignition / latched_frame_identity / nether_portal / wrong_ignition / frame_not_built / public_ignition_target / frame_interior / FrozenFrameIdentity / FrozenIgnition / IgnitionActionEvidence / PortalActivationEvidence 等 12 个关键字 token 全部缺席于 `Observation` 任意字符串字段与列表元素；
   - test orchestrator 在 `tests/test_r6_casting_c4_ignition_driver.py` 内**独立**通过 `set_ignition_evaluation_state` 注入 truth；driver 表面无访问 frame truth 接口的能力。

8. **未验证限制**：
   - 真实 MineRL / Minecraft 浇筑、门框建造与点火仍未验证；
   - 真实 backend 仍未完整接通 use_item 动作、公开 selected item、目标方块 truth、流体 truth、维度切换 truth；
   - C5 Nether-entry evaluator/driver、真实 task-origin marker 与 truth-grid origin 的世界坐标锚定、Ruined / Adaptive / Multi-Agent 仍**未**实现；
   - 当前仅在 FakeBackend 上完成离线证明；C4 driver 与真实 backend 的接线下游仍需后续阶段验证；
   - 9 个新增 PHASE 常量与 4 个 ROLE 常量在闭集内（`PHASE_VALUES` / `ROLE_VALUES`），但 PHASE_RECOVERY 仍保留为 driver 内部事件专用，不进入 plan。

9. **下一任务只能根据本次真实完成范围谨慎填写**：本轮 R6-C4-DETERMINISTIC-DRIVER 已完成 C4 deterministic driver + 严格 public context 边界 + 完整计划 fail-closed 校验 + capability gate + 159 个专项测试通过 + 全量 938 个测试通过，**没有**提前实现 C5 Nether-entry evaluator/driver、真实 MineRL、Gradle 或模型 API；下一任务严格限定为 `R6-C5-NETHER-ENTRY-EVALUATOR`。

1. **新增 C4 ignition evaluator** [`obsidianlink/evaluation/casting_ignition_evaluator.py`](obsidianlink/evaluation/casting_ignition_evaluator.py)：
   - 不可变、类型严格、可序列化的 evaluator-only 表面：
     `FrozenFrameIdentity` /
     `IgnitionActionEvidence` / `PortalActivationEvidence` /
     `FrozenIgnitionEvaluationState` /
     `FrozenIgnitionEvaluationResult` /
     `FrozenIgnitionEvaluator`；
   - 闭集 outcome id 19 个（`IGNITION_OUTCOMES`）：`success` /
     `in_progress` / `frame_not_built` /
     `ignition_action_missing` / `wrong_ignition_agent` /
     `wrong_ignition_action` / `wrong_ignition_item` /
     `wrong_ignition_target` / `portal_activation_missing` /
     `activation_before_ignition` /
     `activation_outside_window` / `external_activation` /
     `frame_identity_missing` / `frame_identity_mismatch` /
     `truth_missing` / `step_budget_exceeded` /
     `time_budget_exceeded` / `invalid_initial_state` /
     `abnormal_termination`；
   - 闭集 per-event verdict（`ignition_verdict` /
     `ignition_agent_verdict` / `ignition_action_verdict` /
     `ignition_item_verdict` / `ignition_target_verdict` /
     `activation_verdict` / `activation_window_verdict` /
     `activation_offset_verdict` /
     `activation_agent_verdict` /
     `frame_identity_verdict`），全部在 `__post_init__` 校验
     并由测试套件锁定；
   - 闭集常量 `CASTING_S_C4_AGENT_ID = "agent_1"` /
     `CASTING_S_C4_IGNITION_ITEM = "flint_and_steel"` /
     `CASTING_S_C4_IGNITION_ACTION_TYPE = "use_item"` /
     `CASTING_S_C4_PUBLIC_IGNITION_TARGET = (1, 1, 1)` /
     `CASTING_S_C4_FRAME_INTERIOR_CELLS`（6 个内部 cell 的
     frozenset）以及 4×5 full-ring 的 orientation / min_corner
     / max_corner / width / height 公开常量；
   - `causality_window_steps` 默认 4（与 R3 / R5 / C3 一致），
     最大 32；activation.delta ∈ [0, 4] 视为窗口内，> 4 视为
     超出窗口，< 0 视为早于 ignition（`activation_before_ignition`）；
2. **typed frame identity 合同 `FrozenFrameIdentity`**：
   - frozen dataclass，13 个显式字段：`orientation` /
     `min_corner` / `max_corner` / `width` / `height` /
     `target_offsets` / `interior_offsets` /
     `required_corner_count` / `required_full_ring_count` /
     `activation_offsets` / `episode_id` / `step_id` / `agent_id`；
   - 构造期只做结构/类型校验（xyz 元组、非负/正整数、orientation
     非空、min ≤ max、bool 拒绝、JSON 兼容值树）；语义判断（是否
     匹配 C3 固定门框）由 evaluator 完成；
   - `as_dict()` 返回 detached、JSON-serializable 快照；
   - 单一权威构造器 `build_c4_c3_frame_identity(episode_id, step_id,
     agent_id="agent_1", activation_offsets=())`：它从
     `CASTING_S_C3_FRAME_CELLS` /
     `CASTING_S_C3_INTERIOR_CELLS` /
     `CASTING_S_C3_CORNER_CELL_COUNT` /
     `CASTING_S_C3_TARGET_CELL_COUNT` 与 C4 公开
     `orientation` / `min_corner` / `max_corner` / `width` /
     `height` 拼出唯一可被 evaluator 接受的 episode-built 身份；
   - 两个相同但任意的 mapping 不再能冒充成功——identity 必须是
     typed `FrozenFrameIdentity` 实例且所有几何字段与 C3 固定门框
     一致；任何 orientation / corner / width / height /
     target_offsets / interior_offsets 偏差都让
     `_check_frame_identity` 在 priority 4 阶段产出
     `OUTCOME_FRAME_IDENTITY_MISMATCH` +
     `FRAME_IDENTITY_VERDICT_GEOMETRY_MISMATCH`；
3. **分层构造合同**（构造器只做结构、evaluator 做语义）：
   - `IgnitionActionEvidence` 构造期只校验 `episode_id` /
     `step_id` / `agent_id` / `action_type` / `item` /
     `target_cell` 的类型、xyz 元组、bool / 负数 / 空字符串拒绝；
     `agent_id` 是不是 `agent_1`、`action_type` 是不是 `use_item`、
     `item` 是不是 `flint_and_steel`、`target_cell` 是不是
     `(1, 1, 1)` 等语义判断**全部交给** evaluator 产出
     `OUTCOME_WRONG_IGNITION_AGENT` /
     `OUTCOME_WRONG_IGNITION_ACTION` /
     `OUTCOME_WRONG_IGNITION_ITEM` /
     `OUTCOME_WRONG_IGNITION_TARGET`；
   - `PortalActivationEvidence` 构造期校验 `episode_id` /
     `update_step` / `agent_id` / `nether_portal_offset` /
     `latched_frame_identity`（**必须**是已构造的
     `FrozenFrameIdentity` 实例），但**不**做 interior set 成员
     判断、**不**做 agent-id 语义判断、**不**做 identity geometry
     判断；这些语义判断交给 evaluator 产出
     `OUTCOME_EXTERNAL_ACTIVATION` /
     `OUTCOME_WRONG_IGNITION_AGENT`（activation agent 错时）/
     `OUTCOME_FRAME_IDENTITY_MISMATCH`；
   - `PortalActivationEvidence.agent_id` 现在是**必填**非空标识符
     （不是 `Optional`），由构造期 fail closed 拒绝空 / 非字符串；
     语义上要求与 ignition action agent 一致，由 evaluator 在
     priority 8 产出 `OUTCOME_WRONG_IGNITION_AGENT`；
   - `FrozenIgnitionEvaluationState` 构造期严格身份校验：
     `frame_state.episode_id` / `step_id` / `agent_id` /
     `max_environment_steps` / `max_game_time_seconds` /
     `episode_terminated` / `terminated_step` / `terminated_reason`
     与 C4 wrapper 完全一致；`frame_state.frame_outcome ==
     "in_progress"` 时 wrapper 也必须 `in_progress`；
     `ignition_action.step_id ≤ step_id`、
     `activation.update_step ≤ step_id`；
     `latched_frame_identity.episode_id` / `agent_id` 与 wrapper
     一致；`latched_frame_identity.step_id ≤ step_id`；
4. **优先级（高 → 低）由 `FrozenIgnitionEvaluator.evaluate()` 锁定**：
   1. `step_budget_exceeded`
   2. `time_budget_exceeded`
   3. `abnormal_termination`（仅在 `episode_terminated` 且
      `terminated_reason ∉ NORMAL_TERMINATION_REASONS` 时）
   4. **state-level frame identity geometry check（priority 4）**：
      `state.latched_frame_identity` 的 orientation / min_corner /
      max_corner / width / height / target_offsets /
      interior_offsets / required_corner_count /
      required_full_ring_count 必须与 C3 固定门框完全一致；任一
      偏差 → `OUTCOME_FRAME_IDENTITY_MISMATCH` +
      `FRAME_IDENTITY_VERDICT_GEOMETRY_MISMATCH`，**在任何 C3
      success 检查之前** fail closed；
   5. C3 `in_progress` 透传：frame_outcome == `in_progress` →
      `OUTCOME_IN_PROGRESS`；
   6. `frame_not_built`（C3 frame 评估非 success 也非 in_progress）
   7. `ignition_action_missing`
   8. `wrong_ignition_agent`（ignition agent 不对）
   9. `wrong_ignition_action`（ignition action_type 不对）
   10. `wrong_ignition_item`（ignition item 不对）
   11. `wrong_ignition_target`（ignition target_cell 不对）
   12. `portal_activation_missing`
   13. `OUTCOME_WRONG_IGNITION_AGENT`（activation agent 与 ignition
       agent 不一致）
   14. `activation_before_ignition`（delta < 0）
   15. `activation_outside_window`（delta > 4）
   16. `external_activation`（offset 不在 `CASTING_S_C4_FRAME_INTERIOR_SET`）
   17. `frame_identity_mismatch`（state identity 与 activation identity
       不一致）
   18. `in_progress`（episode 未 terminated 但其它条件都满足）
   19. `success`（episode terminated + normal reason + 全部条件满足）；
5. C4 evaluator 内部**复用**
   [`FrozenFrameEvaluator`](obsidianlink/evaluation/casting_frame_evaluator.py)
   重新验证 C3 14-cell 浇筑条件（不重复实现 frame 评估逻辑），
   C3 评估结果在 `FrozenIgnitionEvaluationResult.frame_outcome`
   中暴露，便于审计关联；
6. 闭集 `failure_type` 与 outcome 对齐；`as_dict()` 返回
   detached、JSON-serializable 快照；相同 state 重复 evaluate
   产生完全相同 result 与 `as_dict()`。
7. **FakeBackend C4 独立 truth 槽位** [`obsidianlink/env/fake.py`](obsidianlink/env/fake.py)：
   - 新增 `_ignition_evaluation_state: FrozenIgnitionEvaluationState | None`
     与 `set_ignition_evaluation_state` /
     `get_ignition_evaluation_state` /
     `clear_ignition_evaluation_state` 三个方法；
   - 与 `_casting_evaluation_state` (C1) /
     `_continuous_casting_evaluation_state` (C2) /
     `_frame_evaluation_state` (C3) **完全独立**的 4 个槽位
     在同一 FakeBackend 上共存；
   - `set_ignition_evaluation_state` 严格身份校验：类型
     `FrozenIgnitionEvaluationState` / `task.workflow ==
     "casting_s_c4_fixed"` / `episode_id == task.task_id` /
     `step_id == self._step_id` / `agent_id ∈ task.agent_ids`，
     任一不符 fail closed；
   - `reset()` / `step()` / `close()` 一律清空 C4 槽位，杜绝跨
     step / episode 的 truth 泄漏；FakeBackend 永远不会把
     ignition truth 复制到 `Observation`；
   - 公开 `Observation` schema 字段集（`episode_id` / `agent_id` /
     `step_id` / `timestamp` / `frame` / `visible_inventory` /
     `messages` / `workflow_stage`）保持不变。
8. **`obsidianlink/evaluation/__init__.py`** 新增 C4 ignition 相关
   19 个 outcome 常量、5 个 per-event verdict 集合、4 个 frame
   identity verdict、3 个 activation 字段、5 个 ignition 字段、
   1 个 C4 causality_window、6 个 interior cell 常量、4 个
   公开 4×5 frame geometry 常量、3 个 `_MISMATCH` 阻塞条件标签
   以及 `FrozenFrameIdentity` / `FrozenIgnitionEvaluationState` /
   `FrozenIgnitionEvaluationResult` / `FrozenIgnitionEvaluator` /
   `IgnitionActionEvidence` / `PortalActivationEvidence` /
   `build_c4_c3_frame_identity` 类型，全部加入 `__all__`。
9. **新增 144 个专项离线测试** [`tests/test_r6_casting_c4_ignition_evaluator.py`](tests/test_r6_casting_c4_ignition_evaluator.py)，覆盖：
   - 闭集 outcome / per-event verdict 常量稳定字符串 /
     `frame_interior` 6 cell / `public_ignition_target` 冻结；
   - **公开构造 API 下的语义错误**：wrong agent / wrong
     action_type / wrong item / wrong target 全部通过正常构造
     `IgnitionActionEvidence(..., action_type="place_block")` /
     `item="water_bucket"` / `target_cell=(2, 1, 1)` /
     `agent_id="agent_2"` 注入，由 evaluator 产出
     `OUTCOME_WRONG_IGNITION_*`；external activation 通过
     `_activation(nether_portal_offset=(0, 0, 1))` /
     `(5, 0, 1)` 公开构造，由 evaluator 产出
     `OUTCOME_EXTERNAL_ACTIVATION`；**没有任何业务负例**依赖
     `object.__setattr__` 篡改 frozen dataclass；
   - **malformed 构造期拒绝**：空 episode_id / 空 agent_id /
     bool step_id / bool update_step / 非 xyz 元组 target /
     负 step_id / 非 string orientation / min_corner > max_corner
     / 非 int width / 非 str identity keys / `latched_frame_identity`
     缺失或不是 `FrozenFrameIdentity` / `causality_window_steps`
     为 0 或非 int 全部 fail closed；
   - **typed frame identity 正反例**：
     `build_c4_c3_frame_identity` 默认产出被 evaluator 接受；
     orientation / min_corner / max_corner / width / height /
     target_offsets / interior_offsets 任意字段偏移 →
     `OUTCOME_FRAME_IDENTITY_MISMATCH` +
     `FRAME_IDENTITY_VERDICT_GEOMETRY_MISMATCH`；
     target / interior offsets 重排、重复均 fail closed；
     activation offsets 必须为非空、无重复、canonical-order 的内部
     子集，且必须包含实际观测 `nether_portal_offset`；
     任意但相等的 mapping 不再 success（identity 必须是 typed
     `FrozenFrameIdentity`）；state identity 与 activation identity
     mismatch → `OUTCOME_FRAME_IDENTITY_MISMATCH`；identity
     `as_dict()` detached、JSON-serializable、frozen 不可写；
   - **4-step 因果窗口**：delta = 0 / 1 / 4 全部 success，5 / 6 步
     → `activation_outside_window`，delta < 0（早于 ignition）→
     `activation_before_ignition`；
   - **C3 non-success 不能被 C4 覆盖**：
     `partial_completion` / `wrong_block` / `interior_blocked` /
     `causality_missing` 全部产出 `frame_not_built`；
     `truth_missing` 透传；C3 `in_progress` 透传为 C4 `in_progress`；
   - **身份一致性**：state 拒绝 non-int step_id（字符串）、
     bool step_id、负 step_id、空 episode_id、非 int
     max_environment_steps、bool max_environment_steps；
     `latched_frame_identity` 的 episode_id / agent_id / step_id
     与 wrapper 不一致时构造期 fail closed；
     `activation.agent_id` 缺失（空字符串）构造期 fail closed；
     `activation.agent_id != ignition.action.agent_id` → priority 13
     `OUTCOME_WRONG_IGNITION_AGENT`；
   - **truth missing / budget / abnormal termination**：
     ignition_action_missing（含 evidence vs no evidence）、
     activation_missing、terminated 但无 reason（构造期拒绝）、
     step 超过 `max_environment_steps` → `step_budget_exceeded`、
     time 超过 `max_game_time_seconds` →
     `time_budget_exceeded`、`terminated_reason="explosion"` →
     `abnormal_termination`、所有 `NORMAL_TERMINATION_REASONS` 都
     能得到 success；
   - **deterministic replay** 与稳定 `as_dict()`：相同 state 重复
     `evaluate()` 产生完全相同 result 与 detached JSON 快照；
     `as_dict()` 不允许 list 替代 tuple offset；`as_dict()` 用
     `json.dumps` / `json.loads` 双向 round-trip 不丢失信息；
   - **FakeBackend C1 / C2 / C3 / C4 truth 槽位互不污染**：
     4 套 set / get / clear 在同一 C4 任务上各自工作；`reset` /
     `step` / `close` 清空 C4 槽位；wrong episode / wrong step /
     wrong type / wrong workflow / wrong agent 全部 fail closed；
     C1 / C2 state 不能注入 C4 任务（workflow 不匹配 fail closed）；
   - **Observation 不泄漏**：walking `Observation` 字段无
     `FrozenIgnition` / `ignition_evaluation` / `latched_frame_identity`
     / `nether_portal` / `flint_and_steel` / `wrong_ignition` /
     `frame_not_built` / `public_ignition_target` / `frame_interior`
     等 token；schema 字段集严格保持 8 个公开字段；`step` 后
     frame 仍只含 `{"backend": "fake", "step_id": N}`；
   - **evaluator AST 隔离**：evaluator 源文件不 import
     `obsidianlink.agents` / `obsidianlink.workflows` /
     `obsidianlink.drivers`；不通过 `ast.Attribute` 访问
     `scenario_parameters` / `evaluator_contract` /
     `instruction`；`evaluate()` 第二参数严格注解为
     `FrozenIgnitionEvaluationState`；
   - **immutability 专项测试**：仅在此组测试中用
     `object.__setattr__` 验证 `FrozenInstanceError`，不依赖
     篡改后的对象证明业务结论；
   - **C1 / C2 / C3 / portal 回归**：`CastingEvaluator`、
     `ContinuousCastingEvaluator`、`FrozenFrameEvaluator`、
     `PortalEvaluator` 全部仍能 success。
10. **全量离线测试通过**：
    `python -m unittest discover -s tests -p 'test_*.py'` →
    `Ran 779 tests in 54.446s` → `OK`（R6-C3-frame + driver 之前的
    635 个测试 + R6-C4-ignition 144 个新测试，无回归）；
    `python -m obsidianlink --check` → `status: "ok"`；
    `python scripts/check_environment.py` →
    `python: "3.10.20"`、catalog 7 条；`git diff --check` 干净。
11. **evaluator-only 信息隔离验证**：
    - C4 ignition evaluator 源文件
      （`obsidianlink/evaluation/casting_ignition_evaluator.py`）
      的 AST 检查确认：未 import `obsidianlink.agents` /
      `obsidianlink.workflows` / `obsidianlink.drivers`；
      未通过 `ast.Attribute` 访问 `scenario_parameters` /
      `evaluator_contract` / `instruction`；`evaluate()` 第二参数
      严格注解为 `FrozenIgnitionEvaluationState`；
    - C4 state 构造时 `frame_state.episode_id` / `step_id` /
      `agent_id` / `budget` / `episode_terminated` /
      `terminated_step` / `terminated_reason` 与 C4 wrapper 严格
      相等——任一不一致在 `__post_init__` 阶段就 fail closed；
    - `latched_frame_identity.episode_id` / `agent_id` 与
      wrapper 一致；`latched_frame_identity.step_id ≤ step_id`；
    - `Observation` 公开字段集（8 个字段）保持不变；C4 ignition
      truth 通过 FakeBackend 显式 set/get 注入，不进入
      `Observation`；test 验证 24 个 ignition 关键字 token 全部
      缺席于 `Observation` 任意字符串字段与列表元素；
    - test orchestrator 独立通过 `set_ignition_evaluation_state`
      注入 C4 truth，evaluator 与 FakeBackend 互不读对方 surface。
12. **未验证限制**：
    - 真实 MineRL / Minecraft 浇筑与门框建造仍未验证；
    - 真实 backend 仍未完整接通 use_item 动作、目标方块 truth、
      nether_portal block 出现检测、pre-transition 位置等；
    - C4 ignition driver、C5 Nether-entry evaluator/driver、
      真实坐标锚定、Ruined / Adaptive / Multi-Agent 仍**未**
      实现；
    - 当前仅在 FakeBackend 上完成离线证明；C4 / C5 driver 与
      真实 backend 的接线下游仍需后续阶段验证；
    - 4 个新增 `_MISMATCH` outcome 常量
      （`OUTCOME_IGNITION_AGENT_MISMATCH` /
      `OUTCOME_IGNITION_ACTION_MISMATCH` /
      `OUTCOME_IGNITION_ITEM_MISMATCH` /
      `OUTCOME_IGNITION_TARGET_MISMATCH`）保留在 `__all__` 中
      作为审计可见的 blocking-condition 标签，但
      `IGNITION_OUTCOMES` 闭集（19 个）保持不变；评估顶层仍
      走 `OUTCOME_WRONG_IGNITION_*` 与
      `OUTCOME_FRAME_IDENTITY_MISMATCH`。
13. **下一任务只能根据本次真实完成范围谨慎填写**：本轮
    R6-C4-IGNITION-EVALUATOR 已完成 C4 ignition evaluator +
    typed `FrozenFrameIdentity` + 分层构造合同 + 144 个专项
    测试通过 + 全量 779 个测试通过，**没有**提前实现 C4
    deterministic driver、C5 Nether-entry evaluator/driver、
    真实 MineRL、Gradle 或模型 API；下一任务严格限定为
    `R6-C4-DETERMINISTIC-DRIVER`。

## B1 已完成

## R6-C3-DETERMINISTIC-DRIVER 已完成（FakeBackend 离线证明）

1. **新增 C3 公开 14-cell 上下文边界** [`obsidianlink/core/casting_s_c3_frame_context.py`](obsidianlink/core/casting_s_c3_frame_context.py)：
   - `build_public_c3_frame_driver_context_from_task(task)` 是整个 R6 C3 driver 家族中**唯一**允许读取 task `scenario_parameters` 的函数；它只读取 `public_task_spec.frame_plan.fixed_offsets` 与 task limits / initial inventories，**忽略** `evaluator_contract` 与任何 evaluator-only 字段；
   - 返回严格冻结的 [`PublicC3FrameDriverContext`](obsidianlink/drivers/casting_s_c3_frame.py)：workflow / family / mode / level / layout / agent_id / 14 个有序 target offsets / 不可变 initial_inventory（MappingProxyType）/ task_step_limit / task_time_limit；任何未知 family / mode / level / layout、越界或重排 target offsets、bool 充当 int、缺失或重复 cell 都 fail closed。
2. **新增 C3 deterministic driver** [`obsidianlink/drivers/casting_s_c3_frame.py`](obsidianlink/drivers/casting_s_c3_frame.py)：
   - 显式接受 immutable `PublicC3FrameDriverContext` 作为唯一 TaskInstance-shaped 输入；`run_casting_s_c3_frame_driver` 不读取 scenario_parameters / evaluator_contract / `FrozenFrameEvaluationState` 或任何 evaluator-only 字段；
   - 默认 plan：14 cell × 24 step = **336 step**，落在 640 step 任务预算内；每 cell 2 个 `use_item(water|lava)` 视为 evaluator 的 relevant action，2 个 `place_block(cobblestone)` 支撑方块是 plan 内的机械动作但不进入 per-cell evidence；
   - 动作白名单严格闭合：`equip_item` / `use_item` / `place_block` / `wait`；目标白名单严格闭合：`water_bucket` / `lava_bucket` / `cobblestone`；永不放置 `obsidian`、永不点火、永不进 Nether；
   - 预算硬上限：environment step 640、game time 600 秒、plan wait 320、plan length 640、per-action recovery 2、total recovery 32、per-step `duration_ticks` 1..40；
   - 恢复只响应 typed `RecoverableBackendError`，受 per-action 与 total 双重预算；其他异常 fail closed；
   - driver status 闭集 `completed` / `blocked` / `failed`；永不返回 `success` / `passed`（这些 verdict 仍由 `FrozenFrameEvaluator` 独立判定）；
   - 结构化事件必须带 `episode_id` / `step_id` / `agent_id` / `cell_index` / `target_offset` / `label` / `phase` / `action_type` / `target` / `relevant_action` / `attempt`；结果对象 `CastingC3FrameDriverResult` 不可变、类型严格、可序列化、暴露 `as_dict()`；
   - 新增 `per_cell_relevant_action_records`（cell_index → `((step_id, item), …)` 序列）与 `per_cell_target_offset`（cell_index → 公开 `(x, y, z)`），让 orchestrator 完全不读 evaluator truth 即可独立构造 per-cell `FrozenFrameActionEvidence`。
3. **不重命名旧 driver**：旧 `obsidianlink/drivers/casting_c3.py` 仍是 R5 / Casting-S-C2 的三 cell 连续浇筑 driver；新模块明确命名为 `casting_s_c3_frame.py`，反映 B0 taxonomy 的 C3（完整 4×5 full-ring 14 cell），与历史 ID 命名解耦。
4. **新增 95 个专项离线测试** [`tests/test_r6_casting_c3_frame_driver.py`](tests/test_r6_casting_c3_frame_driver.py)，覆盖：
   - 公开上下文严格解析与不可变性（含 family / mode / level / layout / agent / 14 cell / grid 边界 / 重复 / 重排 / bool 充当 int / 库存缺失 / 库存 bool / 库存未知 item / step 预算下限）；
   - 固定 14-cell plan 顺序、坐标与确定性（默认 336 step / 14 × 2 = 28 relevant action / 14 × 17 = 238 wait / 14 个 cell 顺序匹配 `CASTING_S_C3_FRAME_CELLS` / 与 evaluator 模块同名常量交叉校验 / 通过 `parse_macro_action` 反向接受）；
   - 动作与物品白名单（`ALLOWED_C3_FRAME_ACTION_TYPES` / `ALLOWED_C3_FRAME_TARGETS` / 默认 plan 每个 action 都用 `parse_macro_action` 通过）；
   - 默认计划长度 / wait 计数 / recovery 预算处于任务预算内；
   - 640 step / 600 秒 / wait / plan / recovery 预算失败的 fail-closed 行为；
   - 重复 `RecoverableBackendError` 有限重试、per-step 与 total budget 各自独立耗尽、metadata 透传；
   - 非 `RecoverableBackendError` 异常立即 fail closed；
   - backend 提前 terminated / truncated 的 blocked 行为；
   - 每个事件的 episode / step / agent / cell / target_offset / relevant_action 身份；
   - 相同输入的 action 序列、events 与 `as_dict()` 快照完全一致（确定性 replay）；
   - AST + 源码双门锁：driver 源文件**不** import frame evaluator、**不**调用 `set_frame_evaluation_state` / `get_frame_evaluation_state` / `clear_frame_evaluation_state` / `_frame_evaluation_state`、**不**以代码形式访问 `scenario_parameters` / `evaluator_contract`；
   - `Observation.__getattribute__` 守护禁止 driver 读取 `target_cell` / `current_block` / `success` / `outcome` / `per_cell_outcomes` / `first_failed_cell` / `completed_cells` / `interior_blocker_cells` 等隐藏字段；
   - backend spy 验证 driver 不调用 frame truth set / get / clear 表面；
   - test orchestrator 独立构造 `FrozenFrameEvaluationState` 并通过 `set_frame_evaluation_state` 注入 14-cell truth 后，`FrozenFrameEvaluator` 返回 `success`；
   - 1–13 cell 仅得 `partial_completion`（覆盖 `completed` ∈ {1..13} 的全参数子测试）；
   - 错误方块（cell 7 = cobblestone）→ `wrong_block`；内框阻挡（interior[0] = dirt）→ `interior_blocked`；缺失因果（cell 0 transition_evidence=None）→ `truth_missing`；transition 越界 4 步窗口 → `causality_missing`；身份不一致（cell 0 evidence `agent_2`）→ 状态构造即 fail closed。
5. **C1 / C2 / portal / 既有 R6 C3 frame evaluator 回归不受影响**：完整测试集 635 个全过（`python -m unittest discover -s tests -p 'test_*.py'`），`python -m obsidianlink --check` 与 `python scripts/check_environment.py` 均通过；`git diff --check` 干净。
6. **未验证限制**：
   - 真实 MineRL / Minecraft 浇筑与门框建造仍未验证；
   - 真实 backend 仍未完整接通桶动作、公开 selected item、目标方块 truth 和流体 truth；
   - C4 ignition driver、C5 Nether-entry driver、真实坐标锚定、Ruined / Adaptive / Multi-Agent 仍**未**实现；
   - 当前仅在 FakeBackend 上完成离线证明；C3 / C4 / C5 driver 与真实 backend 的接线下游仍需后续阶段验证。
7. **evaluator-only 信息隔离验证**：
   - 整个 driver 源文件（`obsidianlink/drivers/casting_s_c3_frame.py`）的 AST + 字符串双门锁确认：`set_frame_evaluation_state` / `get_frame_evaluation_state` / `clear_frame_evaluation_state` / `_frame_evaluation_state` / `FrozenFrame*` 名称均**只**出现在 docstring 的明示性引用中，从不作为代码访问；
   - `scenario_parameters` / `evaluator_contract` 在 driver 源文件中仅以 docstring 文字出现，AST `ast.Attribute` / `ast.Name` 扫描零命中；
   - test orchestrator 在 `tests/test_r6_casting_c3_frame_driver.py` 内**独立**通过 `set_frame_evaluation_state` 注入 truth；driver 表面无访问 frame truth 接口的能力。
8. **下一任务只能根据本次真实完成范围谨慎填写**：本轮 R6-C3-DETERMINISTIC-DRIVER 已完成 C3 deterministic driver 离线实现，**没有**提前实现 C4 ignition evaluator/driver、C5 Nether-entry evaluator/driver、真实 MineRL、Gradle或模型 API；下一任务严格限定为 `R6-C4-IGNITION-EVALUATOR`。

## B1 已完成

1. 新增权威 [`benchmark/catalog/tasks.json`](benchmark/catalog/tasks.json)，统一 canonical name、compatibility ID、task instance ID、workflow、taxonomy、实例/实验路径、实现状态、发布可见性和 live-run policy。
2. 新增严格 [`TaskCatalog`](obsidianlink/core/task_catalog.py) loader：frozen 类型、未知字段拒绝、family/level 匹配、唯一性、安全相对路径和 active entry 约束全部 fail closed。
3. `validate_catalog_references()` 校验所有 task/experiment 路径存在，task ID、workflow、taxonomy、compatibility name、实验引用和 live-run policy 相互一致。
4. 现有 `casting_c1_fixed` / `casting_c3_fixed` 保持原 ID、workflow 和路径；正式分类分别是 Casting-S-C1 / C2。
5. `route_a_a0_development` / `route_a_a0_phase3` 保持历史路径，但明确分类为 `calibration`、`benchmark_visible=false`、`legacy_regression`，不得混入正式 Benchmark 指标。
6. CLI `--check` 与 `scripts/check_environment.py` 从 catalog 解析 active task 和 taxonomy，不再各自硬编码分类，并验证 catalog 的所有文件引用。
7. 新增 [TASK_REGISTRY.md](docs/architecture/TASK_REGISTRY.md)，冻结 R6 起的 canonical 目录规则和历史迁移边界。
8. B1 没有移动历史文件，没有修改 evaluator、driver、backend、任务语义或依赖。

## R5 冻结合同

- workflow / task ID：保留历史兼容 ID `casting_c3_fixed`；
- taxonomy：family=`casting`、mode=`single`、level=`C2`、layout=`fixed`；
- 目标：三个冻结、有序 cell `[2,4,3]`、`[3,4,3]`、`[4,4,3]`；
- 初始资源：water_bucket=3、lava_bucket=3、cobblestone=6；
- task 预算：240 environment steps、180 秒 game time、最多 1 次 model call；
- driver：默认 72 步固定 plan、最多 96 wait、per-action recovery≤2、total recovery≤8；
- 动作白名单：`equip_item` / `use_item` / `place_block` / `wait`；
- outcome：`success` / `in_progress` / `partial_completion` / `wrong_block` / `truth_missing` / `step_budget_exceeded` / `time_budget_exceeded` / `invalid_initial_state` / `causality_missing` / `abnormal_termination`；
- 世界变化只能来自 Agent 白名单动作和原版水/熔岩方块更新。

完整任务规则见 [`casting_c3_fixed` 任务页](docs/tasks/casting/casting_c3_fixed.md)。

## R5 已完成

1. `ContinuousCastingCellTruth`、`ContinuousCastingEvaluationState`、`ContinuousCastingEvaluationResult` 和 `ContinuousCastingEvaluator` 均为不可变、类型严格、可序列化的 evaluator-only 表面。
2. Evaluator 严格要求三个冻结、有序 cell；每个 cell 的 relevant actions、水、熔岩和 transition 证据独立归属，拒绝跨 cell 重复 action step。
3. 完成非空严格有序前缀返回 `partial_completion`；中间空洞或零进展不能冒充成功。
4. `run_casting_c3_driver` 使用公共 `MacroAction`、封闭白名单、有限计划和有限预算，不调用模型、不执行代码或命令。
5. 恢复只响应类型受控的 `RecoverableBackendError`，同时受 per-action 和 total recovery 硬上限约束；其他异常 fail closed。
6. Driver 不 import continuous evaluator、不调用 truth set/get 表面、不读取 observation 上的 target/truth/outcome 字段；专项 AST、spy 和 observation guard 测试锁定隔离。
7. FakeBackend truth 注入接口校验 `episode_id`、`step_id`、`agent_id`，并在 reset/step/close 后清空陈旧状态。
8. 任务实例、CLI 和环境检查现已显式报告 Casting-S-C2 taxonomy，同时保留 `casting_c3_fixed` 兼容 ID。
9. C1/C2 任务页、README、BENCHMARK_SPEC、ROADMAP、DATASET_CARD 和 taxonomy 已与实际离线能力统一。

## R6-C3-FRAME-EVALUATOR 已完成（FakeBackend 离线证明）

1. **新增 C3 frozen-frame evaluator** [`obsidianlink/evaluation/casting_frame_evaluator.py`](obsidianlink/evaluation/casting_frame_evaluator.py)：
   - `FrozenFrameActionEvidence` / `FrozenFrameCellTruth` / `FrozenFrameInteriorCellTruth` / `FrozenFrameEvaluationState` / `FrozenFrameEvaluationResult` / `FrozenFrameEvaluator` 都是不可变、类型严格、可序列化的 evaluator-only 表面；动作证据绑定 episode / step / Agent / target cell，并只允许水桶或熔岩桶的 `use_item`；
   - 14 个 target cell 顺序与 [`casting_s_c3_fixed.json`](benchmark/instances/casting/single/casting_s_c3_fixed.json) 的 `public_task_spec.frame_plan.fixed_offsets` 严格一致；6 个 interior cell 同样在合同中冻结；
   - 14 cell 全部 obsidian、6 interior cell 全部在 `INTERIOR_ALLOWED` 内、且因果窗口合法 ⇒ `success`；
   - `partial_completion` 覆盖任意无序的 1–13 cell 完成子集，只要其余目标仍为 `air` / `water` / `lava`；缺四角的 vanilla 10-cell 形式因此判为 partial，绝不冒充 success；真正的阻挡方块仍判为 `wrong_block`；
   - 闭集 outcome：`success` / `in_progress` / `partial_completion` / `wrong_block` / `truth_missing` / `step_budget_exceeded` / `time_budget_exceeded` / `invalid_initial_state` / `causality_missing` / `abnormal_termination` / `interior_blocked`；
   - 优先级与 R3 / R5 对齐：truth_missing 和 interior_blocked 都 outrank in_progress；budget / invalid_initial_state / abnormal_termination 仍最先判定；
   - `causality_window_steps` 默认 4（与 R3 合同一致），最大 32；任何 out-of-window transition 直接 `causality_missing`；
   - 优先级由测试 `test_priority_is_stable_for_same_input` 锁定；`evaluate()` 重复调用产生完全相同的 `FrozenFrameEvaluationResult` 与 `as_dict()` 快照。
2. **task-origin / truth-grid 坐标锚定**：
   - `FrozenFrameOriginAnchor` 是纯函数、不可变、类型严格的转换器，签名只接收 `(task_origin_relative_offset)`；
   - 默认 `default_c3_anchor()` 把 task-origin 标记对齐到 grid 原点 `(0, 0, 0)`，grid 范围沿用 `obsidianlink.env.portal_spec.PORTAL_GRID_MIN/MAX` (`(-3,-1,0)`–`(3,5,6)`)；
   - 14 个公开 cell 落点 `x=0..3` / `y=0..4` / `z=1`，完全在现有 truth grid 数值范围内；越界、缺失 origin、类型错误、bool 混入、grid 边界反向（`grid_min > grid_max`）全部 fail closed；
   - 不会为了本轮合同扩展 grid；不会修改 MineRL `PortalA0EnvSpec`。
3. **FakeBackend evaluator-only truth 注入路径** [`obsidianlink/env/fake.py`](obsidianlink/env/fake.py)：
   - 新增 `_frame_evaluation_state` 槽位，与 `_casting_evaluation_state`（C1）和 `_continuous_casting_evaluation_state`（C2）严格隔离；
   - `set_frame_evaluation_state` 只接受 `casting_s_c3_fixed` workflow，并校验类型、`episode_id` 与当前 task 一致、`step_id` 与当前 backend step 一致、`agent_id` 必须在 `task.agent_ids` 内，否则 fail closed；
   - `get_frame_evaluation_state` / `clear_frame_evaluation_state` 显式可调用；
   - `reset()` / `step()` / `close()` 一律清空该槽位，杜绝跨 step 的 truth 泄漏。
4. **C1 / C2 / portal 回归不受影响**：
   - R3 `CastingEvaluationState` + `CastingEvaluator` 测试套件全绿；
   - R5 `ContinuousCastingEvaluationState` + `ContinuousCastingEvaluator` 测试套件全绿；
   - `PortalEvaluator` + `EvaluationState` 测试套件全绿；
   - FakeBackend 三个 truth 槽位独立存在；注入 C3 state 不会让 C1 / C2 slot 出现 truth。
5. **信息隔离**：
   - 整个 evaluator 源文件不 import `obsidianlink.agents` / `obsidianlink.workflows` / `obsidianlink.drivers`（AST 检查锁定）；
   - 不读取 `scenario_parameters` / `evaluator_contract` / `instruction`（AST 检查锁定）；
   - `evaluate()` 唯一参数是 `state: FrozenFrameEvaluationState`；
   - FakeBackend 生成的 `Observation`（`step_id=0` 与 `step_id=1`）都不携带任何 frame / cell / outcome / 归因 truth，公开 schema 字段集严格不变（`episode_id` / `agent_id` / `step_id` / `timestamp` / `frame` / `visible_inventory` / `messages` / `workflow_stage`）；
   - C3 frame evaluator 103 个专项测试通过；全量 539 个离线测试通过。
6. **公开 / 隐藏边界**：
   - 公开 `public_task_spec` 可以进入未来公开上下文（instruction 已显式声明 `task-origin marker` / `plane_z` / `[0,0,1]` / `width 4` / `height 5` / `14 obsidian cells`）；
   - `evaluator_contract` 仍是 policy-only（baseline_policy / required_mechanism / required_items / causality_window_steps / fail_closed_on_missing_truth），runtime truth 仍由 FakeBackend evaluator-only 状态独立验证；
   - 真实 MineRL 接入、task-origin marker 与 truth-grid origin 的世界坐标锚定仍需在 R6 之后 driver / backend 阶段验证；本轮只能证明 evaluator / FakeBackend 边界，不声称已端到端隔离未来 driver。

## 下一任务（历史指针，已被顶部当前唯一目标取代）

`R6-C5-DETERMINISTIC-DRIVER`、`R6-C5-LIVE-MINERL-BACKEND-WIRING`、`R6-C1-LIVE-MINERL-SMOKE-VALIDATION-CONTRACT-FREEZE` 与 `R6-C1-LIVE-MINERL-SMOKE-RUNNER-WIRING` 均已完成（offline）。当前唯一目标见文首。真实 C1 MineRL smoke、Gradle 或模型 API 工作需用户另行授权；不得直接进入 C5 live 或 R7。

## R6-COMPLETE-PORTAL-FRAME — CONTRACT FREEZE 已完成（保留历史）

1. **审计** [`obsidianlink/evaluation/portal.py`](obsidianlink/evaluation/portal.py) 与 [`obsidianlink/evaluation/frame_geometry.py`](obsidianlink/evaluation/frame_geometry.py)：
   - 现有 frame 几何在 `frame_geometry.detect_portal_frame` 中以 `MIN_WIDTH=4 / MAX_WIDTH=23 / MIN_HEIGHT=5 / MAX_HEIGHT=23` 冻结；两个朝向 `plane_z`（X-Y 平面，Z 恒定）与 `plane_x`（Y-Z 平面，X 恒定）；
   - 4×5 frame 的 `2W + 2H - 4 = 14` 完整外周和 `2W + 2H - 8 = 10` 必需（不含角）cell 都被 `FrameCandidate` 显式表达，4 个角 cell 在 `allow_missing_corners=True` 时是可选的；
   - `interior_allowlist = {air, nether_portal, fire}`，`dirt` / `bedrock` / `grass` / `grass_block` / `other` / `missing` 全部判为阻挡；`missing` / `other` 触发 fail-closed；
   - `PortalEvaluator` 已有 latched `portal_built_by_episode` / `valid_portal_frame` / `portal_activated`、`latched_frame_identity` / `latched_activation_offsets`、`entered_via_episode_portal_by_agent` 和 `matched_frame_identity_by_agent`；激活与 Nether entry 严格绑定到本 episode 锁存的 frame identity；
   - **`is_episode_built` 已经能证明"门框由当前 episode 建造"**：`is_geometric_valid=True` 且 baseline 中没有 required cell 是 `obsidian`；`PortalA0EnvSpec` 仍在用 14 obsidian + 1 flint_and_steel + 2 dirt 的固定受控 A0 资源。
2. **新增 3 个 R6 Casting-S 任务实例**，统一放于 canonical `benchmark/instances/casting/single/`，遵循 B0 taxonomy 与 [TASK_REGISTRY.md](docs/architecture/TASK_REGISTRY.md) 目录规则：
   - `casting_s_c3_fixed.json`：提供水、熔岩和支撑方块，通过原版 block update 浇筑公开 4×5 full-ring 方案；Minecraft 最小合法计数 10 与本实例额外要求的 14-block full ring 分开冻结；
   - `casting_s_c4_fixed.json`：继承 C3，并公开冻结唯一计分点火目标 `[1,1,1]`、`use_item` 与 `flint_and_steel`；
   - `casting_s_c5_fixed.json`：继承 C4，并公开指定 `agent_1` 进入 `minecraft:the_nether`；机器合同冻结 episode-portal、frame-identity、pre-transition-position 与 transition-step 归因要求。
3. **新增 3 个 contract-only 实验配置**于 `configs/experiments/active/`，`backend="not_implemented"`、`planner="not_implemented"`、`evaluator="not_implemented"`、`status="contract_only"`、`max_real_runs=0`。
4. **更新 [catalog](benchmark/catalog/tasks.json)**：新增 3 个 `kind=benchmark` 条目（`implementation_status="contract_only"`、`benchmark_visible=true`、`live_run_allowed=false`），保持 `active_compatibility_id="casting_c3_fixed"`（C2 仍为实际实现的 active slice，C3/C4/C5 仅为合同冻结）。
5. **新增 3 个任务页** [C3](docs/tasks/casting/casting_s_c3_fixed.md) / [C4](docs/tasks/casting/casting_s_c4_fixed.md) / [C5](docs/tasks/casting/casting_s_c5_fixed.md)，明确公开任务规则与 evaluator-only 运行时真值边界、success / failure 边界、合同复用与未实现项。
6. **更新 [TASK_REGISTRY.md](docs/architecture/TASK_REGISTRY.md)**：目录规则与路径示例已与 R6 canonical 一致；calibration `route_a_a0_*` 仍保持历史路径与 `legacy_regression` 分类。
7. **更新 [TASK_TAXONOMY.md](docs/benchmark/TASK_TAXONOMY.md)**：C3/C4/C5 里程碑表与文档级命名规则保持与 B0 一致；明确说明 R6 合同已冻结但 evaluator/driver 未实现。
8. **更新 [README.md](README.md) / [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md) / [DATASET_CARD.md](DATASET_CARD.md) / [ROADMAP.md](ROADMAP.md)**：把"当前阶段 = R6 合同冻结"和"C3/C4/C5 仅为合同、未实现"显式标注，不声称 Casting 端到端已支持。
9. **新增离线合同测试** [tests/test_r6_casting_c3c4c5_contract.py](tests/test_r6_casting_c3c4c5_contract.py)：验证 Casting 资源/机制、公开 frame plan、C4 精确点火、C5 机器归因合同、公开/隐藏命名空间和现有 Agent 源码不读取 `scenario_parameters`，并验证 `route_a_a0_*` 仍为 calibration。R6 runtime 尚未实现，因此不能声称已验证未来 driver 的运行时隔离。

## R6 已复用 / 未实现

复用：

- `frame_geometry.FrameCandidate` / `detect_portal_frame` 字段语义作为合同字段语义基准；
- `obsidianlink.env.portal_spec.PORTAL_GRID_BLOCKS` 的 truth 编码与 `FrameCandidate` 几何语义；`PortalA0EnvSpec.PORTAL_INVENTORY` 的预置黑曜石仅属于历史 calibration，不复用于正式 Casting C3–C5；
- `EvaluationState` 全部字段语义（包括 latched booleans / frame identity / activation offsets / entered-via-portal 归因 / 6 个稳定 failure types）；
- `obsidianlink.env.portal_spec.PortalTransitionObservation` 提供的 `entered_via_portal` / `from_dimension` / `to_dimension` server-side truth 作为合同承认的上游 evaluator-only 真值。

未实现（下一子任务）：

- C4 ignition evaluator（把 `use_item(flint_and_steel)` 与 `[1,1,1]` 内框 cell 的 `nether_portal` 关联到 `latched_frame_identity`）；
- C5 Nether entry evaluator（绑定 `pre_transition_position_by_agent` 与 `latched_frame_identity` 的因果链）；
- C4 / C5 deterministic driver；
- 真实 MineRL 后端接通桶动作、公开选中物品、目标方块 truth、流体 truth 与维度切换 truth；
- 真实 MineRL、Gradle 与模型 API 调用。

R6-C3-DETERMINISTIC-DRIVER 子阶段已完成：C3 deterministic driver + 严格 public context 边界 + capability gate + 95 个专项测试通过（详见上方 R6-C3-DETERMINISTIC-DRIVER 已完成节）。

已知 contract 局限（不在 R6 合同冻结阶段处理）：

- `PORTAL_GRID_MIN/MAX = (-3,-1,0)–(3,5,6)` 已覆盖本固定合同 x=`0..3`、y=`0..4`、z=`1`；真实 backend 接入时仍需验证 task-origin marker 与 truth-grid origin 的锚定关系；
- `obsidianlink/drivers/scripted_a0.py` 使用预置黑曜石和不同世界坐标，仍为 calibration，不冒充水/熔岩 Casting C3 合同；
- R6 runtime 尚未实现；当前隔离测试只能证明现有 Agent/Planner 代码不读取 `scenario_parameters`。实现 driver 时必须增加显式 public-context 构造与端到端泄漏测试。

## 已完成历史（保留）

### R1 — `casting_c1_fixed` 任务合同

固定 seed、Single-Agent、出生点、资源、单 target cell、里程碑、预算和禁止 evaluator/命令改世界规则已冻结。

### R2-CAPABILITY-MANIFEST — 后端能力清单

不可变 `BackendCapabilities`、稳定 capability IDs、FakeBackend 正反例、reset 前 fail-closed gate 和 JSON 快照已完成。真实 backend 缺关键能力时不得开始 casting episode。

### R3 — 单块 evaluator

`CastingEvaluationState`、`CastingEvaluationResult`、稳定 outcome、有限因果窗口和 evaluator-only truth 隔离已完成。

### R4-DETERMINISTIC-CASTING-DRIVER — 单块 driver

公共动作协议、封闭白名单、有限 plan/step/time/wait、确定性重放和 driver/evaluator 隔离已在 FakeBackend 完成。

### B0-BENCHMARK-SCOPE-FREEZE — 总范围冻结

三个 task family、两种 agent mode、C/R/A 能力层级、命名规范、难度维度、指标和数据证据协议已冻结。长期 scope 与 active implementation 分开。

### R5-CONTINUOUS-CASTING — 连续浇筑

`casting_c3_fixed` 的三 cell continuous evaluator、deterministic driver、per-cell 因果证据、部分完成和有限恢复已在 FakeBackend 完成，并映射为 Casting-S-C2。

### B1-TASK-CATALOG-FOUNDATION — 严格 Catalog

Task catalog、严格解析器、active entry 约束、calibration 分类与 C1/C2 可见性规则已在 catalog 落地；R6 合同冻结阶段继续把 C3/C4/C5 接入同一 catalog 体系。

## 当前限制

- R5 / R6 FakeBackend evaluator/driver 与 R6-C5 MineRL backend typed truth wiring 均已离线证明；**真实** MineRL/Minecraft 中的水、熔岩、黑曜石变化、task-origin/grid 锚定和 portal transition **仍未验证**；
- production MineRL backend 的 typed truth 入口已在 stub raw observations 上接通；这不等于 live 成功；
- `casting_c3_fixed` 是 C2 连续浇筑切片，C2 success 不等于进入 Nether；
- `casting_s_c5_fixed` 仍为 `implementation_status="contract_only"`、`live_run_allowed=false`；不得冒充 live implementation；
- Ruined、Adaptive、Multi-Agent、真实 MineRL episode 集和 Benchmark 公开指标发布均未实现；
- 当前没有正式真实 Benchmark 数据；
- 禁止真实 MineRL、Gradle 和模型调用，除非用户针对每次操作单独授权；
- 本次收尾修改开始时，独立仓库 `vendor/minerl` 已存在 dirty 工作区；外层仓库无法判断这些内容的来源。本次收尾未编辑该嵌套仓库，也未把“当前为 dirty”误写成“仓库整体无修改”；固定依赖和历史兼容 ID 未改动。

## 测试要求

Task catalog 解析/路径/分类正反例、R5 evaluator 与 driver 专项测试、R6-C3/C4/C5 evaluator/driver 专项测试、R6-C5 MineRL backend wiring 专项测试、C1 live smoke 合同冻结不变量、capability、benchmark file、CLI、R3/R4 回归、portal / frame geometry 旧测试必须保持通过。任何合同整理不得削弱严格解析、预算、因果、兼容性或信息隔离合同，也不得把 `live_run_allowed` 偷偷改为 `true`。

R6-C5-LIVE-MINERL-BACKEND-WIRING 最终离线验证（历史）：全量 **1175** 个离线测试通过；`phase="r6_c5_live_minerl_backend_wiring_done"`。`R6-C1-LIVE-MINERL-SMOKE-VALIDATION-CONTRACT-FREEZE` 将 phase 更新为 `r6_c1_live_minerl_smoke_validation_contract_freeze`（历史）。本轮 `R6-C1-LIVE-MINERL-SMOKE-RUNNER-WIRING` 将 phase 更新为 `r6_c1_live_minerl_smoke_runner_wiring_done`；安全收尾后全量 **1211** 个离线测试通过（`Ran 1211 tests in 181.479s → OK`），仍未启动真实 MineRL/Minecraft、Gradle 或模型 API。

## 下一任务

下一任务：用户单独授权的一次 **C1** 真实 MineRL smoke run（`casting_c1_fixed`）。`R6-C1-LIVE-MINERL-SMOKE-RUNNER-WIRING` 已完成 offline；不得直接进入 C5 live 或 R7。若需要 Gradle，须另行单独批准。`live_run_allowed` 仍为 `false`。
