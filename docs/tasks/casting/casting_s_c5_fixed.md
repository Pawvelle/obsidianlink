# `casting_s_c5_fixed` 浇筑、点火与进入 Nether 任务（C5 合同冻结）

`casting_s_c5_fixed` 是 **Casting-S-C5 / fixed** 端到端任务合同。它继承 C3 的水/熔岩 full-ring 浇筑和 C4 的固定点火规则，再要求指定 Agent 通过同一个本 episode 门框进入 Nether。C5 evaluator + 347-step C5 deterministic driver 均已在 FakeBackend 上离线实现；真实 MineRL 接入、真实维度切换证据采集、正式 benchmark episode 与 live runs 仍未实现。

## 固定合同

- family / mode / level / layout：`casting` / `single` / `C5` / `fixed`
- 指定 Agent：`agent_1`（公开任务规则）
- 初始资源：`water_bucket=14`、`lava_bucket=14`、`cobblestone=28`、`flint_and_steel=1`
- 公开源维度：`minecraft:overworld`
- 公开目标维度：`minecraft:the_nether`
- 预算：800 environment steps、720 秒 game time、最多 1 次 model call
- 状态：`contract_only`

公开门框方案与 C3 相同；公开点火方案与 C4 相同，且只承认 `[1,1,1]`。

## 机器可读 Nether entry 合同

`public_task_spec.nether_entry_goal` 冻结 designated Agent、源维度和目标维度。`evaluator_contract.nether_entry_attribution` 冻结：

- `require_entered_via_episode_portal=true`；
- `require_matched_frame_identity=true`；
- `require_pre_transition_position=true`；
- `require_transition_step=true`；
- 归因未知 outcome：`nether_entry_portal_unknown`；
- 明确经其他方式进入 outcome：`nether_entry_not_via_episode_portal`；
- 任一必要 truth 缺失时 fail closed。

## C5 evaluator 合同

C5 success 必须同时满足：

1. C3 与 C4 success 的全部条件成立；
2. `agent_1` 在本 episode 内从 `minecraft:overworld` 切换到 `minecraft:the_nether`；
3. `entered_via_episode_portal_by_agent["agent_1"]` 明确为 `True`；
4. `matched_frame_identity_by_agent["agent_1"]` 与 episode-built、episode-activated 的 `latched_frame_identity` 一致；
5. `pre_transition_position_by_agent` 与 `transition_step_by_agent` 完整且顺序合法；
6. 外部门框、预存门、命令、直接修改 dimension、文本声明或缺失归因均不成功。

C5 是 Casting family 的端到端层级；但在真实 MineRL episode 证据可用前，contract-only 或 FakeBackend 结果不得作为公开 Benchmark 成绩。

## 信息隔离

指定 Agent、目标维度和“必须经本 episode 门进入”的规则是公开目标。以下运行时值属于 evaluator-only：

- `agents_in_nether`、`first_nether_step_by_agent`；
- `pre_transition_position_by_agent`、`transition_step_by_agent`；
- `entered_via_episode_portal_by_agent`、`matched_frame_identity_by_agent`；
- `latched_frame_identity` 和全部评分结果。

Agent 自身正常 observation 中的公开 dimension 状态可以保留，但 evaluator 的门框匹配与进入归因结果不得进入 Agent 输入。

## 当前实现状态

已冻结 C5 合同、catalog、配置和文档，并实现 `FrozenNetherEntryEvaluator`、typed `NetherEntryEvidence`、C4 success 复验、transition/agent/dimension/portal/frame-identity 归因、FakeBackend 独立 evaluator-only truth 槽、347-step C5 deterministic driver（14-cell × 24 step C3 cast sub-plan + 4 step C4 ignition sub-plan + 7 step C5 portal approach / entry sub-plan：4 次接近移动、1 次对齐移动、1 次穿门移动、1 次 settle）和离线测试。

C5 driver 仅是 FakeBackend 离线证明，**没有**真实 MineRL/Minecraft Nether entry 证据，没有正式 benchmark episode；Agent 初始朝向、portal 平面与固定前进轨迹在真实环境中的对齐也尚未验证。下一阶段工程任务只能基于实际完成范围谨慎填写。
