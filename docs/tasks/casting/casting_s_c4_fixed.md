# `casting_s_c4_fixed` 门框浇筑与点火任务（C4 合同冻结）

`casting_s_c4_fixed` 是 **Casting-S-C4 / fixed** 合同。它完整继承 C3 的水/熔岩 full-ring 浇筑要求，再增加一次可归因的固定位置点火。本阶段不实现 evaluator、driver 或真实 MineRL 接入。

## 固定合同

- family / mode / level / layout：`casting` / `single` / `C4` / `fixed`
- Agent：`agent_1`
- 初始资源：`water_bucket=14`、`lava_bucket=14`、`cobblestone=28`、`flint_and_steel=1`
- 预算：700 environment steps、640 秒 game time、最多 1 次 model call
- 状态：`contract_only`

门框公开方案与 C3 完全相同：`plane_z`、`min_corner=[0,0,1]`、4×5、Minecraft 最小合法计数 10、本实例要求含四角的 14-block full ring。

## Agent-visible 点火方案

`scenario_parameters.public_task_spec.ignition_plan` 冻结：

| 字段 | 值 |
|---|---|
| `required` | `true` |
| `action` | `use_item` |
| `item` | `flint_and_steel` |
| `target_offset` | `[1,1,1]` |
| `target_policy` | `exact` |

`[1,1,1]` 是公开的唯一计分点火目标。其它内部 cell 上点火即使激活门框，也不满足此固定实例的 C4 合同。该规则同时出现在公开 instruction 中，不属于 evaluator-only truth。

## C4 evaluator 合同

C4 success 必须同时满足：

1. C3 的 14-cell 水/熔岩浇筑与归因条件全部成立；
2. `agent_1` 手持 `flint_and_steel`，在公开目标 `[1,1,1]` 执行合法 `use_item`；
3. 本 episode 建造的门框内部随后出现 `nether_portal`；
4. 激活证据位于上述动作后的 4-step 因果窗口内；
5. 激活绑定同一个 `latched_frame_identity`，不能来自外部门框、命令或 evaluator/driver 写世界；
6. 缺少动作、激活、frame identity 或 step 证据时 fail closed。

这些规则已写入 `evaluator_contract.activation_attribution`，包括 `require_exact_public_target=true` 和 `require_latched_frame_identity_match=true`。

## 信息隔离

点火动作、物品和目标位置属于公开规则。隐藏内容仅包括实际 `nether_portal` 方块变化、`first_activation_step`、`latched_activation_offsets`、`latched_frame_identity` 及 evaluator 判定。

## 当前实现状态

已冻结 C4 合同、catalog、配置、文档和离线一致性测试。未实现 ignition evaluator、deterministic driver、真实 MineRL 或模型接入。

下一子任务：`R6-C4-IGNITION-EVALUATOR`，但必须在 C3 evaluator 完成后进行。
