# ObsidianLink v2.0 Task Taxonomy

v2 只有一个 task family：`nether_portal_construction`。Diagnostic、End-to-End、Generalization & Recovery 是评测维度；Single-Agent 与 Multi-Agent 是正交 execution modes。

## Stable dimensions

每个未来 task identity 必须声明：

- `family`: `nether_portal_construction`
- `suite`: `diagnostic` / `end_to_end` / `generalization_recovery`
- `mode`: `single` / `multi`
- `level`: Diagnostic 使用 D1–D6；End-to-End 与其 Generalization/Recovery 变体使用 L1–L4
- `layout`: `controlled` / `randomized` / `hidden` / `challenge`

当前只冻结 taxonomy，不批量创建空 task instances。实例 ID、seed、variation profile、预算和 evaluator version 必须在对应 P3/P4/P5 阶段单独冻结。

## Diagnostic levels

| Level | Name |
|---|---|
| D1 | Perception |
| D2 | Grounding |
| D3 | Manipulation |
| D4 | Planning |
| D5 | State Tracking |
| D6 | Recovery |

Diagnostic outcome 不能冒充 Nether entry。

## End-to-End levels

| Level | Name | Final success |
|---|---|---|
| L1 | Controlled Construction | attributed Nether entry |
| L2 | Resource Interaction | attributed Nether entry |
| L3 | Resource Acquisition | attributed Nether entry |
| L4 | Open-World Construction | attributed Nether entry |

level 间只通过初始条件、资源依赖、距离和环境变化增加难度。单块、多块、frame 或 ignition 是 milestone，不是 L-level。

## Generalization & Recovery

`generalization_recovery` 任务保留其基础 L1–L4 level，并添加冻结 variation/recovery profile。未来 profile 覆盖 seed、spawn、yaw、资源距离/分布、terrain、obstacles，以及 no-world-effect、placement failure、resource missing、path blocked、subgoal infeasible、state mismatch、casting error。

## Execution modes

- `single`: 一个 Agent 的 observation/inventory/memory；
- `multi`: 多个私有 Agent state，只能通过显式消息/shared protocol 交换信息。

Multi-Agent 研究条件包括 `fixed_role`、`autonomous_role_assignment`、Natural Multi-Agent 和 Compute-Matched Multi-Agent。

## Naming guidance

未来 canonical name 建议：

```text
<family>_<mode>_<suite>_<level>_<layout>
```

例如 `nether_portal_construction_s_end_to_end_l1_controlled`。Canonical name 不替代唯一 `task_instance_id`。

## Legacy IDs

`casting_c1_fixed`、`casting_c3_fixed`、`casting_s_c3_fixed`、`casting_s_c4_fixed`、`casting_s_c5_fixed` 及 Route A0 保留兼容 ID，但只属于 v1 legacy/calibration/regression。它们的 C1–C5 是历史 capability progression，不是 v2 L1–L4。

Roadmap phase P0–P8、Environment Validation E0–E12、Diagnostic D1–D6 与 End-to-End L1–L4 是互不重叠的命名空间。

历史三-family taxonomy 归档于 [TASK_TAXONOMY_V1.md](../legacy/v1/TASK_TAXONOMY_V1.md)。权威可见性见 [Task Registry](../architecture/TASK_REGISTRY.md)。
