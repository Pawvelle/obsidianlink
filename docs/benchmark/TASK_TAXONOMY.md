# ObsidianLink 任务分类与命名规范

本文档冻结 ObsidianLink Benchmark 的任务分类、Agent 模式、能力层级和文档级命名规则。评分、信息边界和证据要求以 [Benchmark 总规范](../../BENCHMARK_SPEC.md) 为准。

## Benchmark 核心目标

ObsidianLink 的端到端目标是：至少一名由任务合同指定的 Agent，必须通过当前 episode 内由团队完成建造、修复或激活的下界传送门进入 Nether。任务实例必须在 reset 时冻结可进入 Nether 的指定 Agent 集合，evaluator 只依据世界真值和结构化事件判定。

以下情况都不是端到端完整成功：

- 仅找到普通或废弃传送门；
- 只完成门框但未激活；
- 只激活但没有指定 Agent 进入 Nether；
- 使用预先完整且与当前 episode 无关的传送门；
- evaluator 或 Minecraft 命令直接修改世界；
- 仅由模型文本声称成功；
- driver 正常退出，但 evaluator 未验证成功。

C1 等低层能力任务仍可产生其局部 evaluator 的 `success` outcome；它表示该冻结切片完成，不等同于上述端到端成功。汇总结果必须区分“能力层级完成”和“进入 Nether”。

## 三个任务族

### Casting

Casting 从流体、支撑结构和方块操作开始，要求 Agent 通过原版水、熔岩和方块更新机制构造传送门。

| Level | 能力里程碑 |
|---|---|
| C1 | 生成一块黑曜石 |
| C2 | 连续生成多个目标黑曜石 |
| C3 | 完成有效门框 |
| C4 | 完成点火 |
| C5 | 通过本 episode 完成的传送门进入 Nether |

当前历史任务 `casting_c1_fixed` 的分类为：

- family：`casting`
- mode：`single`（命名缩写 `s`）
- level：`C1`
- layout：`fixed`
- status：`implemented_offline`
- 文档级兼容名称：`casting_s_c1_fixed`

`casting_c1_fixed` 是稳定历史兼容 ID，不重命名任务文件、workflow、实例或实验配置。

当前 R5 任务 `casting_c3_fixed` 的分类为：

- family：`casting`
- mode：`single`（命名缩写 `s`）
- level：`C2`
- layout：`fixed`
- status：`implemented_offline`
- 文档级兼容名称：`casting_s_c2_fixed`

`casting_c3_fixed` 中的 `c3` 是早期“三个 cell”的数量型命名，不代表本 taxonomy 的能力层级 C3。为保持任务、workflow、重放和测试兼容性，现有 ID 不重命名；新任务不得继续使用这种数量型歧义命名。

### Ruined Portal

Ruined Portal 评测探索、结构理解、材料利用、修复和使用废弃传送门。

| Level | 能力里程碑 |
|---|---|
| R1 | 识别废弃传送门 |
| R2 | 定位并到达结构 |
| R3 | 判断门框缺失部分 |
| R4 | 收集或使用资源完成修复 |
| R5 | 点火并进入 Nether |

“找到废弃传送门”只是中间里程碑，不是完整任务成功。

### Adaptive

Adaptive 评测 Agent 能否在 Casting 与 Ruined Portal 路线之间判断可行性、选择策略，并在失败或条件变化后重规划。

| Level | 能力里程碑 |
|---|---|
| A1 | 两条候选路线中只有一条可行 |
| A2 | 两条路线都可行，但成本明显不同 |
| A3 | 局部信息不足，需要先探索再选择 |
| A4 | 初始路线失败后切换 |
| A5 | 动态条件变化下重新规划 |

每个 Adaptive episode 至少记录：可行路线集合、Agent 最初选择的路线、是否发生路线切换、切换原因、最终完成路线、参考成本或参考路线，以及最终是否进入 Nether。可行路线集合和参考成本是 evaluator-only 真值，不能进入 Agent observation、prompt 或 memory。

## Agent 模式

### Single-Agent

一个 Agent 独立完成观察、规划、执行和恢复。

### Multi-Agent

多个 Agent 在同一任务族中合作。Multi-Agent 不是第四个任务族，而是与任务族正交的执行模式。

| Task Family | Single-Agent | Multi-Agent |
|---|---|---|
| Casting | Casting-S | Casting-M |
| Ruined Portal | Ruined-S | Ruined-M |
| Adaptive Routing | Adaptive-S | Adaptive-M |

Multi-Agent 任务评测角色分工、私有观察、消息传递、共享任务状态、重复劳动、空闲时间、冲突动作、团队完成时间，以及是否至少一名指定 Agent 进入 Nether。每个 Agent 的 observation 和 memory 相互隔离；共享内容只能通过任务合同允许的消息或共享状态协议传播。Evaluator truth 对所有 Agent 都不可见。

## 文档级命名规则

新任务使用：

```text
<family>_<mode>_<level>_<layout>
```

允许值：

- `family`：`casting` / `ruined` / `adaptive`
- `mode`：`s` / `m`
- `level`：`c1`–`c5` / `r1`–`r5` / `a1`–`a5`
- `layout`：`fixed` / `randomized` / `hidden` / `challenge`

示例：

- `casting_s_c1_fixed`
- `casting_s_c5_randomized`
- `ruined_s_r1_fixed`
- `ruined_s_r5_randomized`
- `adaptive_s_a1_fixed`
- `adaptive_s_a4_randomized`
- `casting_m_c3_fixed`
- `ruined_m_r5_randomized`
- `adaptive_m_a5_challenge`

名称描述任务分类，不替代唯一的 `task_instance_id`。同一分类下的不同 seed、资源配置或场景变体必须使用不同实例 ID。旧 ID `casting_c1_fixed` 和 `casting_c3_fixed` 保留为兼容 ID；当前实例通过 `scenario_parameters` 显式声明 taxonomy，不为整理命名而破坏 schema 或重放。

任务的 canonical name、兼容 ID、实际路径和 Benchmark/calibration 可见性统一登记在 [`benchmark/catalog/tasks.json`](../../benchmark/catalog/tasks.json)，规则见 [TASK_REGISTRY.md](../architecture/TASK_REGISTRY.md)。新任务必须先进入 catalog，不能只靠 README 或文件名声明分类。

## 难度维度

实例不得只依赖含义模糊的单一 `difficulty` 数字。未来任务合同应显式记录可组合参数：

- `layout_variation`
- `resource_availability`
- `exploration_distance`
- `observation_uncertainty`
- `execution_noise`
- `recovery_required`
- `route_ambiguity`
- `communication_constraint`
- `agent_count`

发布时可以用 D1–D4 表示便于阅读的难度档案：D1 为固定、充足资源、低噪声；D2 引入单一受控变化；D3 组合探索、不确定性或恢复；D4 为隐藏布局、路线歧义或受限通信等挑战组合。D1–D4 只是参数档案标签，每个实例仍必须显式保存上述具体参数，不能用标签替代参数。

## 范围与兼容性

该 taxonomy 冻结长期设计，不表示所有矩阵单元已经实现。当前 active implementation 包含 Casting-S-C1 的 `casting_c1_fixed` 和 Casting-S-C2 的 `casting_c3_fixed`，两者都仅在 FakeBackend 离线验证。Casting C3–C5、Ruined、Adaptive、Multi-Agent 和端到端进入 Nether 均属于后续计划。
