# ObsidianLink

ObsidianLink 是一个面向 Minecraft 长程任务的研究平台与评测基准。项目以
“制造、激活并进入地狱门”为统一目标，研究视觉语言智能体能否把配方知识转化为
稳定、可复现、可诊断的具身行为，以及两个独立角色之间的并行行动、通信和明确
分工能否带来足以抵消额外推理成本的协作收益。

项目代号为 **ObsidianLink-Bench**，计划发布的数据集称为
**ObsidianLink Dataset**。总体设计来源于
[`../ObsidianLink_项目总体规划.docx`](../ObsidianLink_项目总体规划.docx)。

> 当前唯一优先级：先让单智能体路线 A 在可控场景中稳定运行、完整记录，并被
> 自动评测器正确评分。路线 B、多角色和大规模实验必须建立在这个基础上。

## 研究问题

1. 黑曜石采集与水火浇筑两条路线对单智能体构成怎样的难度差异？
2. 长程任务的主要瓶颈来自感知、规划、探索、执行、记忆还是错误恢复？
3. 两个独立具身智能体能否提高任务成功率与完成效率？
4. 多角色收益来自并行行动，还是来自通信和明确分工？
5. 协作收益是否足以抵消额外的模型调用、Token、延迟和协调成本？

## 第一版研究边界

### 路线 A：Obsidian Mining

使用钻石镐定位并采集黑曜石，准备点火工具，选择位置，建造门框，激活地狱门，
并至少让一名角色进入下界。

### 路线 B：Lava Casting

使用水、岩浆和辅助方块逐步浇筑黑曜石门框。每次放置和方块变化都要经过视觉或
环境状态验证，失败时允许有限重试，最终激活并进入下界。

### 第一版不做

- 从空手状态完整发展到钻石镐；
- 寻找或修复自然生成的废弃传送门；
- 三个以上角色、复杂战斗、多人对抗；
- 无约束的路线发现；
- 训练新的视觉语言模型或进行大规模强化学习；
- 在评测器和可完成性没有验证前生成大量任务实例。

## 开发方法

ObsidianLink 不沿用旧项目“为一个任务不断增加专用状态机”的开发方式。后续工作
遵守以下顺序：

1. **规范先行**：先冻结任务、观察边界、动作边界、成功条件和日志格式。
2. **评测器优先**：先用人工或确定性脚本完成任务，证明环境真值评分正确。
3. **纵向切片**：先跑通 A0 的完整链路，再增加采集、浇筑和场景变化。
4. **接口迁移**：旧项目中已验证的能力迁移到新接口，不复制旧目录和主循环。
5. **假环境先测**：单元测试和 FakeBackend 通过后，才申请运行真实 MineRL。
6. **证据式验收**：代码完成不等于阶段完成；必须有真实轨迹、自动评测和人工抽查。
7. **单智能体优先**：Benchmark 定义稳定后，才进入双角色后端和协作实验。

## 系统边界

```text
TaskInstance / Workflow
          |
          v
EnvironmentBackend <-> AgentSession / Planner
          |                    |
          v                    v
   Environment truth      MacroAction JSON
          |                    |
          v                    v
   PortalEvaluator       allowlist + clamp
          |                    |
          +------> structured events <------+
```

- Agent 只能获得第一人称画面、允许公开的自身物品栏、当前语义子目标、私有记忆
  和队友主动发送的消息。
- Evaluator 可以读取用于评分的环境真值，但不得把隐藏状态反馈给 Agent。
- 模型只能提出结构化语义宏动作；模型输出不能成为 Python、Shell 或 Minecraft
  命令直接执行。
- 环境步进线程不等待本地或远程模型推理。
- 每条观察、动作、消息和日志都带有 `episode_id`、`agent_id`、`step_id`。

## 当前代码结构

```text
ObsidianLink/
├── README.md
├── ROADMAP.md
├── BENCHMARK_SPEC.md
├── DATASET_CARD.md
├── AGENTS.md
├── pyproject.toml
├── environment.yml
├── model.lock.json
├── benchmark/
│   ├── instances/
│   └── schemas/
├── configs/
│   └── experiments/
├── obsidianlink/
│   ├── actions/
│   ├── core/
│   ├── env/
│   ├── evaluation/
│   ├── logging/
│   └── workflows/
├── scripts/
├── tests/
├── runs/
└── vendor/minerl/
```

这里只建立当前阶段真正使用的接口和模块。`agents/`、`models/`、`vision/`、
`coordination/` 等目录将在对应阶段开始时创建，不预先放置空文件。

## 核心数据对象

- `TaskInstance`：任务路线、难度、角色、种子、初始资源、工作流、里程碑和预算。
- `Observation`：带身份字段的单角色观察。
- `MacroAction`：经过白名单、类型检查和数值限制的语义动作。
- `BackendStep`：一次多角色环境步进的观察、奖励、终止和公开信息。
- `EvaluationState`：仅供评测器读取的环境真值快照。
- `EvaluationResult`：里程碑、成功状态和失败类型。
- `StructuredEvent`：可回放的 JSONL 事件。

## 语义动作

第一版动作协议预留以下动作：

- `wait`
- `look`
- `move`
- `equip_item`
- `mine_target`
- `place_block`
- `use_item`
- `craft_item`

动作只是高层意图，不是固定鼠标坐标脚本。真实 MineRL 执行器将在后续阶段把它们
转换为有时间上限、可中断、可记录的低层动作。挖掘和使用物品只能在当前工作流
允许时执行；任何结构错误都会退化为一个安全的单 tick `wait`。

## 成功条件

完整成功必须同时满足：

1. 当前回合中的角色或团队建造或浇筑出有效门框；
2. 门框被激活并生成传送门方块；
3. 至少一名角色完成维度切换并进入下界。

仅在附近发现已有地狱门不算成功。评测器必须区分当前回合产生的方块变化、门框
有效性、激活事件和维度切换。

## 完整开发路线

| 阶段 | 主要工作 | 阶段出口 |
|---|---|---|
| Phase 0 Clean Core | 重建代码结构、核心类型、动作解析、FakeBackend、日志 | `python -m obsidianlink --check` 与核心测试通过 |
| Phase 1 Portal Environment | 自定义单角色任务、固定场景、资源和重置 | A0 场景可重复 reset，基础动作可执行 |
| Phase 2 Portal Evaluator | 门框、激活、维度切换和负例检测 | 人工轨迹评分准确，无已有结构误判 |
| Phase 3 A0 Vertical Slice | 材料齐全，只建造、点火和进入 | 至少一个固定配置可重复完成并回放 |
| Phase 4 Route A | 逐步加入附近黑曜石采集和资源补全 | A0-A4 有分层基线与失败归因 |
| Phase 5 Route B | 固定场景水火浇筑，逐步增加变化 | 每个浇筑阶段可验证、可恢复或明确终止 |
| Phase 6 Benchmark Alpha | 冻结模板、实例、划分、指标和运行器 | 小规模高质量实例先通过可完成性检查 |
| Phase 7 Multi-Agent Core | 同世界双角色、独立观察/动作、通信和交接 | 无模型双角色原型稳定运行 |
| Phase 8 Route A Collaboration | Single、Dual-NoComm、Dual-Chat、Dual-Workflow | 能分离并行与通信收益 |
| Phase 9 Route B Collaboration | 固定水/岩浆职责、同步与恢复 | 单/双智能体结果可公平比较 |
| Phase 10 Paper Release | 冻结配置、正式实验、数据和论文 | 所有表格、轨迹、代码和数据可复现 |

每个阶段的实现项、测试、证据、退出条件和禁止提前开展的工作见
[`ROADMAP.md`](ROADMAP.md)。任务定义、指标和数据格式见
[`BENCHMARK_SPEC.md`](BENCHMARK_SPEC.md)。

## 本地验证

Phase 0 只依赖 Python 标准库。进入固定环境后运行：

```bash
conda activate mc-agent
python -m unittest discover -s tests -v
python -m obsidianlink --check
python scripts/check_environment.py
```

Phase 1 的离线与真实环境验证：

```bash
conda activate mc-agent
python scripts/smoke_test_portal_env.py --mode spec
python scripts/smoke_test_portal_env.py --mode real --exercise-actions
python scripts/run_scripted_a0.py
```

真实检查在核心生命周期通过但 evaluator transport 不可用时返回退出码 2，并写入
`status=blocked`。这表示已经取得有效的部分能力证据，不表示地狱门任务成功。
任何 MineRL Gradle 构建仍需要用户明确批准。

## 固定技术栈

- Python 3.10.20
- OpenJDK 8.0.472
- MineRL 1.0.2 / Minecraft 1.16.5
- Gym 0.23.1 / NumPy 1.23.5
- Qwen3-VL-2B-Instruct 本地基线
- MiniMax 作为后续远程视觉模型候选

除非单独立项并提供迁移证据，不得在普通功能开发中升级以上版本或模型提交。

## 当前状态

**Phase 0：完成。Phase 1：完成。下一阶段为 Phase 2 Portal Evaluator。**

2026-07-30 的框架与环境验证结果：

- 38 个单元测试全部通过；
- `python -m obsidianlink --check` 通过；
- Python 语法编译通过；
- 3 个 JSON Schema/实例/配置文件均可解析；
- `git diff --check` 通过；
- 固定解释器 `/opt/anaconda3/envs/mc-agent/bin/python` 为 Python 3.10.20；
- 固定运行时版本为 Gym 0.23.1、NumPy 1.23.5、Torch 2.13.0 和
  Transformers 4.57.6。
- `PortalA0EnvSpec`、真实 MineRL backend 和低层动作翻译已经实现；
- 真实运行完成 14/14 tick，POV、物品栏、装备、放置、打火石使用和关闭链路通过；
- 黑曜石由 10 减到 9，MineRL 统计记录了黑曜石和打火石的各一次使用。

MineRL 1.0.2 Java `EnvServer` 桥接已经扩展并重新构建：

- 固定出生点和 25x25 确定性平整平台生效；
- evaluator grid 回传 343 个方块；
- dimension 真值正常回传；
- 放置一块黑曜石后，inventory 和 grid 同时记录变化；
- 打火石使用后 grid 记录到 fire；
- Gradle `compileJava` 和 `shadowJar` 通过。

确定性 Scripted-A0 端到端验证已经通过：

- 使用 14 块黑曜石和 2 块泥土，在生存模式完成 4x5 完整门框；
- 门框激活真值被锁存为 `portal_activated_latched=true`；
- 角色在门内等待 84 tick 后进入 `minecraft:the_nether`；
- 运行共完成 251 environment step，未提前终止；
- 运行证据位于
  [`runs/history/phase1-scripted-a0/20260730-214356/`](runs/history/phase1-scripted-a0/20260730-214356/)。

Scripted-A0 已证明环境闭环，但 Phase 2 的门框几何评测、负例和证据里程碑仍未实现，
因此不能把当前结果扩展为 VLM Agent 成功。

可复现补丁位于
[`patches/minerl/obsidianlink-envserver.patch`](patches/minerl/obsidianlink-envserver.patch)。
真实成功证据位于
[`runs/history/phase1-portal-env-smoke/20260730-203826-real/summary.json`](runs/history/phase1-portal-env-smoke/20260730-203826-real/summary.json)。

当前已经完成确定性策略下的合法门框构造和 overworld→the_nether 维度切换。
尚未完成的是 Phase 2 Portal Evaluator、VLM Agent 和双角色运行时；下一步从
Phase 2 开始。后端决策见
[`docs/decisions/0001-portal-environment-backend.md`](docs/decisions/0001-portal-environment-backend.md)。
