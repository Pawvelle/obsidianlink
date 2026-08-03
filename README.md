# ObsidianLink

ObsidianLink 是一个面向 Minecraft 长程任务的研究平台与评测基准。项目以
“制造、激活并进入地狱门”为统一目标，研究视觉语言智能体能否把配方知识转化为
稳定、可复现、可诊断的具身行为，以及两个独立角色之间的并行行动、通信和明确
分工能否带来足以抵消额外推理成本的协作收益。

项目代号为 **ObsidianLink-Bench**，计划发布的数据集称为
**ObsidianLink Dataset**。总体设计已经整理为仓库内可直接查看的
[开发路线图](ROADMAP.md)、[基准规范](BENCHMARK_SPEC.md)和
[数据集说明](DATASET_CARD.md)。

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

**Phase 0：完成。Phase 1：完成。Phase 2：完成。Phase 3：完成。**

- Phase 0 / Phase 1 验证结果保持不变（72 个测试套件中的 Phase 0/1 部分
  全部通过、Scripted-A0 真实 MineRL 闭环已存档）。
- Phase 2 离线几何、契约和负例测试已扩展（见
  [`docs/decisions/0002-portal-frame-rules.md`](docs/decisions/0002-portal-frame-rules.md)、
  [`obsidianlink/evaluation/frame_geometry.py`](obsidianlink/evaluation/frame_geometry.py)、
  [`tests/test_frame_geometry.py`](tests/test_frame_geometry.py)）。
- 历史 Scripted-A0 运行仍因缺少每步 grid 与精确 transition evidence
  正确报告 `status=insufficient_evidence`；它没有被事后升级为 Phase 2
  证据。
- Phase 2 审计（2026-07-30 Round 3）已落地的 7 项修复：
  - 外部生成完整门框（带或不带 nether_portal、dimension flip）→ terminal
    failure 升级为 `frame_not_built_by_episode`（`external_structure_candidate_count`
    由 backend 计算并暴露给 evaluator）；
  - accepted `place_block(obsidian)` credit 仅在当前 post-step observation
    有效；fresh delta 数与 credit 数不完全相等时全部 fail closed 为
    external，credit 不得跨 observation 存活，且 external cell 永不重新归因；
    该边界由
    `regression_single_place_block_is_one_obsidian` 守护；
  - 仅有门框邻近位置不能证明进入因果；必须同时有精确 latched-frame
    identity 的显式 bridge transition evidence。缺失证据为 unknown，
    外部/其他 portal 证据为 false，两者均 `success=False`；
  - `atSpawn` grid offset 通过 reset 时实际 world position 锚定，避免把
    y=64 的门框错误映射到 y=0；
  - 三块分布在不同边的 obsidian 不再触发 `build_site_selected`；
  - 缺失 `latched_timestamps` 键的 `EvaluationState` 构造直接 raise；
  - 全 missing grid → `has_missing_truth=True`；
  - `EvaluationState.milestone_events()` 严格只读 `latched_timestamps`，
    emission-time 不再回填 `time.time()`，重复 emission 返回完全相同的
    timestamp。
- 2026-07-31 离线验证：121 / 121 tests 通过；经用户单次批准执行
  `./gradlew compileJava`，但该进程从默认 PATH 取得 Java 25，ForgeGradle
  在配置阶段因要求 Java 8 而终止，未进入 Java 编译，也未启动 Minecraft。
  随后使用 `/opt/anaconda3/envs/mc-agent` 的固定 OpenJDK 8.0.472
  重新执行，`compileJava` 成功（5 个任务：4 executed、1 up-to-date）。
  两次构建均未启动 Minecraft。
- Phase 2 bridge source 已加入可重建 patch：输出 atSpawn grid world origin、
  dimension，以及由 server-side portal change guard 产生的 typed
  `portal_transition` source block。该 Java 改动已通过 `compileJava` 和
  `shadowJar`。
- Phase 2 canonical 真实证据位于
  [`runs/phase2-scripted-a0/20260731-173302/`](runs/phase2-scripted-a0/20260731-173302/)：
  251 条动作记录全部带身份字段，14 块 obsidian 全部归因、0 external，
  门框 step 148、激活 step 162、typed transition/Nether entry step 251；
  正式 `PortalEvaluator` 报告 `success=true`、
  `entered_via_episode_portal=true`、无 blocking condition，并有显式终止
  和人工复核。

Phase 2 退出条件已满足。Phase 3 的 Scripted-A0 已完成正式 evaluator
闭环；本地 Qwen 已能在 MPS 完成隔离预加载，但与 MineRL 同时常驻的实际运行仍在
底层提前终止，尚缺一份受控、可评分的 VLM 运行记录（参见 `ROADMAP.md`）。
MiniMax-M3 可作为不依赖本地模型常驻的远程视觉 planner；密钥仅从
`MINIMAX_API_KEY` 读取，首个实验固定为单帧、一次调用和标准服务档。

- 2026-08-03 受控 VLM A0 真实运行已落地（**两个** run，旧 run 仅作
  诊断产物，close-out 证据在干净提交下的新 run）：
  - 修复了
    [`obsidianlink/agents/local_qwen.py`](obsidianlink/agents/local_qwen.py)
    里的 PIL 衍生 numpy 负 stride bug（PIL 视图 → Qwen image processor
    报 `ValueError: tensors with negative strides`），新加
    `np.ascontiguousarray` 归一化与 4 个
    [`tests/test_local_qwen.py`](tests/test_local_qwen.py) 回归测试；
  - 单元测试套件 140 / 140 通过，`python -m obsidianlink --check` 通过；
  - **Pre-instrumentation run**
    [`runs/phase3-vlm-a0/20260803-215738/`](../runs/phase3-vlm-a0/20260803-215738/)：
    320 step 全部跑完，Qwen 决策到达 owner 时已超过
    `max-decision-age-steps=160` 窗口被丢弃为 stale，所有 action 为
    `wait`，evaluator 输出 `failure_type=frame_never_valid`、
    `last_successful_milestone=task_reset`；脚本语义下 `status=blocked`、
    exit code 2；无残留进程。该 run 的 `code_version.json` 仍指向
    修复前提交 `40e84e8`、未记录 dirty 标记、未记录推理延迟，
    **不作为 Phase 3 close-out 证据**。
  - **Phase 3 close-out run（干净提交）**
    [`runs/phase3-vlm-a0/20260803-222729/`](../runs/phase3-vlm-a0/20260803-222729/)：
    - 来自本地提交
      `280ec920df963522355335137a57f0e2083c6fcd`，未 push；
      `code_version.json.working_tree_dirty=false`、
      `summary.json.reproducible_from_clean_commit=true`；
    - 同样 320 step 全部跑完，所有 action 为 `wait`；
    - 实测 Qwen3-VL-2B 在 MPS 上的**推理延迟**
      `responder_latency_seconds = 41.47003112499806`
      （owner 端时间戳
      `started_at_monotonic = 34646.38780725` →
      `completed_at_monotonic = 34687.857838375`）；
    - 决策 `source_step = 0`、`return_step = 164`、
      `decision_age_steps = 164 > 160`、
      `drop_reason = "stale_age_exceeded"`；模型输出为空文本
      `decision_error = "Expecting value: line 1 column 1 (char 0)"`；
    - `formal_evaluation.failure_type = "frame_never_valid"`、
      `last_successful_milestone = "task_reset"`、
      `success = false`、exit code 2；无残留进程；
    - 同目录 `manual_review.md` 接受此 run 为 Phase 3 VLM close-out
      证据。
  - 真实诊断：在 `0.25s` step 节奏 + capacity-1 mailbox 之下，
    Qwen3-VL-2B 在 MPS 上的单次推理延迟 ≫ episode 预算
    （41.47s ≫ 160 × 0.25s = 40s），单帧单调用契约无法影响 episode
    走向；这是受控运行的设计边界，不是新 bug。
- 本轮 VLM runner 增量（`scripts/run_vlm_a0.py`）：
  - 每次 owner poll 写出 `model_requests.jsonl`，每条含
    `source_step` / `return_step` / `decision_age_steps` /
    `max_decision_age_steps` / `drop_reason` /
    `responder_started_at_monotonic` /
    `responder_completed_at_monotonic` / `responder_latency_seconds` /
    `responder_device` 字段；budget 结束后再做 final flush 避免
    残留决策被静默丢弃；
  - `code_version.json` 新增 `working_tree_dirty` + `dirty_paths`；
    `summary.json` 同步 `working_tree_dirty` /
    `working_tree_dirty_paths` / `code_commit` /
    `reproducible_from_clean_commit`，dirty 时一律不声称完全可复现；
  - `summary.json.local_qwen_request` 镜像
    `remote_request` 槽位，方便把单次推理 timing 嵌进 summary。
- **Phase 3 退出条件全部满足**（详见
  [`ROADMAP.md`](ROADMAP.md) 2026-08-03 close-out 条目）：Scripted-A0
  canonical 闭环、VLM 产生可诊断里程碑失败、Agent 不读 evaluator-only
  状态、失败有明确类型和最后有效里程碑、可从干净提交复现。本轮未授权
  启动 Phase 4。
