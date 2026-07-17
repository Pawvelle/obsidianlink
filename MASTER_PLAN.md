# mc-agent 唯一主规划

**版本：** 1.0.14
**冻结日期：** 2026-07-16（Asia/Shanghai）
**状态：** 生效
**适用目录：** `/Users/joey/Desktop/Projects/mc-agent`

## 1. 规划效力

本文是本项目后续工作的唯一执行基线。所有新增计划、实现、调试、测试和优化都必须服从本文的目标、架构、阶段顺序和验收门。

允许修正错误和补充证据，但不得默默改变路线。若确需更换 MineRL/Qwen 版本、Python/JDK 主版本、动作协议、首个任务或整体架构，必须：

1. 先说明原因、影响、回滚方式；
2. 获得用户明确同意；
3. 更新本文版本与末尾“规划变更记录”；
4. 再开始实施。

## 2. 最终目标与边界

最终交付是在本机运行的闭环 Minecraft 智能体：

1. MineRL 启动 Minecraft 1.16.5，并产生第一人称 RGB 画面；
2. Qwen3-VL-2B-Instruct 在 Apple M4 的 MPS 上理解画面、当前目标和短期历史；
3. 模型只输出受约束的 JSON 宏动作；
4. 本地执行器把宏动作转换为 MineRL 的键盘、鼠标和相机动作；
5. 系统记录每一步观察、推理、动作、延迟、奖励与终止原因；
6. 首先在 `MineRLBasaltFindCave-v0` 上形成稳定基线，再逐步扩展任务。

初始版本不做以下事情：

- 不让模型生成并直接执行 Python/shell 代码；
- 不在每个 Minecraft tick 上调用大模型；
- 不先做微调或强化学习；
- 不先接入复杂长期记忆、向量数据库或多智能体；
- 不为了“装得上”而随意升级 MineRL 的 Gym/NumPy 依赖。

## 3. 已冻结的上游与本机基线

### 3.1 MineRL

- 仓库：`https://github.com/minerllabs/minerl`
- 本地路径：`vendor/minerl`
- 分支：`dev`
- 提交：`cdeae668c2f334e3c9117adf651b5a94436b45f8`
- 上游包版本：1.0.2
- Minecraft：1.16.5
- Gym：0.19.0–0.23.1，本项目固定 0.23.1
- NumPy：`<1.24`，本项目固定 1.23.5
- JDK：8
- BASALT 默认观察：`obs["pov"]`，RGB `360×640×3`
- 动作：接近真人的键盘、鼠标和相机动作；1.x 不提供旧版自动合成/冶炼动作。

### 3.2 模型

- 模型：`Qwen/Qwen3-VL-2B-Instruct`
- Hugging Face 提交：`89644892e4d85e24eaac8bacfd4f463576704203`
- 本地路径：`models/Qwen3-VL-2B-Instruct`
- 权重：`model.safetensors`，4,255,140,312 bytes
- 权重 SHA-256：`7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`
- 推理基线：Transformers 4.57.6 + PyTorch 2.13.0 + MPS

### 3.3 本机

- MacBook Air，Apple M4，10 核 CPU
- 统一内存：16 GB
- 架构：arm64
- macOS：26.5.2
- 可用磁盘（规划冻结时）：约 253 GiB
- Conda：`/opt/anaconda3`
- 环境名：`mc-agent`
- Python：3.10.20
- 环境内 OpenJDK：1.8.0_472 arm64
- Rosetta 2：可用

### 3.4 文档权威顺序

MineRL ReadTheDocs 的 `latest` 页面标题仍显示 “MineRL 0.4.0 documentation”，但同一站点又包含 1.x、Minecraft 1.16.5 和 BASALT 2022 内容，属于混合版本文档。后续判断按以下顺序取证：

1. 本项目锁定的 `vendor/minerl@cdeae668...` 源码与实际运行结果；
2. 上游 `dev` 分支 README、requirements 与安装脚本；
3. ReadTheDocs 用于核对公开 API、任务语义和通用故障说明；
4. 旧版教程示例不得直接覆盖本项目的 1.0.2 依赖和源码事实。

从官方文档确认并纳入基线的行为：

- v1.x 在安装阶段编译 Minecraft，而不是首次 `reset()` 时才编译；
- 本机是有显示器的 headed macOS，基线不使用 Linux 的 `xvfb`/VirtualGL 路线；
- `MineRLBasaltFindCave-v0` 最长 3600 step，初始物品栏为空，找到洞穴后以 `ESC=1` 主动结束，且不允许从地表直接向下挖；
- BASALT 没有可直接依赖的预定义奖励成功信号，因此首版成功判定必须保留人工复核证据。

本次校准页面：

- `https://minerl.readthedocs.io/en/latest/tutorials/index.html`
- `https://minerl.readthedocs.io/en/latest/notes/versions.html`
- `https://minerl.readthedocs.io/en/latest/environments/index.html`
- `https://minerl.readthedocs.io/en/latest/environments/basalt.html`
- `https://minerl.readthedocs.io/en/latest/notes/faq.html`

## 4. 核心技术判断

1. **兼容中心采用 Python 3.10。** 它能同时承载 MineRL 的旧 Gym/NumPy 栈与 Qwen3-VL 的新 Transformers/PyTorch 栈。
2. **MineRL 与推理逻辑同处 `mc-agent` 环境，但保持模块边界。** 旧依赖不能扩散到动作协议和智能体业务代码。
3. **模型不能逐 tick 决策。** MineRL 按约 20 tick/s 推进，而本机 2B 视觉模型一次推理远慢于单 tick；因此必须采用“低频视觉规划 + 高频确定性宏动作执行”。Qwen 推理必须与 `env.step()` 解耦：推理期间，执行线程继续按固定 tick 执行上一个安全宏动作或 no-op，不能让 Minecraft 等着模型慢慢想。
4. **第一阶段只用官方 Transformers 权重基线。** MLX、GGUF、量化或独立推理服务只有在基线正确且有性能证据后才评估。
5. **Apple Silicon 需要 MineRL 补丁。** LWJGL 从 3.3.0 起支持 Apple Silicon；本地补丁把 3.2.1 提升到 3.3.1，并为 macOS JVM 加入 `-XstartOnFirstThread`。真实启动前还要完成 `MainWindow.java` 的已知修复并重建。
6. **Gradle 构建是显式安全门。** MineRL 安装脚本会联网克隆 MCP-Reborn 并执行第三方 Gradle 代码。未获得用户明确批准，不得在沙箱外运行。

## 5. 目标架构

```text
MineRL env.step() loop（固定高频，唯一写环境的线程）
  -> obs["pov"] 640x360 RGB
  -> FrameSampler / StateBuilder
  -> bounded observation queue
  -> Qwen3-VL Planner worker（低频，MPS，不阻塞 env loop）
  -> JSON MacroAction
  -> Parser + Schema Validator + Safety Limits
  -> latest-action queue
  -> MacroExecutor（高频、确定性、可中断）
  -> MineRL action_space.no_op() + 键鼠/相机字段
  -> Episode Logger / Metrics / Watchdog
```

模块固定为：

- `env/`：MineRL 生命周期、reset/step/close、异常恢复；
- `perception/`：画面预处理、选帧、变化/卡住检测；
- `planner/`：Qwen 加载、提示词、结构化响应；
- `actions/`：动作 schema、解析、限幅、宏动作执行；
- `memory/`：短期状态摘要，不保存无界原始上下文；
- `evaluation/`：任务、指标、回放与报告；
- `cli.py`：统一运行入口。

## 6. 动作协议

模型输出必须是单个 JSON 对象，不接受 Markdown 代码块、Python 表达式或自然语言动作。首版白名单：

```json
{
  "action": "move_forward",
  "duration_ticks": 10,
  "camera": {"pitch": 0.0, "yaw": 0.0},
  "attack": false,
  "jump": false,
  "sprint": false,
  "reason": "short diagnostic text"
}
```

固定安全限制：

- `duration_ticks`: 1–40；
- 单次 pitch/yaw: -30°–30°；
- 键位只允许当前 MineRL 环境 action space 已声明字段；
- 未知字段拒绝，缺失字段补安全默认值；
- 解析失败返回 no-op，不猜测执行；
- watchdog、用户中止或 episode done 时可立即打断宏动作；
- `ESC` 默认禁止，只由终止策略显式允许。

## 7. 分阶段实施与验收门

### Phase 0 — 冻结基线与可复现环境

**状态：已完成。**

交付：

- 读取并锁定 MineRL 上游代码；
- 创建 `mc-agent` Conda 环境；
- 安装 JDK 8、MineRL Python 依赖、PyTorch/Transformers/HF CLI；
- 下载并校验 Qwen3-VL 权重；
- 写入主规划、模型锁、环境文件与执行日志；
- 在真实 M4 GPU 上验证 MPS 张量。

验收：环境检查无破损依赖；模型文件哈希一致；`Qwen3VLForConditionalGeneration` 可导入；MPS 可用。

### Phase 1 — MineRL 原生运行时

**状态：已完成并通过验收。**

顺序：

1. 获得用户对“联网克隆并执行第三方 MCP-Reborn Gradle 代码”的明确批准；
2. 记录 Gradle wrapper、MCP-Reborn tag、仓库地址和本地补丁差异，先完成静态审核；
3. 完成 Apple Silicon 补丁：LWJGL 3.3.1、`-XstartOnFirstThread`、`MainWindow.java` 图标/GLFW 错误回调修复；
4. 激活 `mc-agent`，确认 `JAVA_HOME` 和 `java -version` 指向环境内 JDK 8；
5. 用 `pip install --no-deps` 安装本地锁定源码，禁止 pip 自动重解 Gym/NumPy/PyTorch；
6. 保存完整构建日志、最终补丁和生成物哈希；
7. 运行 `import minerl` 和不启动 Minecraft 的注册/fake-env 检查；
8. 在有显示器的本机真实启动 `MineRLBasaltFindCave-v0`，不引入 `xvfb`；
9. 执行 `reset()`，断言 `obs["pov"]` 为 `uint8 (360,640,3)`，并核对实际 action-space 字段；
10. 从 `env.action_space.no_op()` 构造动作，连续执行 10 个 no-op/小幅相机动作；测试中保持 `ESC=0`，在 `finally` 中调用 `close()`；
11. 保存 Minecraft/Malmo 日志、第一张真实 POV 截图与进程清理证据。

验收门：构建完成且依赖锁未漂移；Minecraft 窗口可启动；`obs["pov"]` 是 `uint8 (360,640,3)`；实际 action space 与源码/文档可解释地一致；连续 10 步无崩溃；进程能干净退出且没有残留 Java/Minecraft 子进程。未通过不得进入 Phase 2。

### Phase 2 — Qwen 本地视觉推理基线

**状态：已完成并通过验收。**

顺序：

1. 只从锁定的本地模型路径加载；
2. 用单张合成图和一张 MineRL 截图分别推理；
3. 固定输入尺寸、`max_new_tokens` 和生成参数；
4. 记录首次加载时间、单次推理延迟、进程峰值内存；
5. 验证 MPS，不支持的算子才允许受控 CPU fallback；
6. 验证输出可稳定约束为动作 JSON。

验收门：连续 10 次推理无 OOM；动作 JSON 解析率 100%；内存峰值为 Minecraft 留出安全余量。若 16 GB 下不稳定，先减小视觉输入与上下文，再讨论量化。

### Phase 3 — 确定性环境与动作层

**状态：已完成并通过验收。**

实现 MineRL adapter、动作 schema、解析器、限幅器、宏动作执行器、watchdog 和结构化日志。先用脚本策略，不接模型。环境线程是唯一允许调用 `reset/step/close` 的线程；规划器通过容量受限的 observation/action queue 通信，旧观察和旧动作可丢弃，环境步进不得被模型推理阻塞。

验收门：脚本策略可完成 500 tick 连续运行；非法动作测试全部被拒绝或降级为 no-op；中止延迟不超过 1 tick。

### Phase 4 — 闭环视觉智能体 MVP

**状态：已完成并通过验收。**

首个任务固定为 `MineRLBasaltFindCave-v0`。Qwen 每个决策周期读取当前帧、目标、最近动作与短期摘要，输出一个宏动作；执行器在若干 tick 内执行并可中断。任务上限固定为 3600 step；只有终止策略确认“已进入洞穴”后才允许发出 `ESC=1`，且不得采用向下直挖策略。

初始控制参数：

- 决策频率：每 10–40 tick 或宏动作结束时；
- 视觉上下文：当前帧为主，最多附带少量关键帧；
- 文本历史：摘要化，设置固定 token 上限；
- temperature：先用低温/确定性配置；
- 任何解析失败：no-op + 立即重新规划。

验收门：至少完成 5 个可回放 episode；模型确实根据画面改变动作；没有代码执行通道；崩溃能自动 close 并保存日志。由于 BASALT 无预定义奖励成功信号，每个 episode 同时保存终止前关键帧、模型终止理由和人工复核字段，不能把 `ESC=1` 本身当作成功。

### Phase 5 — 稳定性与行为改进

**状态：进行中。**

依次加入：画面变化检测、原地打转检测、重复动作惩罚、恢复宏动作、短期地图/方位摘要、分层提示词。一次只改变一个变量并做 A/B 对比。六个原定变量全部验证后仍未通过卡住率总门；经用户明确批准，追加第七个单变量“受限前向探测恢复宏”，仍不得进入 Phase 6。

第七变量固定边界：A 组使用画面变化反馈、安全相机恢复与异步 decision-ack；B 组仅在最近连续 2 个观察窗口均为 `LOW`，并且当前已接受模型动作仍为语义 no-op 时，将原本的相机恢复替换为恰好 1 tick 的 `move_forward`。探测动作固定 camera=0、attack/jump/sprint=false，经同一白名单与二次限幅；每次探测后清零连续低变化计数，至少重新观察两个 `LOW` 窗口才允许再次探测。有效模型动作永不改写，MineRL step loop 不等待 Qwen。

第七变量预检门在运行前固定为：处理实际触发；所有机会均执行上述受限动作且无有效模型动作被覆盖；两组执行无效率均为 0、`ESC=0`、stale=0；B 组实际前进 tick 多于 A 组且 no-op tick 率下降；探测后的下一画面变化样本至少 50% 为 `CHANGED`；B 组低变化率不高于 A 组；总推理计算成本比不超过 1.25。单-seed 预检通过后才运行三-seed、每组 3×800-tick 正式配对 A/B，并以同一门裁决是否保留。

验收门：固定种子与预算下，卡住率和无效动作率相对 MVP 明显下降，且推理成本没有失控。

### Phase 6 — 评测体系

固定记录：成功率、episode 时长、有效位移、无效动作率、卡住次数、模型延迟 P50/P95、每决策 token、峰值内存、Minecraft 崩溃率。

评测顺序：

1. FindCave；
2. MakeWaterfall；
3. CreateVillageAnimalPen；
4. BuildVillageHouse。

每个新任务都必须先定义成功判据和动作白名单，再运行模型。

### Phase 7 — 性能优化与可选后端

只有 Phase 6 数据证明瓶颈后才进行。优先级：减少视觉 token > 减少上下文 > 缓存/批处理 > 半精度调优 > 量化 > MLX/GGUF/独立服务。任何后端更换必须保留相同动作协议和对照基线。

## 8. 测试策略

- 单元测试：动作 schema、限幅、解析失败、状态摘要、选帧；
- 集成测试：fake env、真实 env 10/500 tick、模型单图推理；
- 端到端测试：固定任务、固定预算、日志完整性；
- 回归资产：代表性画面、模型 JSON 输出、动作序列、崩溃日志；
- 每次依赖或补丁变化：先跑环境检查，再跑最小真实 Minecraft 测试。

## 9. 运行与数据规范

- 所有命令从项目根目录执行；
- 激活环境：`conda activate mc-agent`；
- 权重不提交版本库，模型身份由 `config/model.lock.json` 固定；
- 每次运行写入独立时间戳目录，包含 config、events.jsonl、metrics.json、stderr/stdout；
- 日志不得写入 Hugging Face token 或其他凭据；
- 失败必须保留最小复现命令和完整异常；
- 不以“成功 import”代替真实 `env.reset()/step()/close()` 验收。

## 10. 风险与处理顺序

1. **第三方 Gradle 代码执行：** 明确批准后才在沙箱外构建；保留提交与构建日志。
2. **Apple Silicon/LWJGL：** 使用 3.3.1 与已知 macOS 补丁；必要时才评估 Rosetta/x86_64 JDK 8 备选，不先切换。
3. **16 GB 统一内存：** Minecraft 与 2B VLM 同机竞争内存；限制图像、上下文与生成长度，串行推理，避免多模型副本。
4. **旧 Gym API：** adapter 层吸收 reset/step API 差异，业务代码不直接依赖 Gym 细节。
5. **视觉模型动作不稳定：** JSON schema、白名单、限幅、no-op fallback 与 watchdog 是不可移除的安全边界。
6. **模型延迟：** 用宏动作与低频规划解决，不尝试逐 tick VLM。

## 11. 当前唯一下一步

Phase 5 六个原定变量均已完成逐项验证，但保留基线仍没有前进，卡住率总门未通过。用户已明确批准继续下一部分；当前只实施第七变量“受限前向探测恢复宏”：A 组为画面变化反馈、安全相机恢复和 decision-ack 基线，B 组只按本规划固定的连续 2 个 `LOW` 窗口条件，将语义 no-op 的相机恢复替换为 1-tick 安全前向探测。先完成单元测试和单-seed 预检，只有预检门通过才进行三-seed 正式 A/B；严格停留在 Phase 5，Phase 6 不得开始。

## 12. 规划变更记录

- 2026-07-16 / v1.0：首次冻结。确定 MineRL 1.0.2 + Qwen3-VL-2B-Instruct + Python 3.10 + JDK 8 + MPS 路线，以及严格的阶段验收和 Gradle 安全门。
- 2026-07-16 / v1.0.1：根据官方 ReadTheDocs 补充文档权威顺序、FindCave 的 3600-step/ESC/无奖励语义、有显示器运行方式，并明确 Qwen 推理不得阻塞 MineRL 步进；总体路线与阶段顺序不变。
- 2026-07-16 / v1.0.2：用户批准第三方 Gradle 风险；Phase 1 完成 Apple Silicon 构建、安装及真实 FindCave reset/10-step/close 验收。当前唯一下一步推进至 Phase 2；总体路线与阶段顺序不变。
- 2026-07-16 / v1.0.3：Phase 2 使用锁定本地 Qwen 在 MPS/FP16 下完成合成图与真实 FindCave 首帧基准；336×336、48-token 最终配置连续 10 次 JSON 解析率 100%，并通过 2 GiB 最低可用内存门禁。当前唯一下一步推进至 Phase 3；总体路线与阶段顺序不变。
- 2026-07-16 / v1.0.4：Phase 3 完成 MineRL adapter、严格动作 schema/解析/二次限幅、容量 1 mailbox、确定性 executor、watchdog 与逐 tick 结构化日志；16 项测试及最终代码的真实 500-tick 脚本策略验收通过。当前唯一下一步推进至 Phase 4；总体路线与阶段顺序不变。
- 2026-07-16 / v1.0.5：Phase 4 为 Qwen planner worker 加入 generation-based idle/barrier，episode reset 前使旧请求失效、等待在途推理结束并清空旧 observation/decision；MineRL step loop 独立限制为 20 Hz，不等待 Qwen。21 项测试、1×800-tick 预检及最终 5×800-tick 正式验收通过：5/5 回合均有有效决策，31/31 决策被接受，stale=0、ESC=0，并产生 6 种动作签名。当前停在 Phase 4 边界，等待用户明确批准 Phase 5；总体路线与阶段顺序不变。
- 2026-07-16 / v1.0.6：用户明确批准进入 Phase 5。按既定顺序先实施画面变化检测，并固定为单变量 A/B：A 组仅旁观记录变化指标，B 组只向 planner 增加变化反馈；两组使用相同 MineRL seed 与 tick 预算。其余 Phase 5 变量保持关闭，待本变量完成验证后再依次推进；总体路线与架构不变。
- 2026-07-16 / v1.0.7：Phase 5 第一变量画面变化检测完成三-seed、每组 3×800-tick 配对 A/B；无效决策率由 28.57% 降至 27.27%，总推理计算量增加 3.07%，检测单次最大开销约 2.53 ms，6/6 回合安全与日志门通过。改进幅度尚不足以通过 Phase 5 总验收，但信号与成本受控，予以保留；当前按顺序只推进第二变量原地打转检测。
- 2026-07-17 / v1.0.8：Phase 5 第二变量原地打转检测完成 36 项测试、单-seed 预检与三-seed、每组 3×800-tick 配对 A/B。正式运行 6/6 回合安全与日志门通过，但 B 组在固定预算内零次达到连续 3 次、累计 30° 偏航的提示触发条件，偏航决策率由 58.33% 升至 66.67%，故该变量不保留；当前按顺序只推进第三变量重复动作惩罚，总体路线与架构不变。
- 2026-07-17 / v1.0.9：Phase 5 第三变量重复动作惩罚完成 42 项测试、单-seed 预检与三-seed、每组 3×800-tick 配对 A/B；正式 B 组 4 个后续决策实际暴露惩罚提示，但动作名重复率 A/B 均为 100%，收紧提示后的最终预检仍未改善且无效决策率变差，故该变量不保留。保留非阻塞 decision-ack 正确性修复，确保下一次推理不消费动作执行前的旧 observation；当前按顺序只推进第四变量恢复宏动作，总体路线与架构不变。
- 2026-07-17 / v1.0.10：Phase 5 第四变量安全恢复宏完成 45 项测试、两次单-seed 预检与三-seed、每组 3×800-tick 配对 A/B。正式 B 组 7/7 个已接受语义 no-op 均被替换为白名单内 ±20° camera-only 动作，执行层无效率由 38.46% 降至 0，低变化率由 68.42% 降至 56.14%，恢复后续有效率 60%，总推理计算量增加 4.76%，安全与日志门通过，故予以保留；当前只推进第五变量短期地图/方位摘要，总体路线与架构不变。
- 2026-07-17 / v1.0.11：Phase 5 第五变量短期地图/方位摘要完成 50 项测试、两次单-seed 预检与三-seed、每组 3×800-tick 配对 A/B。正式 B 组 11 个决策实际读取 3 样本上限的方位摘要；重访率由 80.70% 降至 75.44%，相对下降 6.52%，未达到预设 10% 门，故不纳入保留基线；低变化率和模型原始无效率有改善、成本增加 9.09%，作为正向证据保留。当前只推进第六个也是最后一个变量分层提示词，总体路线与架构不变。
- 2026-07-17 / v1.0.12：Phase 5 第六变量分层提示词完成 52 项测试与两次单-seed、每组 1×800-tick 配对预检。初版 A/B 行为完全相同；在同一变量内收紧一次后，B 组模型原始无效率由 70% 降至 40%、成本下降 0.46%，但仍为 0 次前进，低变化率由 26.32% 升至 36.84%，预设预检门未通过，因此不进行三-seed 正式运行且不保留该提示。六个既定变量至此全部验证完毕；保留基线仍未解决不前进的卡住行为，Phase 5 总验收不通过并停留于本阶段，等待用户批准新的 Phase 5 规划扩展；不得进入 Phase 6。
- 2026-07-17 / v1.0.13：修正项目迁移后遗留的适用目录；将已实现的有界方位状态迁入规划预留的 `memory/` 模块，并保留 `mc_agent.perception` 兼容导出；外层 `mc-agent` 初始化为独立本地 Git 仓库，通过 `.gitignore` 与内层 `vendor/minerl` 上游 checkout 隔离，并在 53 项测试通过后创建根提交 `cdcc8742badc46d673ee7adfd0fb6a396c4836ca`。未配置远端、未改动 MineRL 历史，也未改变 Phase 5 行为、验收口径或阶段状态。
- 2026-07-17 / v1.0.14：用户在 Phase 5 六个原定变量验收结束后明确要求继续下一部分，批准追加第七个单变量“受限前向探测恢复宏”。处理只在连续 2 个低变化窗口后用 1-tick、无冲刺/跳跃/攻击/相机动作的前进替换语义 no-op 相机恢复，并预先冻结触发、安全、变化、no-op 与成本门；不更换 MineRL、Qwen、Python/JDK、动作协议或总体架构，不进入 Phase 6。
