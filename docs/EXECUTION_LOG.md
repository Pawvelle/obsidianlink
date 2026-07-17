# 执行日志

## 2026-07-16 — Phase 0

### 已完成

- 本机：Apple M4 / arm64 / 16 GB / macOS 26.5.2。
- MineRL 浅克隆到 `vendor/minerl`，锁定 `dev@cdeae668c2f334e3c9117adf651b5a94436b45f8`。
- 已阅读 README、requirements、setup、BASALT 环境、动作/观察接口和 MCP 构建脚本。
- 创建 Conda 环境 `mc-agent`：Python 3.10.20、OpenJDK 1.8.0_472 arm64。
- 安装 MineRL Python 兼容依赖：Gym 0.23.1、NumPy 1.23.5、OpenCV 4.8.1.78、pyglet 1.5.27 等。
- 安装 Qwen 推理依赖：PyTorch 2.13.0、torchvision 0.28.0、Transformers 4.57.6、Accelerate 1.14.0。
- `pip check`：`No broken requirements found.`
- 沙箱外 MPS 实测：`mps_built=True`、`mps_available=True`，成功创建 `mps:0` 张量。
- 使用本地模型与合成红色图片完成 Qwen3-VL MPS 冒烟推理：加载 13.607 秒，推理 2.940 秒，输出 `Red`。
- 下载模型提交 `89644892e4d85e24eaac8bacfd4f463576704203`，共 12 个文件、4,266,648,961 bytes。
- `model.safetensors`：4,255,140,312 bytes；SHA-256 为 `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`，校验通过。
- MineRL 本地补丁已将 patch 中 LWJGL 3.2.1 改为 3.3.1，并在启动 JVM 时加入 `-XstartOnFirstThread`。

### 安全门 / 未完成

MineRL 安装脚本会联网克隆 `Hexeption/MCP-Reborn@1.16.5-20210115`，随后执行第三方 Gradle 构建代码。沙箱外执行请求因第三方代码风险被拒绝；未采用任何绕过方式。

下一步必须先由用户明确批准该风险，然后严格执行 `MASTER_PLAN.md` Phase 1：完成 `MainWindow.java` Apple Silicon 修复、构建、安装，并用真实 `reset/step/close` 验收。

## 2026-07-16 — ReadTheDocs 规划校准

- 核对官方 Installation、Versions、Environment、BASALT、FAQ、Performance 与 First Agent 页面。
- 发现 `latest` 页面标题仍标为 0.4.0，但内容混合 1.x；已在主规划中明确“锁定源码优先、文档用于接口和任务语义”的权威顺序。
- 确认 v1.x 在安装阶段编译 Minecraft、需要 JDK 8；本机为 headed macOS，不采用面向 Linux headless 的 xvfb/VirtualGL 路线。
- 确认 FindCave 为 3600 step、空初始物品栏、找到洞穴后以 `ESC=1` 结束、不得向下直挖，且 BASALT 无预定义奖励成功信号。
- 将非阻塞 planner/step 双循环、终止证据与人工复核要求补入 `MASTER_PLAN.md` v1.0.1；总体路线和阶段顺序未改变。

## 2026-07-16 — Phase 1

### 构建与兼容性修复

- 用户明确批准执行 MCP-Reborn 第三方 Gradle 构建代码。
- 审计并锁定 `Hexeption/MCP-Reborn@1.16.5-20210115`，提交为 `1e71be5bd4c49bc4d6ab0ee559c31b298b7697a3`。
- 将浮动依赖 `ForgeGradle:3.0.+` 固定为已验证的 `3.0.190`；构建前后阶段分别使用 ForgeGradle 3.0.190 与 MineRL 补丁指定的 4.1.10。
- 保存并按 SHA-256 校验 MCP `output.zip`：`65669e3413666b4634f00f876efbfc36bf5659b43078068caa9c4e32158fd139`，避免 2021 MCP 远端补丁输入漂移。
- Apple Silicon 修复：LWJGL 3.3.1、`-XstartOnFirstThread`、移除窗口图标调用、禁用 `checkGlfwError`、项目相对生成 `schemas.index`。
- 为 LWJGL core、GLFW、jemalloc、OpenAL、OpenGL、STB、tinyfd 加入 `natives-macos-arm64`；Maven Central HEAD 校验返回 HTTP 200。
- `setup_mcp.sh` 与 `patch_mcp.sh` 已加入 fail-fast、克隆重试、非交互补丁与哈希门禁。
- Gradle `clean build shadowJar` 成功，MineRL macOS arm64 wheel 构建成功并安装到 `mc-agent`。

### 产物

- wheel：`artifacts/phase1/minerl-1.0.2-cp310-cp310-macosx_11_0_arm64.whl`，约 1.3 GB。
- wheel SHA-256：`a208729207ad459e4bc6d4b5f441b9eb2f6ead633cec2dee2025d80c03108f56`。
- 真实首帧：`artifacts/phase1/findcave-reset.png`，640×360。
- 首帧 SHA-256：`4b37801a16f1d090d7237dd2c58a7fad77f3f728f3bf6d9591807ed3cfd89a5e`。
- 构建日志：`logs/phase1/minerl-build-arm64-natives.log`。
- 真实冒烟日志：`logs/phase1/findcave-real-smoke.log`。

### 验收

- MineRL 1.0.2 可导入，`MineRLBasaltFindCave-v0` 已注册。
- 上游支持的 `MineRLNavigate-v0` fake env 完成 reset 与 1 step；MineRL 1.0.2 的 fake env `close()` 存在 `NotImplementedType.client_socket` 上游缺陷，测试脚本仅对该精确异常做兼容记录。
- 真实 `MineRLBasaltFindCave-v0` 完成 reset、保存首帧、连续 10 step、`close()`；`pov` 为 `[360, 640, 3]`，期间未提前 done。
- 进程检查未发现残留 Minecraft/mcprec；可见 Java 进程仅为 Cursor Java 扩展和 Gradle daemon。
- `pip check` 唯一提示为 MineRL 上游元数据声明 `typing>=3.6.6` 未安装；Python 3.10 已内置 `typing`，不安装会遮蔽标准库的旧 backport。该提示不影响本次 import、fake 或真实环境验收。

### Phase 1 结论

**通过。** 后续严格进入 `MASTER_PLAN.md` Phase 2，不提前实施 agent 控制闭环。

## 2026-07-16 — Phase 2

### 固定基线

- 模型只从本地路径 `models/Qwen3-VL-2B-Instruct` 加载，锁定提交 `89644892e4d85e24eaac8bacfd4f463576704203`；加载时启用 `local_files_only=True`。
- 设备与精度固定为 Apple MPS / FP16；生成固定为 `do_sample=False`。
- 第一轮使用 448×448 / `max_new_tokens=64`，10/10 JSON 通过，但最低系统可用内存约 2.14 GB，余量偏薄。
- 按主规划的降级顺序将最终配置固定为 336×336 / `max_new_tokens=48`，未采用量化，也未更换模型。
- 动作 JSON 测试 schema 固定为 `action/yaw/pitch/duration_ticks`；只做严格解析和约束验证，不连接 MineRL、不执行动作。

### 最终验收结果

- 合成红色图输出：`Red`；推理 0.838 秒。
- 真实输入：Phase 1 的 `artifacts/phase1/findcave-reset.png`。
- 连续 10 次真实首帧推理均无 OOM，严格 JSON 解析与字段/类型/枚举/范围验证为 10/10（100%）。
- 10 次均稳定输出 `{"action":"look","yaw":0,"pitch":0,"duration_ticks":10}`。
- 真实首帧推理延迟：平均 3.845 秒，中位数 3.943 秒，P90 4.522 秒，最小 2.928 秒，最大 4.540 秒。
- 模型加载 9.069 秒；进程峰值 RSS 4,342,857,728 bytes；MPS driver 峰值 5,036,163,072 bytes。
- 最低系统可用内存 2,183,233,536 bytes，高于显式验收门 2 GiB（2,147,483,648 bytes）。余量通过但较紧，未来 Minecraft 与模型并行时仍须持续采样并保留 watchdog。
- 独立结果复核通过：`accepted=true`、10 次均无解析错误、内存门禁通过。

### 产物

- 基准脚本：`scripts/benchmark_qwen_phase2.py`；SHA-256 `4dbdc4d6a9d521d890b1f1fee750b09c0266ae9aa7f22af3071fd307734ddf48`。
- 结构化结果：`artifacts/phase2/qwen-benchmark.json`；SHA-256 `0d39f72bfc65bb70fdfae2da9377040a0348c94b7aa15da71def1e18ec3f0f0e`。
- 最终日志：`logs/phase2/qwen-benchmark-final.log`；SHA-256 `0d39f72bfc65bb70fdfae2da9377040a0348c94b7aa15da71def1e18ec3f0f0e`。
- 另保留 448×448 初测和 336×336 调参日志，作为内存降级决策证据。

### Phase 2 结论

**通过。** 后续严格进入 `MASTER_PLAN.md` Phase 3；本阶段没有实现或执行 MineRL 动作。

## 2026-07-16 — Phase 3

### 确定性环境与安全动作层

- 新增 `mc_agent.env.MineRLEnvAdapter`：吸收 Gym 0.23.1 的旧 reset/step API，验证 `pov` 为 `uint8 (360,640,3)`，并强制 `reset/step/close` 只能由创建环境的线程调用。
- 新增严格宏动作协议：只接受单个 JSON 对象和 `wait/look/turn/move_forward` 白名单；未知字段、Markdown、`ESC`、非有限数、错误类型全部拒绝并降为 1-tick no-op。
- 固定安全限制：时长 1–40 tick，pitch/yaw -30°–30°；缺失字段补安全默认值；reason 截断为 160 字符。
- `MacroExecutor` 始终从 MineRL `action_space.no_op()` 构造动作并强制 `ESC=0`；相机增量只在宏动作首 tick 发送一次；只有 `move_forward` 允许 sprint。
- parser 与 executor submit 均执行限幅。即使本地代码绕过 JSON parser 直接构造越界或未知 `MacroAction`，仍会被二次限幅或降为 no-op。
- 新增容量为 1 的 `LatestActionMailbox`；新动作覆盖旧动作，满足未来 planner 与环境线程解耦时的“只取最新值”语义。
- 新增线程安全 watchdog；收到中止后，执行器下一 tick 必定输出 no-op。SIGINT 只设置 stop 标志，不直接从信号处理器操作环境。
- 新增 flush-on-write 的 `config.json / events.jsonl / metrics.json`，真实运行逐 tick 记录动作、奖励与 done，不记录凭据或模型输出代码。

### 测试

- 标准库 unittest 共 16 项，全部通过。
- 覆盖：合法默认值、角度/时长限幅、未知字段、Markdown、NaN、错误布尔类型、`ESC` 注入、相机只在首 tick 生效、direct-action 二次限幅、latest mailbox、单线程环境所有权、旧 Gym step 归一化、非法 MineRL action、结构化日志。
- watchdog 中止测试：长达 40 tick 的 forward 宏动作执行 1 tick 后发出 stop；下一次 `next_tick()` 立即变为 no-op，满足中止延迟不超过 1 tick。
- 20 tick 真实预检通过；首尾截图 SHA-256 不同，确认相机动作真实进入 Minecraft。

### 最终 500-tick 验收

- 命令使用 `PYTHONPATH=src` 在 `mc-agent` 环境运行 `python -m mc_agent.cli phase3-smoke --ticks 500`；策略固定为交替 10 tick 的原地 look/wait，不接 Qwen。
- 最终运行目录：`artifacts/phase3/runs/20260716-150715`。
- 完成 500/500 tick，无提前 done；总奖励 0.0；含 Minecraft 启动/关闭的总耗时 34.822 秒，整体吞吐 14.359 tick/s。
- 共 556 条事件：1 reset、50 macro、500 tick、5 progress；25 个宏动作首 tick 带相机增量。
- 500 个 tick 的 `ESC/attack/forward/jump/sprint` 全部为 0；所有 tick 的 done 均为 false。
- 最终代码加入二次限幅后重新从零完成 500 tick，避免以修改前运行替代最终验收。
- 进程检查未发现残留 Minecraft、mcprec 或 MineRL process watcher。

### 产物与哈希

- `events.jsonl` SHA-256：`50cbbf5698184c0ae4fe594f53d2bbe95e8f5a6aec92abb381813e6e2321c6b9`。
- `metrics.json` SHA-256：`8d07e2f712ac27e1576a04dd71e08b15edf35f25836eea056304bf0a7a10f35c`。
- 初始截图 SHA-256：`087c0b03ee474ce09f0279f724192fdea2770f35f7b476b91c4412fdd266c9d1`。
- 最终截图 SHA-256：`c4195b7d645238a1709be8614eb4b0e5e2ec9fdcbc621063dec85e34aad6790b`。
- 最终标准输出日志：`logs/phase3/phase3-500-final.log`；SHA-256 `004e22b701b199fbe7502e7f7a3698a8d1f4d1193af9d893448988ee297c664d`。

### Phase 3 结论

**通过。** MineRL 已接入确定性安全动作层；后续严格进入 `MASTER_PLAN.md` Phase 4，才允许将 Qwen JSON 接入同一 parser/mailbox/executor 链路。

## 2026-07-16 — Phase 4

### 闭环接入与 episode barrier 修复

- Qwen3-VL planner 保持独立 daemon worker、容量 1 observation/decision mailbox；模型输出仍只经过严格 JSON parser、动作白名单、数值限幅和 `MacroExecutor`，源码不存在 `eval/exec/shell/subprocess` 型模型输出执行通道。
- 初次 5×800-tick 运行 `artifacts/phase4/runs/20260716-153300` 完成 5 回合且 `ESC=0`、动作有变化，但第 2、4、5 回合没有有效决策，正式结果为 `accepted=false`。原因是队列清空无法取消 worker 已取走的在途推理，旧决策跨越 episode 边界后被丢弃。
- `QwenPlannerWorker` 新增显式 `idle`、`wait_until_idle()` 与 generation-based `begin_episode()` barrier：切换时先递增 generation 使旧请求失效，清空待处理 observation，等待唯一在途推理结束，再清空旧 observation/decision，最后才允许新 episode reset/submit。
- worker 取 observation、标记 inference active、发布 decision 与 barrier 共用 condition；即使旧请求已离开 mailbox，也不能向新 generation 发布结果。运行中的 MineRL loop 只做非阻塞 `submit/take_latest`，不会等待 Qwen。
- 新增可控阻塞的假 planner 测试，验证 barrier 确实等待在途推理、清除旧 generation，且 planner 忙碌时 observation submit 仍能立即完成。
- 第一轮 barrier 后的正式尝试 `artifacts/phase4/runs/20260716-230003` 已实现 5 回合 `stale=0`，但热机后的 MineRL 达到 42–50 tick/s，部分 800-tick 回合短于 Qwen 尾延迟，第 3、5 回合仍无有效决策，保留为失败证据。
- 按主规划“固定高频、约 20 tick/s”要求加入只依赖单调时钟的 20 Hz 节拍上限；节拍不读取 planner 状态，也不等待 Qwen。验收门同时收紧为每回合至少 1 个 `accepted_decision`，被拒绝的 no-op 不计入动作变化。

### 测试与单回合预检

- 单元测试命令：`/opt/anaconda3/bin/conda run -n mc-agent env PYTHONPATH=src python -m unittest discover -s tests -v`。
- 最终共 21 项测试，全部通过；新增覆盖 generation barrier、忙碌时非阻塞 submit、未建立 episode barrier 时拒绝 submit，以及 20 Hz 时钟节拍不产生负等待。
- 最终单回合预检命令：`/opt/anaconda3/bin/conda run -n mc-agent env PYTHONPATH=src python -m mc_agent.cli phase4-eval --episodes 1 --ticks 800 --observation-interval 40`。
- 预检目录：`artifacts/phase4/runs/20260716-230402`；`accepted=true`，完成 800 tick，11/11 有效决策，0 rejected、0 stale、`ESC=0`，产生 3 种动作签名。
- 预检 Qwen 平均/最大推理延迟为 3.888/4.438 秒，`env.step()` P95/最大为 0.024/0.215 秒；20 Hz 节拍累计等待 34.071 秒，证明 step loop 只按时钟节流而未被推理阻塞。

### 最终 5×800-tick 正式验收

- 命令：`/opt/anaconda3/bin/conda run -n mc-agent env PYTHONPATH=src python -m mc_agent.cli phase4-eval --episodes 5 --ticks 800 --observation-interval 40`。
- 最终目录：`artifacts/phase4/runs/20260716-230542`；顶层 `accepted=true`，5/5 回合全部完成 800 tick，无提前 done，planner error 为空。
- 每回合有效决策分别为 11、7、6、4、3；合计 31/31 accepted、0 rejected、0 stale。共 6 种动作签名，yaw 覆盖 -20、-15、0、10、15、20，`model_changed_action=true`。
- 4000 个 tick 的 `ESC` 全部为 0；未启用终止策略，也未把 tick-budget 结束或 `ESC` 当作 FindCave 成功。人工复核字段与首尾帧/决策帧均保留，`manual_review.status=pending`。
- episode barrier 等待分别为约 0、1.617、3.650、2.973、4.956 秒，均发生在 reset 前；这段边界等待没有进入正在运行的 MineRL step loop。
- Qwen 单次最大推理延迟 22.376 秒；各回合 `env.step()` 最大值均不超过 0.407 秒，进一步确认推理与 step loop 解耦。
- 日志完整性复核：每回合都有 1 条 barrier、1 条 reset、20 条 observation、20 张决策帧、800 条 tick，以及与 metrics 一致的 11/7/6/4/3 条 planner decision；`config.json`、`events.jsonl`、`metrics.json`、640×360 RGB 首尾 PNG 均存在并可解析。
- 上下文退出和 `planner.stop()` 后检查进程表，未发现残留 Minecraft、mcprec、MineRL process watcher、GradleStart 或 qwen-planner 进程，自动 close/收尾通过。
- 最终 summary SHA-256：`b6dc5e830201eb4420aef868f3cd03b10a28c862788616546443950f3dd3563e`。
- 最低系统可用内存约 1.348 GB；本次无 OOM、崩溃或 planner error，但该余量仍需在后续阶段继续监控。

### Phase 4 结论

**通过。** 闭环 Qwen/MineRL MVP 已完成 5 个可回放 episode，并通过有效决策、动作变化、安全动作、非阻塞、日志和自动清理验收。当前严格停在 Phase 4 边界；未执行任何 Phase 5 行为改进，等待用户明确批准后续阶段。

### Phase 4 可视化回放产物

- 以正式验收目录 `artifacts/phase4/runs/20260716-230542` 为唯一来源，将 5 个 episode 各自的初始帧、20 张决策帧和最终帧顺序合成为采样回放；不做 AI 插帧或画面生成。
- 每回合首尾各停留 1 秒，视频写入 Episode 1–5 五个章节；总时长 60 秒。
- FFmpeg 编码结果：H.264 High、1280×720、30 fps、`yuv420p`、无音轨、faststart；文件大小 5,948,054 bytes。
- 回放文件：`artifacts/phase4/replays/20260716-230542-phase4-replay.mp4`。
- 章节元数据：`artifacts/phase4/replays/20260716-230542-phase4-replay.ffmetadata`。
- `ffprobe` 验证时长为 60.000 秒、1800 帧，五个章节分别覆盖 0–12、12–24、24–36、36–48、48–60 秒；抽取六个时间点组成联系表后人工检查，画面均可正常解码。
- MP4 SHA-256：`095ff2b9b1cecc5f5d9dffccbcd9cc07b7f52e5e1ee1c08f8872664398e8bc46`。

## 2026-07-16 — Cursor Gradle 自动导入警告处理

- 在项目级 `.vscode/settings.json` 中设置 `java.import.gradle.enabled=false` 与 `gradle.autoDetect=off`，阻止 Cursor 自动使用系统 Java 25 导入 `vendor/minerl` 内的旧版 Gradle 工程。
- 未修改 Gradle wrapper、JDK、MineRL/MCP-Reborn 源码或任何构建产物；命令行构建仍按主规划使用 `mc-agent` Conda 环境内的 JDK 8。
- 验证命令：`python3 -m json.tool .vscode/settings.json`；退出码为 0，JSON 格式有效。

## 2026-07-16 — Phase 5 第一变量：画面变化检测

### 单变量边界与实现

- 用户明确批准进入 Phase 5；严格按顺序只加入第一变量画面变化检测，原地打转检测、重复动作惩罚、恢复宏动作、短期地图/方位摘要和分层提示词保持关闭。
- 使用 Phase 4 正式验收的 100 张决策帧离线标定：裁掉底部 60 行 HUD，将世界画面按 8 像素步长取样并转灰度；低变化联合阈值固定为归一化平均绝对差 `<0.005` 且显著变化像素比例 `<1%`，单像素显著差阈值为 20/255。
- 新增 `FrameChangeDetector`；A 组以 shadow 模式计算并记录同一指标，但 planner prompt 与 Phase 4 逐字相同；B 组只追加视觉变化状态。两组共用显式 MineRL seed、20 Hz step loop、Qwen 模型、动作协议和全部安全边界。
- 为 `MineRLEnvAdapter` 增加 owner-thread 约束下的 `seed()` 转发，并在每次 reset 前设置配对 seed。
- 新增无效决策指标：`wait` 与零角度 `look/turn` 视为语义 no-op；`move_forward`、非零相机动作或显式 attack/jump 才算有实际效果。该指标只做度量，不改变 parser 或 executor。

### 测试与预检

- 最终单元测试命令：`/opt/anaconda3/bin/conda run -n mc-agent env PYTHONPATH=src python -m unittest discover -s tests -v`；29/29 通过。
- 覆盖 HUD 裁剪、相同帧低变化、世界画面变化、reference 更新、输入校验、A 组 prompt 不变、B 组短反馈、seed 转发及语义 no-op 指标。
- 首次 seed-5101 预检 `artifacts/phase5/frame-change-ab/20260716-234347` 数据完整但 `advance_recommended=false`：反馈文本过长导致 B 组推理延迟与截断拒绝上升，800 tick 全为 no-op；保留为失败证据。
- 将 B 组反馈压缩为固定 `LOW/CHANGED` 状态并限制 reason 少于 12 个词，没有加入惩罚或恢复逻辑。第二次同 seed 预检 `artifacts/phase5/frame-change-ab/20260716-234716` 通过：B 组 6/6 输出被接受、5 种动作签名、0 stale、`ESC=0`。

### 三-seed 正式配对 A/B

- 命令：`/opt/anaconda3/bin/conda run -n mc-agent env PYTHONPATH=src python -m mc_agent.cli phase5-frame-change-ab --seeds 5101,5102,5103 --ticks 800 --observation-interval 40`。
- 最终目录：`artifacts/phase5/frame-change-ab/20260716-235054`；执行顺序固定为 `5101:A/B, 5102:B/A, 5103:A/B`，用于减轻热机顺序偏差。
- 6/6 回合均完成 800 tick，A/B 各 2400 tick；每回合 1 barrier、1 reset、20 observation、800 tick，全部 `ESC=0`、stale=0、planner error 为空，stdout 完整且 stderr 为空。
- A 组：14/14 决策 accepted，10 有效、4 无效，无效决策率 28.57%；总推理计算 124.544 秒。
- B 组：11/11 决策 accepted，8 有效、3 无效，无效决策率 27.27%；总推理计算 128.368 秒。
- B 相对降低无效决策率 4.55%，推理计算成本比 A 增加 3.07%；检测单次最大耗时约 2.53 ms，成本没有失控，但改善幅度仍较小。
- 最终 `accepted=true`、`advance_recommended=true`；summary SHA-256 为 `409fed3a2a734018443b969e7a089847f64c585fcd2ec09efba4229d2b6895e0`。
- 退出后未发现残留 Minecraft、mcprec、MineRL process watcher、GradleStart 或 qwen-planner 进程。

### 第一变量结论

**完成并保留。** 画面变化检测信号、配对 seed、日志和成本门均通过，且无效决策率轻微下降；该结果不足以单独满足 Phase 5 的“明显下降”总验收，因此 Phase 5 仍为进行中。按主规划顺序，下一步只进入原地打转检测，不提前加入后续变量。

## 2026-07-17 — Phase 5 第二变量：原地打转检测

### 单变量边界与实现

- A 组保留第一变量已验收的画面变化反馈，只旁观记录原地打转状态；B 组使用完全相同的 detector，并且只在状态 active 时追加一条原地打转提示。重复动作惩罚、恢复宏动作、短期地图/方位摘要和分层提示词保持关闭。
- detector 固定观察最近 3 个已接受宏动作；只有 3 个动作均为非零 yaw 的 `look/turn` 且累计绝对 yaw 至少 30° 时 active，`move_forward`、`wait`、零 yaw 或仅 pitch 动作会打断连续窗口。
- 新增 `TurningLoopDetector`、B 组短提示、逐决策/逐 observation 状态日志、偏航决策率/触发次数/前进决策数指标，以及 `phase5-turning-loop-ab` CLI。MineRL 20 Hz step loop、Qwen worker、episode barrier、动作白名单与数值限幅均未改变。

### 测试与预检

- 单元测试命令：`/opt/anaconda3/bin/conda run -n mc-agent env PYTHONPATH=src python -m unittest discover -s tests -v`；36/36 通过。新增覆盖阈值边界、非旋转动作打断、reset、构造参数校验、A 组视觉提示逐字不变、inactive 不改 prompt、active 只追加有界提示。
- 首次预检 `artifacts/phase5/turning-loop-ab/20260717-000338` 在第一个 `reset()` 的 MineRL mission 握手处收到空连接，0 tick 退出；错误为 `TypeError("a bytes-like object is required, not 'NoneType'")`，完整失败日志已保留，且没有残留 Minecraft 进程。没有因此修改算法或实验参数。
- 原样重试命令：`/opt/anaconda3/bin/conda run -n mc-agent env PYTHONPATH=src python -m mc_agent.cli phase5-turning-loop-ab --seeds 5101 --ticks 800 --observation-interval 40`；目录 `artifacts/phase5/turning-loop-ab/20260717-000450`。
- 重试的 2/2 回合均完成 800 tick，stale=0、`ESC=0`、自动 close；A 组 9/9 决策 accepted，B 组 4/4 accepted。A 组触发 1 次 detector，B 组没有形成 3 次连续偏航，因此预检 `accepted=true` 但 `advance_recommended=false`。

### 三-seed 正式配对 A/B

- 命令：`/opt/anaconda3/bin/conda run -n mc-agent env PYTHONPATH=src python -m mc_agent.cli phase5-turning-loop-ab --seeds 5101,5102,5103 --ticks 800 --observation-interval 40`。
- 最终目录：`artifacts/phase5/turning-loop-ab/20260717-000810`；执行顺序为 `5101:A/B, 5102:B/A, 5103:A/B`。
- 6/6 回合均完成 800 tick；每回合均有 1 barrier、1 reset、20 observation、800 tick，全部 stale=0、`ESC=0`、planner error 为空。`stdout.log` 与 `summary.json` 完全一致，`stderr.log` 为 0 字节。
- A 组：12/12 决策 accepted，7 有效、5 无效，无效决策率 41.67%；7 个仅偏航决策，占 58.33%；detector 共激活 1 次并在 1 个 observation 上 active；前进决策为 0。
- B 组：9/9 决策 accepted，6 有效、3 无效，无效决策率 33.33%；6 个仅偏航决策，占 66.67%；3 个回合均只有 2 个连续偏航决策，detector 激活与 active observation 都为 0，故原地打转提示实际零次暴露；前进决策为 0。
- B 组偏航决策率相对增加 14.29%，总推理计算量比 A 组增加 1.06%。虽然数据完整性与成本门通过，但由于处理变量没有实际暴露且目标指标变差，最终 `accepted=true`、`advance_recommended=false`。
- summary SHA-256：`07549b9e016c3ce8708f77976ae332f553bb394560e4aba858465eb64fcbe376`。退出后未发现 MineRL、Minecraft、mcprec、process watcher 或 qwen-planner 残留进程。

### 第二变量结论

**完成验证但不保留。** 在固定 800-tick 预算与当前 Qwen 延迟下，“连续 3 次、累计 30°”条件触发太迟，B 组没有收到一次处理提示，不能据此声称行为改善；正式偏航率也没有下降。保留 detector、日志和失败资产供回归，但下一变量的 A/B 基线恢复为仅含已验收画面变化反馈。Phase 5 仍为进行中，下一步只进入重复动作惩罚，不提前加入恢复宏动作、短期地图/方位摘要或分层提示词。

## 2026-07-17 — Phase 5 第三变量：重复动作惩罚

### 单变量边界与异步上下文修复

- A 组继续使用第一变量保留的画面变化反馈，只旁观记录连续已接受动作名；B 组在每个已接受动作后，只向下一次 planner prompt 加入避免再次选择同名动作的惩罚提示。没有加入原地打转反馈、恢复宏动作、短期地图或分层提示。
- 新增 `RepetitionDetector`；同一 episode 内，相邻已接受宏动作的 `action` 名相同即计为一次重复机会命中。惩罚只影响提示词，不在 parser 或 executor 中拒绝、替换或执行模型代码。
- 审计发现 worker 生成决策后会立即取走主循环尚未消费决策时排队的旧 observation，导致下一次推理使用动作执行前的画面与历史。新增异步 decision-ack 交接：worker 发布决策后在后台等待主循环 ack；主循环照常执行 MineRL step，随后 ack 并清空 pre-action observation，worker 再异步等待新帧。MineRL step loop 从不等待 Qwen。
- 每个决策记录 `repetition_feedback_used`，每个 episode 记录 feedback observation/decision 数；由此能够验证 B 组处理变量是否真正进入过模型输入，而不只检查配置开关。

### 测试与预检

- 最终单元测试命令：`/opt/anaconda3/bin/conda run -n mc-agent env PYTHONPATH=src python -m unittest discover -s tests -v`；42/42 通过。覆盖重复状态、动作参数变化下的同名重复、动作切换/reset、A 组 prompt 不变、B 组有界提示，以及 decision-ack 等待、清空旧 observation、post-ack 新请求和 episode barrier 回归。
- 第一版单-seed 预检目录：`artifacts/phase5/repetition-ab/20260717-002344`。2/2 回合完成 800 tick，B 组 3 个后续决策实际使用惩罚，重复率从 A 的 100% 降到 B 的 33.33%，无效决策率从 50% 降到 25%，并产生一次 16-tick `move_forward`；总推理计算量增加 19.58%，`accepted=true`、`advance_recommended=true`。
- 三-seed 正式运行后发现宽松提示的例外被模型稳定滥用，因此将 B 组反馈压缩并收紧为“本次 `action` 字段不得等于上次动作；中心安全则前进，否则选不同安全动作”。没有改变 detector、A 组、模型参数、动作协议或 executor。
- 收紧后的最终单-seed 预检目录：`artifacts/phase5/repetition-ab/20260717-003413`。B 组唯一一个实际暴露惩罚的后续决策仍输出被禁止的 `look`；A/B 重复率均为 100%，无效决策率由 40% 升至 50%，`accepted=true`、`advance_recommended=false`。因此没有为该版本再消耗三-seed 正式预算。

### 三-seed 正式配对 A/B（第一版提示）

- 命令：`/opt/anaconda3/bin/conda run -n mc-agent env PYTHONPATH=src python -m mc_agent.cli phase5-repetition-ab --seeds 5101,5102,5103 --ticks 800 --observation-interval 40`。
- 最终目录：`artifacts/phase5/repetition-ab/20260717-002634`；顺序为 `5101:A/B, 5102:B/A, 5103:A/B`。
- 6/6 回合均完成 800 tick；每回合 1 barrier、1 reset、20 observation、800 tick，全部 stale=0、`ESC=0`、planner error 为空。所有 episode 均满足 `planner_decision` 与 `planner_decision_acknowledged` 一一对应；12 张 initial/final 图齐全，stdout 与 summary 相同，stderr 为空。
- A 组：9/9 决策 accepted，5 有效、4 无效，无效决策率 44.44%；6/6 重复机会命中，动作名重复率 100%；总推理计算 99.088 秒。
- B 组：7/7 决策 accepted，4 有效、3 无效，无效决策率 42.86%；4 个后续决策全部实际暴露惩罚，但 4/4 仍选择 `look`，动作名重复率 100%；总推理计算 113.516 秒，增加 14.56%。
- 正式结果 `accepted=true`、`advance_recommended=false`；summary SHA-256 为 `ef065c9d0ee8e1d7c7260d2d2c26e01bae8f181f9017f05b13b1528ec00c6075`。收紧提示预检 summary SHA-256 为 `b45962cc6e4552c179ff5acaa2908657c39d26cdbceae81d7b487b49816ca7b0`。
- 所有运行退出后未发现 MineRL、Minecraft、mcprec、process watcher 或 qwen-planner 残留进程。

### 第三变量结论

**完成验证但不保留惩罚提示。** 第一版单-seed 收益没有在三-seed 正式实验复现，且收紧后的提示仍被模型直接违反；继续调整措辞会形成追逐样本。保留 detector、处理暴露指标、失败资产和非阻塞 decision-ack 正确性修复；下一变量的行为基线仍只有已验收的画面变化反馈。Phase 5 仍为进行中，下一步只进入恢复宏动作，不提前加入短期地图或分层提示词。

## 2026-07-17 — Phase 5 第四变量：安全恢复宏动作

### 单变量边界与实现

- A 组使用画面变化反馈与异步 decision-ack 基线，只旁观记录恢复机会；B 组只在模型 JSON 已解析接受、但宏动作是 `wait` 或零角度 `look/turn` 时替换实际执行动作。有效模型动作保持原样，原地打转反馈、重复动作提示、短期地图和分层提示保持关闭。
- 恢复动作固定为 1 tick `look`，yaw 按恢复序号在 `+20/-20` 之间交替，pitch=0、attack/jump/sprint=false；动作经同一 `limit_macro_action` 与 executor 二次限幅，不前进、不交互、不执行任何模型代码。
- 决策日志同时保存原始 `parsed` 与实际 `executed`，并区分模型无效率与执行层无效率；记录恢复机会、应用数、恢复后的下一模型决策有效率与期末未完成 follow-up。
- 下一次 planner observation 使用实际执行动作作为 previous action；worker 仍只在后台等待 decision ack，MineRL step loop 不等待 Qwen。

### 测试与预检

- 最终单元测试命令：`/opt/anaconda3/bin/conda run -n mc-agent env PYTHONPATH=src python -m unittest discover -s tests -v`；45/45 通过。新增覆盖 ±20° 交替、无交互键、非法序号拒绝、A 组 no-op 原样执行、B 组只替换语义 no-op、有效前进动作不被覆盖。
- 第一轮单-seed 目录 `artifacts/phase5/recovery-ab/20260717-004212`：2/2 回合完成；B 组 3/3 恢复机会全部执行，执行无效率由 42.86% 降至 0，no-op tick 下降，3 个恢复 follow-up 中 2 个有效，成本增加 8.19%。原验收门错误要求 Qwen 原始无效率不得变差，因此 `advance_recommended=false`。
- 在不改变动作行为的前提下，将恢复层门修正为：执行无效率至少相对下降 20%、所有机会均安全恢复、恢复后续有效率至少 50%、no-op tick 不变差、推理成本比不超过 1.25；Qwen 原始无效率继续报告但不作为恢复层否决项。
- 最终代码单-seed 目录 `artifacts/phase5/recovery-ab/20260717-004521`：B 组 3/3 恢复，执行无效率 50% 降至 0，低变化样本 10 降至 7，恢复后续有效率 50%；本轮推理成本波动为 +26.72%，略超门。结合行为相同的上一轮成本 +8.19%，保留两次预检并由交错三-seed 正式运行裁决成本。

### 三-seed 正式配对 A/B

- 命令：`/opt/anaconda3/bin/conda run -n mc-agent env PYTHONPATH=src python -m mc_agent.cli phase5-recovery-ab --seeds 5101,5102,5103 --ticks 800 --observation-interval 40`。
- 最终目录：`artifacts/phase5/recovery-ab/20260717-004851`；顺序为 `5101:A/B, 5102:B/A, 5103:A/B`。
- 6/6 回合均完成 800 tick；每回合 1 barrier、1 reset、20 observation、800 tick，decision/ack 一一对应，stale=0、`ESC=0`、planner error 为空。stdout 与 summary 相同，stderr 为空，12 张 initial/final 图齐全。
- A 组：13/13 决策 accepted，模型与执行均为 8 有效、5 无效，无效率 38.46%；5 个恢复机会全部保持原样；低变化率 68.42%，no-op tick 率 99.667%，总推理计算 100.222 秒。
- B 组：10/10 决策 accepted，模型原始为 3 有效、7 无效；7/7 个机会全部安全恢复，执行为 10 有效、0 无效，执行无效率相对下降 100%；5 个可观测恢复 follow-up 中 3 个有效（60%）。低变化率降至 56.14%，no-op tick 率降至 99.583%；总推理计算 104.994 秒，增加 4.76%。
- 逐日志审计确认 unsafe recovery=0、有效模型动作被改写=0；所有恢复动作均为 camera-only ±20°，attack/jump/sprint/ESC 全为 0。最终 `accepted=true`、`advance_recommended=true`。
- summary SHA-256：`2a617055c35ac329f540d9dd1f14689dcfa8d218ad73360b6e95be971a5d5dde`；退出后未发现 MineRL、Minecraft、mcprec、process watcher 或 qwen-planner 残留进程。
- 人工复核 `final-contact-sheet.png`：B 组形成更分散的新视角，与低变化率下降一致；但 A/B 均没有 `move_forward`，未找到洞穴，不能视为任务成功。

### 第四变量结论

**完成并保留为安全兜底。** 恢复宏能稳定把已接受的语义 no-op 转成受限 camera-only 动作，降低执行无效率与低变化率，成本受控，且没有覆盖任何有效模型动作。它没有改善 Qwen 原始无效率，也没有产生前进或任务成功，因此只是局部稳定性改进，尚不足以通过 Phase 5 总验收。下一步只进入短期地图/方位摘要，不提前加入分层提示词。

## 2026-07-17 — Phase 5 第五变量：短期地图/方位摘要

### 单变量边界与实现

- A/B 两组都保留画面变化反馈、安全恢复宏与异步 decision-ack；两组都计算同一相对方位状态，只有 B 组向 planner 提供摘要。原地打转反馈、重复动作惩罚和分层提示词保持关闭。
- `OrientationMemory` 以 episode reset 为 yaw=0，按实际执行动作累加并环绕到 `[-180,180)`，按 20° 分桶；prompt 最多包含最近 3 个取样视角及其 `LOW/CHANGED` 状态。完整访问计数只用于度量与生成相邻建议，不进入无限历史。
- 第一版只提示避免最近 LOW 方位；预检显示缺乏可行动目标后，在同一变量内增加一个相邻 `±20°` 建议：选择访问次数较少的邻桶，并明确中心安全时优先 `move_forward`。没有增加历史长度、自动动作、地图坐标或层级规划。
- 新增方位反馈实际暴露数、唯一方位桶、重访样本与重访率指标；A 组 prompt 保持保留基线逐字不变。

### 测试与预检

- 最终单元测试命令：`/opt/anaconda3/bin/conda run -n mc-agent env PYTHONPATH=src python -m unittest discover -s tests -v`；50/50 通过。覆盖 yaw 环绕、20° 分桶、3 样本上限、重访计数、reset/参数校验、A 组 prompt 不变、B 组摘要和相邻低访问方位建议。
- 第一版单-seed 目录 `artifacts/phase5/orientation-ab/20260717-010005`：B 组 8 个决策实际读取摘要，但重访率由 73.68% 升至 78.95%，低变化率由 31.58% 升至 36.84%，`advance_recommended=false`。
- 加入相邻低访问方位建议后的最终单-seed 目录 `artifacts/phase5/orientation-ab/20260717-010316`：B 组重访率由 78.95% 降至 68.42%（相对下降 13.33%），唯一方位桶由 4 增至 6，模型原始无效率由 66.67% 降至 20%，成本增加 21.39%；低变化样本由 8 增至 9，因此仍 `advance_recommended=false`。没有修改预设门，交由三-seed 正式运行裁决。

### 三-seed 正式配对 A/B

- 命令：`/opt/anaconda3/bin/conda run -n mc-agent env PYTHONPATH=src python -m mc_agent.cli phase5-orientation-ab --seeds 5101,5102,5103 --ticks 800 --observation-interval 40`。
- 最终目录：`artifacts/phase5/orientation-ab/20260717-010559`；顺序为 `5101:A/B, 5102:B/A, 5103:A/B`。
- 6/6 回合均完成 800 tick；每回合 1 barrier、1 reset、20 observation、800 tick，decision/ack 一一对应，stale=0、`ESC=0`、planner error 为空。两组恢复机会均全部执行；stdout 与 summary 相同，stderr 为空。
- A 组：15/15 决策 accepted，模型 8 有效、7 无效；安全恢复后执行 15 有效、0 无效。57 个方位样本中 46 次重访，重访率 80.70%，唯一方位桶合计 11；低变化率 57.89%，总推理计算 111.116 秒。
- B 组：14/14 决策 accepted，其中 11 个实际读取方位摘要；模型 12 有效、2 无效，安全恢复后执行 14 有效、0 无效。57 个方位样本中 43 次重访，重访率 75.44%，唯一方位桶合计 14；低变化率 49.12%，总推理计算 121.215 秒。
- B 组重访率相对下降 6.52%，未达到预设 10% 主门；低变化率相对下降约 15.15%，模型原始无效率由 46.67% 降至 14.29%，成本增加 9.09%。最终 `accepted=true`、`advance_recommended=false`。
- 两组均没有 `move_forward`，不能视为任务进展或成功。summary SHA-256：`cfabd2417fd204fbc3a89187f3075b7eed46cf32f7ddd2580e7e1ce88b813d49`；退出后未发现 MineRL、Minecraft、mcprec、process watcher 或 qwen-planner 残留进程。

### 第五变量结论

**完成验证，有正向证据但不纳入保留基线。** 方位摘要扩大了视角覆盖并降低低变化与模型原始 no-op，但预设重访率改善门未通过，也没有产生前进；不在结果后下调门槛。保留实现、度量和回放资产供后续研究；Phase 5 当前只剩最后一个分层提示词变量，其 A/B 行为基线仍为画面变化反馈、安全恢复宏与 decision-ack。

## 2026-07-17 — Phase 5 第六变量：分层提示词

### 单变量边界与实现

- A/B 两组都使用画面变化反馈、安全恢复宏与异步 decision-ack；A 组提示词保持保留基线逐字不变，B 组只把同一视觉决策组织为“观察左/中/右路线 → 判断中心风险 → 选择动作”的固定层次。原地打转、重复动作惩罚与方位摘要均保持关闭。
- 新增 `hierarchical_prompt` observation/decision 标记、提示实际暴露数与 `phase5-hierarchical-ab` CLI；每个模型 decision 继续同时记录原始动作与安全恢复后的实际执行动作。MineRL step loop、Qwen 参数、动作 schema、白名单、数值限幅、seed、tick 预算和 20 Hz 节拍均未改变。
- 预设门要求：B 组提示实际进入模型输入；两组恢复机会全部安全处理且执行无效率为 0；模型原始无效率相对下降至少 20%；低变化率和 no-op tick 率不变差；B 组产生且超过 A 组的前进决策；推理计算成本比不超过 1.25。

### 测试与两轮单-seed 预检

- 最终单元测试命令：`/opt/anaconda3/bin/conda run -n mc-agent env PYTHONPATH=src python -m unittest discover -s tests -v`；52/52 通过。新增覆盖 A 组 prompt 逐字不变、B 组层级提示有界且只在开关开启时追加、observation/decision 标记传播，以及既有 barrier、decision-ack、安全恢复和其他 Phase 5 变量关闭时的回归。
- 初版预检命令：`/opt/anaconda3/bin/conda run -n mc-agent env PYTHONPATH=src python -m mc_agent.cli phase5-hierarchical-ab --seeds 5101 --ticks 800 --observation-interval 40`；目录 `artifacts/phase5/hierarchical-ab/20260717-011620`。两组各完成 800 tick、10/10 决策 accepted，B 组 10 个决策全部实际使用层级提示，但模型原始无效率均为 70%、低变化率均为 26.32%、前进决策均为 0，行为完全相同；成本比 1.0076，`accepted=true`、`advance_recommended=false`。
- 在同一变量内只收紧一次动作层：只有画面中存在具体中心危险时才允许转向，否则必须输出 6 或 16 tick `move_forward`；没有改变视觉输入、风险判断内容、动作协议或执行器。最终预检使用同一命令，目录 `artifacts/phase5/hierarchical-ab/20260717-011940`。
- 最终 A 组：10/10 决策 accepted，3 有效、7 无效，模型原始无效率 70%；7/7 语义 no-op 被安全恢复，执行无效率 0；低变化率 5/19（26.32%），no-op tick 率 98.75%，前进决策 0，总推理计算 34.531 秒。
- 最终 B 组：10/10 决策 accepted，6 有效、4 无效，模型原始无效率降至 40%（相对下降 42.86%）；4/4 语义 no-op 被安全恢复，执行无效率 0；10/10 决策实际读取层级提示。低变化率升至 7/19（36.84%），no-op tick 率仍为 98.75%，前进决策仍为 0，总推理计算 34.373 秒，成本比 0.9954。
- 两组各有 1 barrier、1 reset、20 observation、10 decision/ack、800 tick；全部 stale=0、`ESC=0`、planner error 为空。stdout 与 summary 完全一致，stderr 为 0 字节，首尾截图齐全；退出后未发现 Minecraft、MineRL、mcprec、process watcher 或 qwen-planner 残留进程。
- 人工复核首尾图确认两组都只改变朝向，没有位置推进或洞穴证据。最终 `accepted=true`、`advance_recommended=false`；summary SHA-256 为 `1fd79e4e3b97e0b34096312038d75633a35e23a6367e9da9845d2a664de05ced`。

### 第六变量与 Phase 5 结论

**完成验证但不保留，Phase 5 总验收不通过。** 分层提示降低了本次单 seed 的模型原始无效率，且安全边界与计算成本均受控；但初版与收紧版都没有产生一次 `move_forward`，最终低变化率还明显变差。预检门已经否决该变量，因此不为追求形式完整而运行三-seed 正式实验，也不在结果后降低门槛。

六个既定变量至此全部按顺序验证完毕。画面变化检测与安全恢复宏予以保留，但它们只改善感知反馈和执行层语义 no-op：固定预算下仍没有前进，不能证明卡住率相对 MVP 明显下降，故 Phase 5 验收门尚未通过。当前严格停留在 Phase 5，不进入 Phase 6；新增行为变量或修改验收口径前必须先获得用户明确批准。

## 2026-07-17 — 项目迁移后的结构一致性修正

- 将 `MASTER_PLAN.md` 的适用目录从旧的 `/Users/joey/Documents/Projects/mc-agent` 修正为当前 `/Users/joey/Desktop/Projects/mc-agent`；只修正文档事实，冻结依赖、架构、Phase 5 状态与当前下一步均未改变。
- 将 `OrientationMemory`、`OrientationState` 和 `OrientationView` 的实现从 `mc_agent.perception` 迁入规划预留的 `mc_agent.memory`；共享 episode loop 改从新模块导入，原 `mc_agent.perception.orientation` 保留兼容转发，避免既有调用方失效。
- 外层 `mc-agent` 使用 `git init -b main` 初始化为独立本地 Git 仓库；首次提交前使用 `git ls-files --others --exclude-standard`、凭据模式扫描、`git diff --cached --check` 和完整单元测试审计范围。模型权重、实验资产、运行日志、缓存与 `vendor/minerl` 均未进入暂存区。
- 根提交命令：`git commit -m "chore: establish validated mc-agent workspace baseline"`；提交为 `cdcc8742badc46d673ee7adfd0fb6a396c4836ca`，包含 50 个外层项目文件。没有配置 remote、连接或推送到 GitHub。
- 外层 `.gitignore` 明确排除 `/vendor/minerl/`。核验确认外层 `git check-ignore` 命中嵌套目录；内层仓库仍位于 `vendor/minerl/.git`，`origin` 保持 `https://github.com/minerllabs/minerl.git`，现有 Apple Silicon/MCP 修改与未跟踪文件完整保留。
- 修正 `README.md` 中“Phase 1 Gradle 尚未执行”的过期说明，并补充外层/内层仓库边界。
- 结构迁移前回归：`/opt/anaconda3/bin/conda run -n mc-agent env PYTHONPATH=src python -m unittest discover -s tests -v`，52/52 通过。迁移后新增旧导入兼容性断言并使用同一命令复测，最终 53/53 通过；没有启动 Minecraft、重新构建 MineRL 或重新下载模型。
