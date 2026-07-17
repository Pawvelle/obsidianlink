# mc-agent

一个适合个人学习与持续开发的 Minecraft Agent。MineRL 负责运行 Minecraft
1.16.5，Qwen3-VL-2B-Instruct 读取第一人称画面并生成受约束的 JSON 宏动作，
本地执行器再把动作安全地送入环境。

核心闭环很简单：

```text
Minecraft 画面 -> Qwen 规划 -> JSON 校验与限幅 -> MineRL 动作 -> 新画面
```

## 项目结构

```text
mc-agent/
├── README.md
├── AGENTS.md
├── ROADMAP.md
├── environment.yml
├── model.lock.json
├── mc_agent/
│   ├── main.py       # 命令行入口
│   ├── agent.py      # Agent 主循环
│   ├── env.py        # MineRL 生命周期封装
│   ├── qwen.py       # 异步 Qwen 视觉规划器
│   ├── actions.py    # JSON 动作协议、执行器和 watchdog
│   ├── memory.py     # 简单方向记忆与画面变化检测
│   └── logger.py     # 可回放 JSONL 日志
├── scripts/          # 环境检查和 smoke test
├── tests/            # 核心单元测试
├── models/           # 本地模型，不提交到外层 Git
├── runs/             # 新运行结果与本地历史记录
└── vendor/minerl/    # 独立的 MineRL 上游仓库
```

## 环境

当前验证环境固定为：

- Conda 环境：`mc-agent`
- Python 3.10.20 / OpenJDK 8.0.472
- MineRL 1.0.2 / Gym 0.23.1 / NumPy 1.23.5
- Qwen3-VL-2B-Instruct / Apple MPS / FP16

已有环境直接激活即可，不需要重新安装 MineRL 或重新下载模型：

```bash
conda activate mc-agent
python scripts/check_environment.py
```

`environment.yml` 保存 Python 依赖版本，`model.lock.json` 是模型仓库、提交和
权重校验值的唯一配置。只有本地权重确实缺失时，才使用
`scripts/download_model.sh`。

MineRL 的本地 Apple Silicon 构建已经验证过。不要随意执行
`pip install ./vendor/minerl` 或重新运行 Gradle；构建会执行第三方代码，并且可能
覆盖当前可用产物。

## 测试

运行核心单元测试：

```bash
python -m unittest discover -s tests -v
```

按需进行小范围检查：

```bash
python scripts/smoke_test_minerl.py --mode fake
python scripts/smoke_test_qwen.py
python scripts/smoke_test_minerl.py --mode real --steps 10
python scripts/smoke_test_agent.py --frame runs/smoke/findcave-reset.png
python scripts/smoke_test_agent.py --frame <frame.png> --after-forward
python scripts/smoke_test_agent.py --frame <non-cave.png> --expect-no-cave
```

## 运行 Agent

先用单回合观察运行情况：

```bash
python -m mc_agent.main --watch --episodes 1 --ticks 800 --observation-interval 40
```

`--watch` 会打开名为 `MineRL Render` 的实时第一人称观察窗口。它只显示 Agent
收到的画面，不接管键盘或鼠标；关闭 Agent 时请在终端按 `Ctrl+C`。不需要观察窗口时
可以省略该参数。

需要连续回放时再增加回合数：

```bash
python -m mc_agent.main --episodes 5 --ticks 800 --observation-interval 40
```

结果默认写入 `runs/episodes/<时间>/`。每回合包含初始画面、最终画面、决策帧、
逐 tick 事件和汇总指标。Qwen 在独立 worker 中推理，不阻塞 MineRL step loop；
episode 切换前的 barrier 会等待旧推理结束并清空旧 observation/decision。

模型动作包含 fail-closed 的 `cave_visible` 字段。字段缺失时按 false 处理；模型只有
明确报告洞穴，并在理由中同时给出暗色、岩石、开口和方向证据时，系统才把对应决策
帧记为洞穴候选。候选仍需人工复核，不会自动当成 FindCave 成功。

## 当前边界

Agent 已经通过 5×800 ticks 安全前进验收，能持续产生模型驱动的 forward ticks 和
有效动作变化。洞穴负样本及运行集成也已验证，但现有回放没有可信的洞穴入口正样本，
因此还不能把“找到洞穴”标为完成。后续按 [ROADMAP.md](ROADMAP.md) 的 Phase 4
继续，只验证真实洞穴正例与接近动作。历史验证记录保存在
`runs/history/EXECUTION_LOG.md`；大型历史运行资产只在本机保留，不进入外层 Git。

`vendor/minerl` 是独立 Git 仓库。外层项目不会提交它、删除它，也不会改写它的
Git 历史。
