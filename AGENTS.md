# mc-agent 执行约束

本项目的唯一主规划是 `MASTER_PLAN.md`。

1. 开始任何工作前，先读取 `MASTER_PLAN.md` 和 `docs/EXECUTION_LOG.md`。
2. 严格按主规划的阶段顺序执行；前一阶段验收门未通过，不进入下一阶段。
3. 不得自行更换 MineRL 版本、Qwen 模型、模型提交、Python/JDK 主版本或总体架构。
4. 任何偏离主规划的改动，都必须先得到用户明确同意，再更新 `MASTER_PLAN.md` 的“规划变更记录”。
5. 每次完成验证后，将命令、结果和未解决问题追加到 `docs/EXECUTION_LOG.md`。
6. 模型输出永远先经过结构化解析、动作白名单和数值限幅，不能直接执行模型生成的代码或 shell 命令。
7. MineRL 的 Gradle 构建会执行第三方代码；只有用户明确批准该风险后才能在沙箱外运行。
