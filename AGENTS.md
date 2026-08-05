# Agent 工作规则

开始前先读 `README.md`、`PROJECT_STATUS.md`、`ROADMAP.md` 和 `BENCHMARK_SPEC.md`。

1. 只做 `PROJECT_STATUS.md` 中的当前任务，不提前开发后续阶段。
2. 当前主线是用原版水和熔岩机制生成黑曜石并构建传送门。
3. 不自行改变 MineRL、Minecraft、Python、JDK、Gym、NumPy、Qwen 或模型版本。
4. `vendor/minerl` 是独立仓库；未经授权不得修改。
5. 每次 Gradle 构建、真实 MineRL/Minecraft 运行、付费 API 调用都要单独获得用户批准。
6. 模型输出必须经过严格解析、动作白名单、类型检查和数值限制；不得执行模型生成的代码、命令或无限输入。
7. Planner 不能阻塞环境 step 循环，过期决策必须丢弃。
8. Agent 可见观察与 evaluator 真值必须分开，真值不能进入 prompt 或 memory。
9. observation、action、message、evaluation 和 log 都应带 `episode_id`、`step_id`，适用时带 `agent_id`。
10. 先用 FakeBackend 和确定性 driver 证明任务，再连接模型。
11. 运行结果写入 `runs/`，不得保存密钥、模型权重或隐藏推理。
12. 修改后运行相关离线测试，并报告结果和仍未验证的限制。

## 开始检查

```bash
git status --short
python -m obsidianlink --check
```

## 完成条件

- 当前任务的代码和测试完成；
- 离线测试通过；
- 没有把 evaluator-only 信息泄漏给 Agent；
- `PROJECT_STATUS.md` 已更新；
- 未经授权不提交、不推送、不启动真实环境。
