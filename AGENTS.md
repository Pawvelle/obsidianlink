# mc-agent Development Constraints

This project is intended for personal learning and long-term development. See
`ROADMAP.md` for its execution plan.

1. Read `ROADMAP.md` before starting project work.
2. Do not independently change the major MineRL, Qwen, Python, or JDK versions,
   or the pinned model commit.
3. Model output must pass structured parsing, an action allowlist, and numeric
   clamping. Never directly execute model-generated code or shell commands.
4. Keep the MineRL step loop decoupled from Qwen inference; the environment loop
   must not wait for model inference.
5. `vendor/minerl` is an independent nested Git repository. The outer repository
   must not commit, delete, or rewrite its history.
6. MineRL Gradle builds execute third-party code and may be rerun only with the
   user's explicit approval.
7. Write new run results to `runs/`; historical validation records live in
   `runs/history/`.
8. After structural or functional changes, run the relevant tests and report the
   verification result and any remaining issues at handoff.
