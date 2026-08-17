# P1 canonical MineRL runtime

Date: 2026-08-17

Status: deployed; E11 real calibration failed; E12 not started.

## Reproducible source and build

`scripts/build_p1_canonical_runtime.py` requires the frozen generated
MCP-Reborn source files to match their recorded SHA-256 values, stages a clean
tree, applies the production allowlist, and runs:

```text
./gradlew --no-daemon shadowJar
```

Production source changes, in order:

1. `obsidianlink-envserver.patch`: controlled EnvServer, initial conditions,
   build area, grid/dimension truth;
2. `p1-canonical-audio-source.patch`: client audio mitigation (the frozen
   launcher already carries the matching JVM property);
3. `e10-drawing-decorator.patch`: lava fixture;
4. `e11-drawing-decorator-obsidian.patch`: obsidian frame fixture while still
   rejecting portal and fire;
5. `p1-env-integrated-server-unpaused.patch`: an environment process
   (`envPort != 0`) does not pause its integrated server.

The build does not use `mcp_patch.diff`, any E11 marshal or diagnostic patch,
the paused-executor patch, or E12 code. Canonical JAR SHA-256:
`684c20ec533897b44e9f2f73340f66ab41a6f61e7c9ae7e0f1db6fae7430751e`.

## Semantic diff

Against the previous production JAR (`836cb5ac…`), changed semantic entries
are only:

- `net/minecraft/server/integrated/IntegratedServer.class`;
- `version.properties`.

EnvServer, SoundEngine, AbstractFireBlock, NetherPortalBlock, PortalSize,
FlintAndSteelItem, Entity, and ServerPlayerEntity are byte-identical.
Forbidden marshal/diagnostic/E12 markers are absent.

## Real E11 result

Episode `p1-e11-canonical-runtime-20260817-002` used one fresh process, one
reset, one accepted `use_item(flint_and_steel)`, and zero retry.

- before: 14/14 obsidian, 6/6 air, 0 portal, overworld,
  `truth_missing_count=0`;
- execution: MineRL `use=1` → EnvServer → ReplaySender → normal Minecraft
  client right-click on the Render thread;
- after: one fire, five air, 0/6 nether portal,
  `truth_missing_count=0`;
- outcome: `portal_activation_not_observed`.

No `Saving and pausing game` event occurred. The current minimal blocker is
therefore no longer paused task execution: existing evidence does not prove
that the normal client use-item packet reached and executed
`FlintAndSteelItem` on the integrated server. No retry was made.
