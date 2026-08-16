# P1 E10 DrawingDecorator runtime geometry

Date: 2026-08-16

Scope: deploy a fail-closed EnvServer DrawBlock applicator for E10 lava
at world `(0, 4, 2)`. This is not E10 vanilla conversion, E11, or E12.

## Why the deployed runtime ignored DrawingDecorator

MineRL MCP-Reborn `EnvServer` unmarshals Mission XML, including
`DrawingDecorator` / `DrawBlock`, but never applied those handlers.
Malmo's `DrawingDecoratorImplementation` lives on the unused Malmo
Minecraft handler path and is not fail-closed for obsidian. The P1
`prepareControlledBuildArea()` path then fills `feetY` through
`feetY+8` with air, so even a pre-platform lava block would be cleared.

## Patch chain

Reconstructed isolated MCP-Reborn from the installed MineRL 1.0.2
snapshot. The dirty `vendor/minerl` HEAD was not built or deployed.

1. MineRL vendor commit `cdeae668c2f334e3c9117adf651b5a94436b45f8`
2. MCP-Reborn `1.16.5-20210115` `1e71be5bd4c49bc4d6ab0ee559c31b298b7697a3`
3. `patches/minerl/obsidianlink-envserver.patch`
   SHA-256 `0944d0f09e1d915e45781ea565ff2696b19eed3a395706a1267ce5b330e20609`
4. `patches/minerl/disable-client-audio.patch`
   SHA-256 `1911d1ca6ba18a445a6c8a5038d3b171bf3aaab8eace21af993fb547684dd676`
5. `patches/minerl/e10-drawing-decorator.patch`
   SHA-256 `a09607bbd3274429e9791f24707d2fb978f62d79fa59da86beb9fd9d50cba26f`

Audio-fixed JAR before this change:
`ac6a46639497117e0813ba2262cc232eca1f2921070dcd8851c6a21501c39d62`

Launcher (unchanged):
`7e15699c0d0aea517f87680eb5d760d02519d9744285fa0d348f799e2ed77183`

Deployed E10-geometry JAR:
`935af788fcce57ce30df6ddcebca491cd4ae3253683cf14bedebea02eb4a0afa`

Build: isolated `/tmp/obsidianlink-p1-e10-runtime`,
`/opt/anaconda3/envs/mc-agent/bin/java` Zulu 8.90.0.19, `./gradlew shadowJar`.

Semantic diff versus the audio-fixed JAR: `EnvServer.class`,
`EnvServer$1.class` (existing socket thread recompiled), and
`version.properties` timestamp. Unexpected class diff count: 0.
`SoundEngine.class` is byte-identical. Portal-transition classes: none.

## Application order

`create world` → `applyServerInitialConditions` →
`prepareControlledBuildArea` → queue `applyMissionDrawingDecorator` →
inventory / position → skip-first-frames → bounded
`awaitMissionDrawingDecorator`. Waiting immediately after world create
deadlocks because the integrated server is paused; attempt
`e10-geometry-20260816-001` timed out and was not a conversion retry.
Allowlist is `minecraft:lava` source only. `obsidian` and any other
DrawBlock fail closed. No decorator means no lava.

## Geometry smoke

Evidence: `runs/p1_e10_obsidian_conversion/e10-geometry-20260816-002`.
Outcome `e10_geometry_ready`. Target `(0, 4, 2)` = lava / lava / source.
Water cell `(0, 4, 1)` = air / no fluid. `truth_missing_count=0`.
`tested_action_count=0`. `integration_verified=false`. Conversion was
not run.
