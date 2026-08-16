# P1 E11 DrawingDecorator obsidian fixture runtime

Date: 2026-08-16

Scope: fail-closed EnvServer DrawBlock extension so E11 can pre-place a
complete obsidian portal **frame** as a calibration fixture. This is not
portal activation, not Agent construction, and not E12.

Status: **DEPLOYED / REAL GEOMETRY VERIFIED**

## Why the E10 runtime could not pre-place the E11 frame

Previous deployed JAR SHA-256:

`935af788fcce57ce30df6ddcebca491cd4ae3253683cf14bedebea02eb4a0afa`

`patches/minerl/e10-drawing-decorator.patch` allowed `BlockType.LAVA`
only and threw `DrawBlock must not pre-place obsidian`.

## Patch chain

Reconstructed isolated MCP-Reborn from the installed MineRL 1.0.2
snapshot. The dirty `vendor/minerl` working tree was not built or
deployed.

1. MineRL vendor commit `cdeae668c2f334e3c9117adf651b5a94436b45f8`
2. MCP-Reborn `1.16.5-20210115` `1e71be5bd4c49bc4d6ab0ee559c31b298b7697a3`
3. `patches/minerl/obsidianlink-envserver.patch`
   SHA-256 `0944d0f09e1d915e45781ea565ff2696b19eed3a395706a1267ce5b330e20609`
4. `patches/minerl/disable-client-audio.patch`
   SHA-256 `1911d1ca6ba18a445a6c8a5038d3b171bf3aaab8eace21af993fb547684dd676`
5. `patches/minerl/e10-drawing-decorator.patch`
   SHA-256 `a09607bbd3274429e9791f24707d2fb978f62d79fa59da86beb9fd9d50cba26f`
6. `patches/minerl/e11-drawing-decorator-obsidian.patch`
   SHA-256 `2ba63ea252e80993010c1cd8a11031ae716719697e0bb1562009e4dcf639a0c9`

Allowed fixture blocks: `lava` (E10) and `obsidian` (E11 frame).
Forbidden: `PORTAL` / `END_PORTAL` / `FIRE` and every other DrawBlock
type (fail closed). No portal-transition, dimension, or player-move
code.

## Deployment

Isolated tree: `/tmp/obsidianlink-p1-e11-runtime`.
Java: `/opt/anaconda3/envs/mc-agent/bin/java` Zulu 8.90.0.19.
Command: `./gradlew shadowJar`.

Previous E10 JAR backup (not overwritten):
`build/libs/backups/mcprec-6.13.jar.e10-geometry-20260816-935af788`

Launcher (unchanged):
`7e15699c0d0aea517f87680eb5d760d02519d9744285fa0d348f799e2ed77183`

Deployed E11-fixture JAR:
`836cb5ac6f89edca3cec255dd895e791212b04794d3349eb13a1b2b313416b6f`

Semantic diff versus the E10 JAR: `EnvServer.class` and
`version.properties` timestamp. Unexpected class diff count: 0.
`SoundEngine.class` is byte-identical. Portal-transition classes: none.

## Geometry smoke

Evidence: `runs/p1_e11_portal_activation/e11-geometry-20260816-001`.
Compact history: `runs/history/p1-e11-geometry-20260816-001/`.
Outcome `e11_geometry_ready`. Frame 14/14 obsidian. Interior 6/6 air.
Portal block count 0. Fire block count 0. Ignition cell `(0, 4, 1)` =
air. Controls expected. `truth_missing_count=0`. `tested_action_count=0`.
`integration_verified=false`. Flint-and-steel was not executed.

## Frozen geometry

Minimum valid 1.16.5 frame from MCP-Reborn `PortalSize.isValid()`:
interior width 2–21, height 3–21, so the frozen minimum is interior
2×3 / outer 4×5, complete 14-cell obsidian ring, axis X at z=1.
Interior starts as air. Ignition cell `(0, 4, 1)`. Spawn `(0, 4, 0)`,
yaw `0.0`, pitch `60.0`. Observation window: 3 ticks (engineering
buffer; unused by geometry smoke).

A later authorized activation run `p1-e11-live-001` produced
`portal_activation_not_observed` on this same deployed JAR: fire at
the ignition cell, 0/6 portal. Compact evidence:
`runs/history/p1-e11-live-20260816-001/`. Offline PortalSize replica of
that snapshot treats Axis.X as valid; see
[P1 E11 live failure diagnosis](P1_E11_LIVE_FAILURE_DIAGNOSIS.md).
This document remains geometry provenance; it does not set
`integration_verified`.
