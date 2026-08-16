# P1 E11 DrawingDecorator obsidian fixture runtime

Date: 2026-08-16

Scope: fail-closed EnvServer DrawBlock extension so E11 can pre-place a
complete obsidian portal **frame** as a calibration fixture. This is not
portal activation, not Agent construction, not E12, and not a Gradle
authorization.

Status: **NEEDS_E11_RUNTIME_GEOMETRY_AUTHORIZATION**

The Python/E11 offline contract is implemented. The current deployed P1
runtime JAR still uses the E10 lava-only allowlist and **rejects
obsidian**. Do not build or deploy this patch until the user authorizes
one isolated Gradle rebuild from the deployed P1 baseline.

## Why the current runtime cannot pre-place the E11 frame

Deployed JAR SHA-256:

`935af788fcce57ce30df6ddcebca491cd4ae3253683cf14bedebea02eb4a0afa`

`patches/minerl/e10-drawing-decorator.patch` allows `BlockType.LAVA`
only and throws `DrawBlock must not pre-place obsidian`. Mission XML can
already emit the frozen 14-cell obsidian frame through
`PortalA0EnvSpec(allow_obsidian_frame_fixture=True)`, but EnvServer will
fail closed before the world contains that fixture.

There is no other stable P1 mechanism that can place a complete
obsidian ring without Agent mining, E10 conversion of every cell, or
forbidden `/setblock` / `DrawBlock portal` cheats.

## Minimal runtime change

Apply **after** the existing chain, never from dirty `vendor/minerl` HEAD:

1. MineRL vendor commit `cdeae668c2f334e3c9117adf651b5a94436b45f8`
2. MCP-Reborn `1.16.5-20210115` `1e71be5bd4c49bc4d6ab0ee559c31b298b7697a3`
3. `patches/minerl/obsidianlink-envserver.patch`
4. `patches/minerl/disable-client-audio.patch`
5. `patches/minerl/e10-drawing-decorator.patch`
6. `patches/minerl/e11-drawing-decorator-obsidian.patch`

The E11 patch only changes `applyAllowedDrawBlocks`:

- still allow E10 lava source (`Blocks.LAVA.getDefaultState()`)
- additionally allow obsidian (`Blocks.OBSIDIAN.getDefaultState()`)
- still reject `BlockType.PORTAL`, `END_PORTAL`, and `FIRE`
- still reject every other DrawBlock type (fail closed)
- no portal-transition, dimension, or player-move code

Allowed fixture blocks after this patch: `lava` (existing E10 path) and
`obsidian` (E11 frame). Forbidden: `portal` / `nether_portal`, `fire`,
and arbitrary world edits.

## How portal blocks remain forbidden

Python `validate_e11_initial_geometry` and
`allow_obsidian_frame_fixture` reject `nether_portal`, `portal`, `fire`,
`lava`, and water. The Java allowlist independently rejects
`BlockType.PORTAL`, `END_PORTAL`, and `FIRE`. Success must still come
from one `use_item(flint_and_steel)` plus vanilla `PortalSize`.

## How future E12 code stays out

Rebuild only from the deployed P1 baseline plus the patch chain above.
Do not `shadowJar` the current `vendor/minerl` working tree. The E11
patch must not mention `entered_via_portal`, `PortalTransition`,
`netherEntry`, or dimension change.

## Gradle / deploy

A future authorized rebuild **does** require Gradle `shadowJar` and a
new JAR deploy. This document does not authorize that. Semantic diff
versus the current E10 JAR should be limited to `EnvServer.class` /
inner class recompilation and `version.properties` timestamp.
Unexpected class diff count must stay 0. Portal-transition classes:
none.

## Frozen geometry (Python, already implemented)

Minimum valid 1.16.5 frame from MCP-Reborn `PortalSize.isValid()`:
interior width 2–21, height 3–21, so the frozen minimum is interior
2×3 / outer 4×5, complete 14-cell obsidian ring, axis X at z=1.
Interior starts as air. Ignition cell `(0, 4, 1)`. Spawn `(0, 4, 0)`,
yaw `0.0`, pitch `60.0`. Observation window: 3 ticks (engineering
buffer for MineRL observation lag; activation itself is synchronous
from `AbstractFireBlock.onBlockAdded`).
