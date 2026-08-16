# P1 E11 live activation failure diagnosis

Date: 2026-08-16

Scope: offline, source-grounded audit of recorded `p1-e11-live-001`.
This is not a second live run, not a geometry change, not E12, and not a
runtime deployment.

Status: **NEEDS_E11_DIAGNOSTIC_RUNTIME_AUTHORIZATION**

Recorded outcome remains `portal_activation_not_observed`. The frozen
E11 evaluator still requires a complete 6/6 `nether_portal` interior.
Fire is not success.

## Recorded snapshot

Episode `p1-e11-live-001`. Compact evidence:
`runs/history/p1-e11-live-20260816-001/`.

BEFORE z=1 (x=-1..2, evaluator truth):

```text
y=7  O O O O
y=6  O A A O
y=5  O A A O
y=4  O A A O
y=3  O O O O
```

AFTER the single accepted `use_item(flint_and_steel)`:

```text
y=7  O O O O
y=6  O A A O
y=5  O A A O
y=4  O F A O
y=3  O O O O
```

Ignition cell `(0, 4, 1)`: air → fire. Other interior cells remain air.
Portal count 0/6. Controls unchanged. Dimension
`minecraft:overworld`. `truth_missing_count=0`.

## Vanilla call chain (MCP-Reborn 1.16.5)

1. MineRL `use=1` → EnvServer mouse button 1 (right click).
2. `FlintAndSteelItem.onItemUse` clicks the UP face of `(0, 3, 1)`
   and places fire at `(0, 4, 1)` with flags `11`.
3. Server `Chunk.setBlockState` calls `FireBlock.onBlockAdded`, which
   calls `super` → `AbstractFireBlock.onBlockAdded`.
4. `canLightPortal` requires `World.OVERWORLD` or `World.THE_NETHER`
   by **reference equality** on `getDimensionKey()`.
5. `PortalSize.func_242964_a(world, firePos, Axis.X)` constructs
   Axis.X first, then Axis.Z if X is invalid.
6. Predicate: `isValid() && portalBlockCount == 0`.
7. On success, `placePortalBlocks()` writes `Blocks.NETHER_PORTAL`
   synchronously with flags `18`.

`PortalSize.canConnect` treats air, `BlockTags.FIRE`, and
`Blocks.NETHER_PORTAL` as interior. Fire remaining after 3 ticks is
therefore not an observation-window problem: placement is synchronous.

## PortalSize audit of the recorded world

Axis.X (`rightDir = WEST`, constant Z, width along X):

| Condition | Source | Evidence | Result |
|---|---|---|---|
| downward scan stops on obsidian | `func_242971_a` | `(0, 3, 1)` obsidian | PASS |
| left-edge distance | `func_242972_a` EAST from `(0, 4, 1)` | hits obsidian at x=2, span=2 | PASS |
| bottomLeft | `func_242971_a` | `(1, 4, 1)` | PASS |
| width 2..21 | `func_242974_d` WEST | width=2; below-interior is obsidian | PASS |
| side pillars | `func_242969_a` | x=-1 and x=2, y=4..6 obsidian | PASS |
| interior canConnect | `func_242969_a` | fire+air / air | PASS |
| height 3..21 | `func_242975_e` | height=3 (top interior y=7 is obsidian) | PASS |
| top frame | `func_242970_a` | `(0, 7, 1)` and `(1, 7, 1)` obsidian | PASS |
| `isValid` and no pre-existing portal | `func_242964_a` | portalCount=0 | PASS |

Axis.Z (`rightDir = SOUTH`) is invalid, as expected for a constant-Z
frame. The first failed Z condition is `func_242971_a` left-edge
search NORTH of the fire cell: `(0, 4, 0)` is air over
`prepareControlledBuildArea` grass at `(0, 3, 0)`, not obsidian.
Axis.Z is not required if Axis.X is valid.

The 14/14 E11 evaluator frame is therefore **not** a vanilla Axis.X
geometry failure. E11_CONFIG matches `PortalSize` for this plane.

## What static evidence cannot prove

The live world still has fire and 0 portal blocks. If Axis.X is valid
and `onBlockAdded` ran, vanilla would have replaced the interior with
`nether_portal`. Missing runtime signals:

1. whether `AbstractFireBlock.onBlockAdded` executed on the server
2. `canLightPortal` / `getDimensionKey() == World.OVERWORLD`
3. whether `BlockTags.FIRE` contained `Blocks.FIRE` at that moment
4. Axis.X/Z `width`, `height`, `bottomLeft`, `isValid` inside the JVM
5. whether `placePortalBlocks` was invoked

Prepared logging-only patch (not applied, not built, not deployed):
`patches/minerl/e11-portal-activation-diagnostic.patch`.

EnvServer DrawBlock patches only place lava/obsidian with flags `2`.
They do not disable `onBlockAdded`, ticks, or portal callbacks.
**NO EVIDENCE** those patches caused this failure.
**NO EVIDENCE OF WORLD-LEVEL PORTAL DISABLE** in Mission game rules.

Grid canonicalization maps `minecraft:nether_portal` / `portal` to
`nether_portal`. If portal blocks had existed, the evaluator would not
have called them air.

## Next exact task

Authorize applying and deploying the diagnostic patch, then one
instrumented E11 ignition. Do not retry the current uninstrumented
run. Do not start E12. Do not treat fire as success.
