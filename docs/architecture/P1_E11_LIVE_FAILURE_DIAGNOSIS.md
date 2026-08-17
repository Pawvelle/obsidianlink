# P1 E11 live activation failure diagnosis

Date: 2026-08-16

Scope: offline, source-grounded audit of recorded `p1-e11-live-001`.
This is not a second live run, not a geometry change, not E12, and not a
runtime deployment.

Status: **ROOT_CAUSE_NARROWED** (Case F)

Recorded live #1 outcome remains `portal_activation_not_observed`. The
frozen E11 evaluator still requires a complete 6/6 `nether_portal`
interior. Fire is not success. The later instrumented clone
`p1-e11-diag-001` is not a formal benchmark result; see
[P1 E11 diagnostic runtime](P1_E11_DIAGNOSTIC_RUNTIME.md).

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

## What the instrumented clone proved

`p1-e11-diag-001` ran the logging-only diagnostic JAR once. JVM
`[E11-DIAG]` on the Render thread showed:

- `onBlockAdded` YES at `(0,4,1)`, dim `minecraft:overworld`
- `canLightPortal=true`, `inFireTag=true`
- Axis.X valid: origin `(0,4,1)`, bottomLeft `(1,4,1)`, width 2,
  height 3, portalCount 0
- Axis.Z not constructed
- Optional present true
- `placePortalBlocks` ENTER and EXIT both YES

Server-visible after-truth remained fire + 0/6 portal. That is Case F:
the placement method completed, but the evaluator-visible world did
not keep nether_portal blocks. Root cause is narrowed to the portal
world-write / subsequent replacement path (including client Render
thread vs unproven Server thread). The later packet-chain diagnostic
`p1-e11-packet-diagnostic-20260817-003` resolved this residual uncertainty:
the normal packet is received and the complete server-side chain executes on
`net.minecraft.world.server.ServerWorld` with `isRemote=false`, including all
six portal writes. Its timestamps show the handler runs only after ReplaySender
stops replay, after the frozen evaluator has observed 0/6 portal. Thus the
current root cause is ordering between delayed client-to-server delivery and
the evaluator observation, not the portal write path.

Live #1 evidence is unchanged. Diagnostic run is not
`integration_verified`.

## Next exact task

Authorized marshal live `p1-e11-live-002` timed out before `addAction`.
Nonblocking live `p1-e11-live-003` queued `server.execute` after
`addAction` then still timed out before `waitForNextObservation`.
Await-after-tick live `p1-e11-live-004` reached await after
`waitForNextObservation` and still timed out; `processRightClickBlock`
never ran. Do not auto-retry. Do not change geometry, evaluator success,
or the observation window. Do not start E12. See
[P1 E11 server-thread marshal](P1_E11_SERVER_THREAD_MARSHAL.md).
