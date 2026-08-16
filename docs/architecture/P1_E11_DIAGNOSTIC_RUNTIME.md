# P1 E11 diagnostic instrumentation runtime

Date: 2026-08-17

Scope: one logging-only JVM instrumentation of the E11 ignition
path. This is **not** a formal capability run, not E12, and not
`integration_verified`.

Status: **COMPLETED** / Case F / `ROOT_CAUSE_NARROWED`

## Identity

Production E11 JAR (restored after the run):

`836cb5ac6f89edca3cec255dd895e791212b04794d3349eb13a1b2b313416b6f`

Launcher (unchanged):

`7e15699c0d0aea517f87680eb5d760d02519d9744285fa0d348f799e2ed77183`

Temporary diagnostic JAR:

`12c320caa072d55b0eb7280f9be489ca3d2ce9912b65a54a09a241042877ae03`

Backup (not overwritten):

`build/libs/backups/mcprec-6.13.jar.e11-geometry-20260816-836cb5ac`

## Patch / build

Isolated tree `/tmp/obsidianlink-p1-e11-diagnostic-runtime` from the
installed MineRL 1.0.2 MCP-Reborn snapshot. Dirty `vendor/minerl` HEAD
was not built.

1. MineRL `cdeae668c2f334e3c9117adf651b5a94436b45f8`
2. MCP-Reborn `1e71be5bd4c49bc4d6ab0ee559c31b298b7697a3`
3. `obsidianlink-envserver.patch`
4. `disable-client-audio.patch`
5. `e10-drawing-decorator.patch`
6. `e11-drawing-decorator-obsidian.patch`
7. `e11-portal-activation-diagnostic.patch`
   SHA-256 `5783c7d4f1be95cbeb7fc3ceaa15b5ad0f2c23f3e8ce8845900cdca398b9c9df`

Java: `/opt/anaconda3/envs/mc-agent/bin/java` Zulu 8.90.0.19.
Command: `./gradlew shadowJar --no-daemon`. Behavior changes: **NONE**
(LOGGER.info only).

Semantic diff versus production: `AbstractFireBlock.class`,
`PortalSize.class`, `version.properties`. Unexpected class diff count:
0. `EnvServer.class` and `SoundEngine.class` byte-identical. Future E12
code: NO.

## Instrumented run

Episode `p1-e11-diag-001`. Raw
`runs/p1_e11_portal_activation/e11-diagnostic-20260817-001/`. Compact
`runs/history/p1-e11-diagnostic-20260817-001/`.

Fresh process 1, launch 1, reset 1, retry 0, tested actions 1. Frozen
geometry / `use_item(flint_and_steel)` / 3-tick window unchanged.

BEFORE: 14/14 obsidian, 6/6 air, portal 0, fire 0, overworld,
`truth_missing_count=0`. AFTER: ignition fire, 0/6 portal. Outcome
`portal_activation_not_observed`. This does **not** replace live #1 and
does not set `integration_verified`.

## JVM trace

All `[E11-DIAG]` lines were on the **Render thread**:

- `onBlockAdded` YES at `(0,4,1)`, dim `minecraft:overworld`
- `canLightPortal=true`, `inFireTag=true`
- Axis.X: origin `(0,4,1)`, bottomLeft `(1,4,1)`, width 2, height 3,
  portalCount 0, valid true
- Axis.Z not constructed (`fallbackAttempted=false`)
- Optional present true
- `placePortalBlocks` ENTER YES / EXIT YES

Case F: placement method completed, server-visible interior stayed
0/6 portal. Root cause is narrowed to the portal world-write /
subsequent replacement path, including the fact that the callback ran
on the client Render thread rather than a proven Server thread.

Live #1 (`p1-e11-live-001`) remains `portal_activation_not_observed`.
E12: NOT STARTED. P1 Hard Gate: NOT PASSED.

## Write-path follow-up (`p1-e11-diag-002`)

Temporary JAR SHA-256:

`9011798f67a93adbb391890c2857249290cc1ac32359d3557c892054a9fe0029`

Production JAR restored afterward to `836cb5ac…`. Patch SHA-256
`0f43178fa2f2466af336c67f2843a0914a41934e87204d07a274e51cb73d3156`.
Semantic diff: `AbstractFireBlock`, `PortalSize`, `NetherPortalBlock`,
switch-map `$1` line numbers, `version.properties`. Unexpected count 0.

Episode `p1-e11-diag-002`. Compact
`runs/history/p1-e11-diagnostic-20260817-002/`.

JVM write path (all six interior cells):

- world class: `net.minecraft.world.server.ServerWorld`
- `isRemote=false`
- thread: `Render thread`
- `setBlockState` accepted=`true`
- immediate after-state: `minecraft:nether_portal[axis=x]`
- `NetherPortalBlock.updatePostPlacement`: not observed

Evaluator after-truth remained fire + 0/6 portal. This is still not
`integration_verified`.

The later authorized marshal live `p1-e11-live-002` is a different
failure: EnvServer waited on `server.execute` while the integrated
server was paused, so flint_and_steel never ran. See
[P1 E11 server-thread marshal](P1_E11_SERVER_THREAD_MARSHAL.md).
Do not treat that timeout as a second diagnostic clone of this write
path.
