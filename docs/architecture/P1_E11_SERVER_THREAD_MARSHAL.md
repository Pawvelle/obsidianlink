# P1 E11 server-thread marshal attempt

Date: 2026-08-17

Scope: marshal `flint_and_steel` world mutation onto the Minecraft
integrated server thread at the EnvServer `execActions` boundary, then
one authorized E11 live run. This is not E12 and not
`integration_verified`.

Status: **FAILED** / marshal wait timed out / no second patch round

## How marshaling was implemented

`patches/minerl/e11-server-thread-marshal.patch` changes only
`EnvServer.execActions`. When the MineRL action contains `use=1` and
the held item is `flint_and_steel`, EnvServer:

1. queues `server.execute(() -> player.interactionManager.func_219441_a(...))`
   with vanilla `ServerPlayerEntity.pick(4.5)` and the held stack;
2. waits up to 30s for that task;
3. if it applied, strips mouse button 1 so ReplaySender does not also
   right-click on the Render thread.

Vanilla causality is unchanged: flint_and_steel → fire →
`canLightPortal` → `PortalSize` → `placePortalBlocks`. No
`setBlockState(nether_portal)` fake success. PortalSize / evaluator /
geometry / observation window were not modified.

## World mutation thread

The marshaled `processRightClickBlock` **never ran**. There is no
`[E11-MARSHAL]` log line. `server.execute` was queued from
`EnvServerSocketHandler` **before** `ReplaySender.addAction`. The
Render thread was blocked in `ReplaySender.wait()`, and the log shows
`Saving and pausing game...`. The 30s `CompletableFuture.get` then
threw `flint_and_steel server-thread marshal timed out`.

## Live run

Episode `p1-e11-live-002`. Raw
`runs/p1_e11_portal_activation/e11-live-20260817-001/`. Compact
`runs/history/p1-e11-live-20260817-001/`.

- fresh process 1, reset 1, retry 0
- BEFORE: 14/14 obsidian, 6/6 air, overworld
- tested actions 0; flint_and_steel never accepted
- AFTER truth missing; outcome `truth_identity_mismatch`
- 6/6 nether_portal: **NO**
- E11 real reviewed success: **NO**
- `integration_verified`: **NO**
- E12: **NOT STARTED**

Live #1 (`p1-e11-live-001`) remains `portal_activation_not_observed`.
No second patch round.

## Runtime identity

Run JAR SHA-256 (marshal, timeout):

`c69fd49e030d501b4c0b1cca9f58b47a2722baf0af9695d679d84188a8499196`

Production restored after failure:

`836cb5ac6f89edca3cec255dd895e791212b04794d3349eb13a1b2b313416b6f`

Launcher unchanged:

`7e15699c0d0aea517f87680eb5d760d02519d9744285fa0d348f799e2ed77183`

Marshal backup:

`build/libs/backups/mcprec-6.13.jar.e11-marshal-timeout-20260817-c69fd49e`
