# P1 E11 non-blocking server-thread marshal attempt

Date: 2026-08-17

Scope: reschedule `flint_and_steel` world mutation onto the integrated
server thread **without** waiting inside `EnvServer.execActions()`.
This is not E12 and not `integration_verified`.

Status: **FAILED** / live #3 queued-then-timeout; live #4 await-after-tick still timeout / no further patch round

## Paused-executor validation (`8b2258e`)

The patch does not call `super.tick()` or change PortalSize, geometry,
the evaluator, the observation window, E12, or audio code. But its
`isGamePaused || super.canRun(...)` condition admits every queued
`TickDelayedTask` while paused, not just E11. It therefore broadens
generic paused-executor semantics and is not approved for deployment or
a fresh E11 run. Historical live #1--#4 evidence is unchanged.

## How scheduling avoided the live #2 deadlock (and what remained)

`patches/minerl/e11-server-thread-marshal-nonblocking.patch` still
changes only `EnvServer`. When the action has `use=1` and the held item
is `flint_and_steel`:

1. `execActions` queues `server.execute(() -> processRightClickBlock)`
   and returns after `ReplaySender.addAction` (no `Future.get` there);
2. mouse button 1 is stripped so ReplaySender does not also right-click
   on the Render thread;
3. `stepClient` then calls `awaitPendingFlintAndSteelMarshal()` **after**
   `execActions` and **before** `waitForNextObservation`.

Vanilla causality is unchanged. No `setBlockState(nether_portal)`.

Live #2 waited **before** `addAction`, so the queue log never appeared.
This attempt did queue from `EnvServerSocketHandler`. That is progress
in scheduling, not portal success.

## World mutation thread

`processRightClickBlock` **never ran**. There is a queued log and no
`thread=` execution log. After `addAction`, `awaitPending` still blocked
`EnvServerSocketHandler` for 30s. The Render thread stayed silent
(`Saving and pausing game...` just before the queue). The integrated
server therefore never drained `server.execute`.

New minimal failure: waiting on the socket thread **before**
`waitForNextObservation` is still too early. `addAction` alone did not
unpause the integrated server.

## Live run

Episode `p1-e11-live-003`. Raw
`runs/p1_e11_portal_activation/e11-live-20260817-002/`. Compact
`runs/history/p1-e11-live-20260817-002/`.

- fresh process 1, reset 1, retry 0
- BEFORE: 14/14 obsidian, 6/6 air, overworld
- queued YES; executed NO; `tested_action_count=0`
- AFTER truth missing; outcome `truth_identity_mismatch`
- 6/6 nether_portal: **NO**
- E11 real reviewed success: **NO**
- `integration_verified`: **NO**
- E12: **NOT STARTED**

Live #1 remains `portal_activation_not_observed`. Live #2 remains the
pre-`addAction` timeout. No second patch round.

## Await-after-tick follow-up (`p1-e11-live-004`)

`patches/minerl/e11-server-thread-marshal-await-after-tick.patch` only
reorders `stepClient` to:

```text
execActions(...)
waitForNextObservation()
awaitPendingFlintAndSteelMarshal()
```

Episode `p1-e11-live-004`. Compact
`runs/history/p1-e11-live-20260817-003/`. Stack is
`stepClient(EnvServer.java:772)` = await after the observation wait.

Queued YES. `processRightClickBlock` still never ran. One
`waitForNextObservation` tick did **not** drain `server.execute` on the
paused integrated server. `tested_action_count=0`. Outcome
`truth_identity_mismatch`. 6/6 nether_portal: **NO**. No second patch
round.

Run JAR SHA-256:

`fc2a36c36519b981444974848447be04a8393908528cdd179e81bc7f66efb1a2`

Production restored afterward to `836cb5ac…`.

## Runtime identity

Run JAR SHA-256 (nonblocking, timeout):

`286e496396b65856bc13ec45034b4b58056bfe5cbf7d47648d606edacbc3c71b`

Production restored after failure:

`836cb5ac6f89edca3cec255dd895e791212b04794d3349eb13a1b2b313416b6f`

Launcher unchanged:

`7e15699c0d0aea517f87680eb5d760d02519d9744285fa0d348f799e2ed77183`
