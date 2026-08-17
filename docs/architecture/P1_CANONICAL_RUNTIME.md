# P1 canonical MineRL runtime

Date: 2026-08-17

Status: canonical deployed for E0–E11; authorized E12 fixture JAR is separate and excluded from this allowlist.

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

`e12-drawing-decorator-portal.patch` exists for the authorized E12 fixture JAR
and is excluded from this canonical allowlist. Canonical live JARs still reject
portal DrawBlocks. The authorized E12 JAR SHA-256 is
`f459c36b7aaacd7e5f98ff9bbe001f1d54e77b73740537c24d5c5540290d36f4`.

The E11 completion-barrier runtime is staged from that canonical baseline and
adds only `p1-e11-action-completion-barrier.patch`. It does not use
`mcp_patch.diff`, any E11 marshal or diagnostic patch, the paused-executor
patch, or E12 code. Deployed JAR SHA-256:
`6b5705e49220f5af33b5b0d06f7c162afef501a849d54cf57b242933bfd3ef72`.

## Semantic diff

Against canonical baseline (`684c20ec…`), executable changes are only:

- `com/minerl/multiagent/env/EnvServer.class`;
- `net/minecraft/client/ReplaySender.class`;
- `net/minecraft/network/play/ServerPlayNetHandler.class`;
- `version.properties`.

`EnvServer$1` and `ReplaySender$Mode` differ only in compiler line-number
metadata. SoundEngine, AbstractFireBlock, NetherPortalBlock, PortalSize,
FlintAndSteelItem, Entity, and ServerPlayerEntity are byte-identical. Forbidden
marshal/diagnostic/E12 markers are absent.

## Real E11 result

Episode `p1-e11-completion-barrier-20260817-004` used one fresh process, one
reset, one accepted `use_item(flint_and_steel)`, and zero retry.

- before: 14/14 obsidian, 6/6 air, 0 portal, overworld,
  `truth_missing_count=0`;
- execution: MineRL `use=1` → EnvServer → ReplaySender → normal Minecraft
  client right-click on the Render thread;
- after: 6/6 nether portal, `truth_missing_count=0`;
- outcome: `portal_activation_ok`.

The barrier arms only for the exact E11 flint-and-steel action. While it is
pending, ReplaySender returns from its empty-queue wait so the normal integrated
server tick can handle the client packet. `ServerPlayNetHandler` acknowledges
only after `processTryUseItemOnBlock` has completed vanilla interaction; the
EnvServer condition then releases evaluator after-truth. This is an action
completion condition, not an observation-window extension or sleep.

## Authorized E12 fixture JAR

The E12 fixture runtime is staged from the same canonical baseline and adds only
`e12-drawing-decorator-portal.patch`. It is not in `CANONICAL_PATCHES` and does
not include the E11 completion barrier. Against canonical `684c20ec…`, semantic
changes are only `EnvServer.class` and `version.properties`. `EnvServer` maps
Malmo `portal` DrawBlocks to `Blocks.NETHER_PORTAL` and still rejects fire and
end portal. PortalSize, NetherPortalBlock, FlintAndSteelItem, Entity,
ServerPlayerEntity, ReplaySender, ServerPlayNetHandler, IntegratedServer, and
SoundEngine are byte-identical.

Episode `p1-e12-dimension-transition-20260817-001` used one fresh process, one
reset, one accepted `move(forward=1, duration_ticks=8)`, and zero retry.
Evaluator-only before dimension was `minecraft:overworld`; after was
`minecraft:the_nether`; `truth_missing_count=0`; outcome
`dimension_transition_ok`. This E12 fixture JAR is the currently deployed
production `mcprec-6.13.jar`; the E11 completion-barrier JAR remains at backup
`6b5705e4…`. This does not change canonical runtime policy and does not set
`integration_verified`.
