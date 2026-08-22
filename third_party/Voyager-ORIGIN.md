# Voyager upstream source

This directory contains a source snapshot of
[MineDojo/Voyager](https://github.com/MineDojo/Voyager), downloaded on
2026-08-22 from its `main` branch.

Voyager is distributed under the upstream [MIT License](voyager/LICENSE),
which is retained verbatim in `voyager/LICENSE`.

## Runtime boundary

The upstream implementation is preserved here unchanged as a reference
implementation.  Its `VoyagerEnv` launches a Mineflayer Node.js bridge and a
Fabric 1.19 Minecraft instance; it is not compatible with this repository's
MineDojo 0.1 / Minecraft 1.11.2 runtime.  Do not import `voyager` from the
ObsidianLink production path.  A future adapter may reuse the curriculum,
critic, action-prompt, and skill-library ideas while executing only through
`MineDojoEnvironment`.

The original source has no nested `.git` directory, so it can be tracked as
ordinary source by this repository.  Upstream JavaScript dependencies are not
installed and raw Voyager logs, checkpoints, and `node_modules` must not be
committed.
