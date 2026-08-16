# P1 Apple Silicon startup reliability root-cause report

Date: 2026-08-16

Scope: P1 MineRL/Minecraft startup infrastructure only. E10-E12 and P2 remain
not started. This report does not promote `integration_verified` or the P1
Hard Gate.

## Observed Facts

- The reviewed calibration contains 20 fresh Python processes: 18 succeeded,
  attempt-006 failed with `minecraft_native_crash`, and attempt-014 failed with
  an unresolved reset error. No validation action ran in either failure.
- The actual Python is `/opt/anaconda3/envs/mc-agent/bin/python` 3.10.20 arm64.
- The actual Java is `/opt/anaconda3/envs/mc-agent/bin/java`, Zulu OpenJDK
  1.8.0_472 (8.90.0.19), Mach-O arm64. The unrelated `/usr/bin/java` is not the
  JVM used by the captured run.
- `org.lwjgl.Version` executed statically from the actual
  `mcprec-6.13.jar` reports `3.3.1 SNAPSHOT`. Every resolved LWJGL Java/native
  artifact found in the Gradle cache is version 3.3.1.
- The fat JAR contains both macOS arm64 and x64 native resources. This is a
  duplicate packaged resource set, but it did not cause an architecture mix in
  attempt-006. The JVM loaded the arm64 extraction at
  `/private/var/folders/51/hgn9x4s96kx9dcdpdq9v98840000gn/T/lwjgljoey/3.3.1-SNAPSHOT/`.
- The loaded `liblwjgl_stb.dylib` is Mach-O arm64. Its SHA-256 is
  `af31cb64fd8e63a8b6ff1121c94a47570ad8c9aaee9a46b1d3147ac6209c861a`,
  exactly matching the fat JAR's
  `macos/arm64/org/lwjgl/stb/liblwjgl_stb.dylib` resource.
- The loaded core (`liblwjgl.dylib`), STB, OpenGL, and OpenAL natives all match
  the fat JAR's 3.3.1 arm64 resources by SHA-256. The OpenAL module's macOS
  native is named `libopenal.dylib`; there is no separately loaded
  `liblwjgl_openal.dylib` in this runtime.
- `launchClient.sh` always invokes
  `java -Xmx4G -XstartOnFirstThread -jar build/libs/mcprec-6.13.jar
  --envPort=...`. Attempt-006 and attempt-014 process evidence contain the same
  `-XstartOnFirstThread` argument. It is a macOS launcher option and therefore
  is not repeated under the hs_err `jvm_args` field.
- Attempt-006's current thread is `Sound engine`; its problematic frame is
  `liblwjgl_stb.dylib+0x4c158`; its Java stack is
  `STBVorbis.nstb_vorbis_decode_frame_pushdata` -> `OggAudioStream.readOgg` ->
  `SoundSource.tick` -> `ChannelManager.tick` -> `SoundEngineExecutor.run`.
  The signal is SIGSEGV with a null fault address.
- Minecraft's `SoundEngine.load()` initializes OpenAL and unconditionally
  preloads configured sounds. This preload still occurs when master volume is
  zero. MineRL's later `EnvServer` call to `setSoundLevel(MASTER, 0)` therefore
  cannot prevent STB initialization/decoding.
- Attempt-014 has no hs_err log, no SIGSEGV/STB evidence, and an empty MineRL
  runtime log. Its watcher records that the launch child exited after about 20
  seconds. The historical runner discarded the traceback, so the exact Python
  call site cannot be recovered from existing evidence.
- The phrase `a bytes-like object is required, not 'NoneType'` is produced when
  MineRL calls `struct.unpack` on a `None` reply returned by
  `comms.recv_message` after the Minecraft/Malmo socket closes. Attempt-014's
  own traceback is unavailable. However, the retained E8 and E9 failures with
  the identical error preserve this inner traceback:

  ```text
  MineRLEnvironmentBackend.reset -> self._env.reset()
  minerl.env._singleagent.reset -> super().reset()
  minerl.env._multiagent.reset:446 -> self._send_mission(...)
  minerl.env._multiagent._send_mission:606 -> struct.unpack("!I", reply)
  TypeError: a bytes-like object is required, not 'NoneType'
  ```

  This makes a missing mission-init reply the best-supported call-site
  hypothesis for attempt-014, but it is not direct proof about that attempt.

The complete read-only runtime inventory is stored at
`runs/p1_startup_diagnostics/native-runtime-20260816-attempt-006-v2.json`. The
historical attempt evidence was not modified.

## Dependency chain

1. `MineRLEnvironmentBackend._default_env_factory` builds `PortalA0EnvSpec` and
   calls MineRL's `EnvSpec.make()`.
2. MineRL allocates a `MinecraftInstance` and starts MCP-Reborn's
   `launchClient.sh` with an environment port.
3. The launcher starts Minecraft 1.16.5 / MCP `20210115.111550` from the shaded
   `mcprec-6.13.jar`.
4. `MCP-Reborn/build.gradle` pins the LWJGL core, GLFW, jemalloc, OpenAL,
   OpenGL, STB, and tinyfd modules and macOS arm64 natives to 3.3.1.
5. LWJGL selects the fat JAR's macOS arm64 resources and extracts them under
   the per-user temporary `lwjgljoey/3.3.1-SNAPSHOT` directory. The hs_err
   dynamic-library table proves which extracted files the JVM loaded.
6. Minecraft's client resource reload calls `SoundEngine.load()`, which starts
   OpenAL and sends OGG data through LWJGL STBVorbis on the `Sound engine`
   executor.

## Root Cause

Confirmed immediate root cause for attempt-006: Minecraft's client sound path
entered LWJGL 3.3.1's arm64 STBVorbis decoder and the native decoder segfaulted
on the `Sound engine` thread while processing OGG data. This is not an x86_64
native-selection failure, a Java-architecture mismatch, or a missing
`-XstartOnFirstThread` failure.

The lower-level reason why this specific STB invocation intermittently reaches
an invalid/null native state is not proven by the hs_err report alone. No
dependency-version change is justified by the current evidence.

Attempt-014 is a distinct observed failure path. Its immediate Python symptom
is a missing Malmo protocol reply; matching retained tracebacks point to the
mission-init reply, but why its JVM exited is not recoverable. It may share an
earlier JVM-exit trigger with attempt-006, but there is no native-crash evidence
to support that conclusion.

## Fix

`patches/minerl/disable-client-audio.patch` is a narrowly scoped, property-gated
MCP-Reborn patch. The launcher passes
`-Dobsidianlink.disableClientAudio=true`; `SoundEngine.load()` checks that
property before OpenAL initialization or STB preload. Audio is neither an
Agent-visible observation nor evaluator truth, so this does not change
benchmark capability semantics. It changes no dependency version.

The patch applies cleanly to the current vendor source, but has not been
applied to `vendor/minerl`, compiled, installed, or run. Applying the Java part
requires an authorized Gradle rebuild:

```bash
git -C vendor/minerl apply --directory=minerl/MCP-Reborn \
  /absolute/path/to/patches/minerl/disable-client-audio.patch
cd vendor/minerl/minerl/MCP-Reborn
./gradlew shadowJar
```

The expected rebuilt artifact is
`vendor/minerl/minerl/MCP-Reborn/build/libs/mcprec-6.13.jar`; the patched
`launchClient.sh` must be packaged alongside it in the pinned MineRL 1.0.2
arm64 artifact before the `mc-agent` runtime can use the fix. This step is
`NEEDS_GRADLE_REBUILD_AUTHORIZATION` and requires a separately auditable
deployment into the frozen Conda environment.

The reliability evidence path now also captures full reset tracebacks,
exception chains, child PID, Java PID, environment port, signal, problematic
frame, thread name, and native library. This instrumentation does not alter
normal reset or retry behavior; reliability measurement remains
`max_reset_attempts = 1`.

## Remaining Risk

- Attempt-006 has a targeted mitigation prepared, but it is not active until
  the patched JAR/launcher are rebuilt and installed.
- Attempt-014 remains independently unresolved. A recurrence with the enhanced
  evidence will identify the exact MineRL line and protocol phase.
- Disabling an unused client subsystem avoids the confirmed crashing path but
  does not prove the underlying LWJGL/STB defect is fixed.
- No real Minecraft smoke test or 20-process rerun was performed in this task.
- `process_release_proven` remains false by design: descendant tracking found
  no residual process in the prior run, but cannot prove that nothing escaped
  or was reparented before inspection.

After authorized build/install, the independent commands are:

```bash
/opt/anaconda3/bin/conda run -n mc-agent python \
  scripts/run_p1_startup_reliability.py --episodes 1 --timeout-seconds 600

/opt/anaconda3/bin/conda run -n mc-agent python \
  scripts/run_p1_startup_reliability.py --episodes 20 --timeout-seconds 600
```

The one-episode smoke and the 20-process calibration each require explicit real
MineRL/Minecraft authorization. P1 remains not passed and E10 remains not
started until the independent evidence is reviewed.
