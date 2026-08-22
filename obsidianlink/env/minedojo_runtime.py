"""Repair and select the legacy native runtime required by MineDojo 0.1.

MineDojo's bundled Malmo is based on Minecraft 1.11.2.  Its Gradle project
references a JitPack commit that is no longer served, and its LWJGL native
libraries are x86_64-only on macOS.  This module keeps those compatibility
details at the platform boundary rather than leaking them into agent code.
"""

from __future__ import annotations

import hashlib
import os
import platform
import urllib.request
from pathlib import Path

_MIXIN_GROUP = "MixinGradle-dcfaf61"
_MIXIN_ARTIFACT = "MixinGradle"
_MIXIN_VERSION = "dcfaf61"
_MIXIN_JAR_NAME = f"{_MIXIN_ARTIFACT}-{_MIXIN_VERSION}.jar"
_MIXIN_URL = (
    "https://raw.githubusercontent.com/verityw/MixinGradle-dcfaf61/main/"
    f"{_MIXIN_ARTIFACT}/{_MIXIN_VERSION}/{_MIXIN_JAR_NAME}"
)
_MIXIN_SHA256 = "bfdcda4cbf23f28392384818a1417c2d6df687e15ce77530ef96381393102f8a"
_MARKER = "// ObsidianLink MineDojo compatibility: local MixinGradle mirror"
_X64_JAVA_HOME = Path("/opt/anaconda3/envs/mc-agent/opt/zulu8-x64")


def prepare_minedojo_runtime(minedojo_root: Path) -> None:
    """Make MineDojo's bundled Minecraft backend runnable on this machine."""
    _select_macos_x64_java()
    mirror_root = _ensure_mixin_mirror()
    _patch_minecraft_gradle(minedojo_root / "sim" / "Malmo" / "Minecraft", mirror_root)


def _select_macos_x64_java() -> None:
    """Use Rosetta Java 8 for MineDojo's x86_64-only macOS LWJGL binaries."""
    if platform.system() != "Darwin" or platform.machine().lower() not in {
        "arm64",
        "aarch64",
    }:
        return
    java_home = Path(os.environ.get("MINEDOJO_JAVA_HOME", _X64_JAVA_HOME))
    java = java_home / "bin" / "java"
    if not java.is_file():
        raise RuntimeError(
            "MineDojo on Apple Silicon requires an x86_64 Java 8 runtime. "
            f"Expected {java}; set MINEDOJO_JAVA_HOME to an installed x86_64 JDK 8."
        )
    os.environ["JAVA_HOME"] = str(java_home)
    current_path = os.environ.get("PATH", "")
    java_bin = str(java.parent)
    if not current_path.startswith(f"{java_bin}{os.pathsep}"):
        os.environ["PATH"] = f"{java_bin}{os.pathsep}{current_path}"


def _ensure_mixin_mirror() -> Path:
    root = Path(__file__).resolve().parents[2] / "vendor" / "minedojo-runtime"
    jar = root / _MIXIN_GROUP / _MIXIN_ARTIFACT / _MIXIN_VERSION / _MIXIN_JAR_NAME
    if jar.is_file() and _sha256(jar) == _MIXIN_SHA256:
        return root
    jar.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(_MIXIN_URL, timeout=60) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != _MIXIN_SHA256:
        raise RuntimeError("downloaded MineDojo MixinGradle mirror failed SHA-256 verification")
    jar.write_bytes(payload)
    return root


def _patch_minecraft_gradle(minecraft_root: Path, mirror_root: Path) -> None:
    build_file = minecraft_root / "build.gradle"
    if not build_file.is_file():
        raise RuntimeError(f"MineDojo Minecraft build file is missing: {build_file}")
    source = build_file.read_text(encoding="utf-8")
    repository = f"        {_MARKER}\n        maven {{ url '{mirror_root.as_uri()}' }}"
    if _MARKER not in source:
        needle = "        maven { url 'https://jitpack.io' }"
        if needle not in source:
            raise RuntimeError("unsupported MineDojo build.gradle: JitPack repository missing")
        source = source.replace(needle, f"{needle}\n{repository}", 1)
    legacy = "classpath('com.github.SpongePowered:MixinGradle:dcfaf61')"
    replacement = f"classpath('{_MIXIN_GROUP}:{_MIXIN_ARTIFACT}:{_MIXIN_VERSION}')"
    if legacy in source:
        source = source.replace(legacy, replacement)
    elif replacement not in source:
        raise RuntimeError("unsupported MineDojo build.gradle: MixinGradle dependency missing")
    build_file.write_text(source, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["prepare_minedojo_runtime"]
