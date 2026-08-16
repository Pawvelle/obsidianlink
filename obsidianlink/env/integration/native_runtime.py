"""Read-only MineRL/LWJGL runtime inspection for P1 diagnostics.

This module never imports MineRL and never launches Minecraft.  It inspects
the installed MCP-Reborn fat JAR, Java executable, extracted LWJGL natives,
and (optionally) a captured ``hs_err_pid`` report.
"""

from __future__ import annotations

import hashlib
import importlib.util
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


NATIVE_NAMES = frozenset(
    {
        "libglfw.dylib",
        "libjemalloc.dylib",
        "liblwjgl.dylib",
        "liblwjgl_opengl.dylib",
        "liblwjgl_stb.dylib",
        "liblwjgl_openal.dylib",
        "libopenal.dylib",
    }
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_minerl_path() -> Path:
    spec = importlib.util.find_spec("minerl")
    if spec is None or spec.origin is None:
        raise RuntimeError("MineRL is not installed in the active Python environment")
    return Path(spec.origin).resolve().parent


def select_java_executable() -> Path:
    environment_java = Path(sys.prefix) / "bin" / "java"
    if environment_java.is_file():
        return environment_java.resolve()
    discovered = shutil.which("java")
    if discovered is None:
        raise RuntimeError("java executable not found")
    return Path(discovered).resolve()


def _run(command: Sequence[str], *, timeout: float = 10.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "command": list(command),
            "exit_code": None,
            "stdout": "",
            "stderr": str(error),
        }
    return {
        "command": list(command),
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def inspect_fatjar(path: Path) -> dict[str, Any]:
    resources: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = Path(info.filename).name
            if name not in NATIVE_NAMES:
                continue
            parts = Path(info.filename).parts
            architecture = None
            if len(parts) >= 2 and parts[0] == "macos":
                architecture = parts[1]
            data = archive.read(info)
            resources.append(
                {
                    "entry": info.filename,
                    "name": name,
                    "architecture_classifier": architecture,
                    "size_bytes": info.file_size,
                    "sha256": _sha256_bytes(data),
                }
            )
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "native_resources": sorted(resources, key=lambda item: item["entry"]),
    }


def parse_loaded_native_paths(text: str) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for raw in re.findall(r"(/[^\n\r\t]+?\.dylib)(?:\s|$)", text):
        path = Path(raw.strip()).resolve()
        if path.name in NATIVE_NAMES and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _inspect_native(path: Path, *, source: str) -> dict[str, Any]:
    file_result = _run(["file", str(path)])
    otool_result = _run(["otool", "-L", str(path)])
    return {
        "path": str(path.resolve()),
        "name": path.name,
        "source": source,
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": _sha256_file(path) if path.is_file() else None,
        "file": file_result,
        "otool": otool_result,
    }


def _find_extracted_natives(temp_root: Path) -> list[Path]:
    if not temp_root.is_dir():
        return []
    found: list[Path] = []
    for directory in temp_root.glob("lwjgl*"):
        for name in NATIVE_NAMES:
            found.extend(directory.glob(f"**/{name}"))
    return sorted({path.resolve() for path in found if path.is_file()})


def _find_gradle_lwjgl_jars(gradle_root: Path) -> list[dict[str, str]]:
    module_root = gradle_root / "caches" / "modules-2" / "files-2.1" / "org.lwjgl"
    if not module_root.is_dir():
        return []
    jars: list[dict[str, str]] = []
    for path in module_root.glob("*/*/*/*.jar"):
        relative = path.relative_to(module_root)
        jars.append(
            {
                "module": relative.parts[0],
                "version": relative.parts[1],
                "path": str(path.resolve()),
            }
        )
    return sorted(jars, key=lambda item: item["path"])


def inspect_runtime(
    *,
    minerl_path: Path | None = None,
    java_executable: Path | None = None,
    hs_err_path: Path | None = None,
    temp_root: Path | None = None,
    gradle_root: Path | None = None,
) -> dict[str, Any]:
    minerl_root = (minerl_path or discover_minerl_path()).resolve()
    mcp_root = minerl_root / "MCP-Reborn"
    fatjar = mcp_root / "build" / "libs" / "mcprec-6.13.jar"
    if not fatjar.is_file():
        raise RuntimeError(f"MCP-Reborn fat JAR not found: {fatjar}")
    java = (java_executable or select_java_executable()).resolve()
    java_version = _run([str(java), "-version"])
    java_file = _run(["file", str(java)])
    lwjgl_version = _run([str(java), "-cp", str(fatjar), "org.lwjgl.Version"])
    extracted_paths = _find_extracted_natives(temp_root or Path(tempfile.gettempdir()))
    extracted = [_inspect_native(path, source="lwjgl_temp_extraction") for path in extracted_paths]
    loaded_paths: list[Path] = []
    if hs_err_path is not None:
        loaded_paths = parse_loaded_native_paths(
            hs_err_path.read_text(encoding="utf-8", errors="replace")
        )
    loaded = [_inspect_native(path, source="hs_err_dynamic_library") for path in loaded_paths]
    gradle_jars = _find_gradle_lwjgl_jars(gradle_root or (Path.home() / ".gradle"))
    fatjar_details = inspect_fatjar(fatjar)
    path_counts = Counter(
        [item["name"] for item in extracted]
        + [item["name"] for item in fatjar_details["native_resources"]]
    )
    return {
        "inspection_kind": "static_read_only_no_minecraft_launch",
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "machine": platform.machine(),
        },
        "java": {
            "executable": str(java),
            "version": java_version,
            "file": java_file,
        },
        "minerl": {
            "installation_path": str(minerl_root),
            "mcp_reborn_path": str(mcp_root.resolve()),
        },
        "lwjgl": {
            "reported_version": lwjgl_version,
            "fatjar": fatjar_details,
            "extracted_natives": extracted,
            "loaded_natives_from_hs_err": loaded,
            "gradle_cache_jars": gradle_jars,
            "possible_duplicate_names": {
                name: count for name, count in sorted(path_counts.items()) if count > 1
            },
        },
        "hs_err_path": None if hs_err_path is None else str(hs_err_path.resolve()),
    }
