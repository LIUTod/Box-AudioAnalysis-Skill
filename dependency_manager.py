#!/usr/bin/env python3
"""
Runtime dependency bootstrap for generic audio analysis skill.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class DependencyStatus:
    os_name: str
    ffmpeg_ready: bool
    sensevoice_ready: bool
    vector_model_ready: bool
    messages: List[str]


def detect_os() -> str:
    name = platform.system().lower()
    if "windows" in name:
        return "windows"
    if "darwin" in name:
        return "macos"
    if "linux" in name:
        return "linux"
    return "unknown"


def _run(cmd: List[str]) -> bool:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return result.returncode == 0
    except Exception:
        return False


def ensure_ffmpeg() -> bool:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return True

    os_name = detect_os()
    if os_name == "windows":
        if shutil.which("winget"):
            _run(["winget", "install", "-e", "--id", "Gyan.FFmpeg"])
    elif os_name == "macos":
        if shutil.which("brew"):
            _run(["brew", "install", "ffmpeg"])
    elif os_name == "linux":
        if shutil.which("apt"):
            _run(["sudo", "apt", "update"])
            _run(["sudo", "apt", "install", "-y", "ffmpeg"])
        elif shutil.which("dnf"):
            _run(["sudo", "dnf", "install", "-y", "ffmpeg"])

    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def ensure_sensevoice_model(workdir: Path) -> bool:
    # Model id stays fixed, path is no longer a bundled dependency.
    from huggingface_hub import snapshot_download

    runtime_dir = workdir / ".runtime_cache" / "sensevoice" / "SenseVoiceSmall-onnx"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id="haixuantao/SenseVoiceSmall-onnx",
            local_dir=str(runtime_dir),
            local_dir_use_symlinks=False,
        )
        return True
    except Exception:
        return False


def ensure_vector_model():
    # Reuse existing embedding bootstrap logic.
    from knowledge_rag import get_sentence_transformer

    try:
        get_sentence_transformer()
        return True
    except Exception:
        return False


def ensure_runtime_dependencies(workdir: Path) -> DependencyStatus:
    messages: List[str] = []
    os_name = detect_os()
    messages.append(f"Detected OS: {os_name}")

    ff_ok = ensure_ffmpeg()
    messages.append("ffmpeg ready" if ff_ok else "ffmpeg install failed")

    sv_ok = ensure_sensevoice_model(workdir)
    messages.append("sensevoice ready" if sv_ok else "sensevoice download failed")

    vec_ok = ensure_vector_model()
    messages.append("vector model ready" if vec_ok else "vector model download failed")

    return DependencyStatus(
        os_name=os_name,
        ffmpeg_ready=ff_ok,
        sensevoice_ready=sv_ok,
        vector_model_ready=vec_ok,
        messages=messages,
    )
