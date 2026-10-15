from __future__ import annotations
import csv
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any


def run_cmd(args: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        cp = subprocess.run(args, cwd=str(cwd) if cwd else None, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"找不到可执行程序：{args[0]}。请安装并确保它在 PATH 中。") from exc
    if check and cp.returncode != 0:
        detail = (cp.stderr or cp.stdout or "").strip()
        raise RuntimeError(f"命令失败（exit={cp.returncode}）：{' '.join(args)}\n{detail[-4000:]}")
    return cp


def tool_version(program: str) -> str:
    cp = run_cmd([program, "-version" if program in ("ffmpeg", "ffprobe") else "--version"], check=False)
    lines = (cp.stdout or cp.stderr or "").splitlines()
    return lines[0].strip() if lines else f"{program}: version unavailable"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_fraction(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s in {"N/A", "0/0"}:
        return None
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            denom = float(b)
            return float(a) / denom if denom else None
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def optional_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() in {"", "N/A"}:
            return None
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def require_ffmpeg() -> None:
    missing = [p for p in ("ffmpeg", "ffprobe") if not shutil.which(p)]
    if missing:
        raise RuntimeError(f"缺少必要工具：{', '.join(missing)}。请安装 FFmpeg 并将可执行文件加入 PATH。")
