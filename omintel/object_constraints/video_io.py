"""Real-video decoding via OpenCV, keeping frame indexing explicit.

``decode_window`` decodes exactly ``[start_frame_no, end_frame_no]`` in order
and raises if decoding ends early, so a dropped/truncated frame can never
silently misalign ``frame_no`` to the wrong image downstream.

Note: ``cv2.VideoCapture`` seeking (``CAP_PROP_POS_FRAMES``) is keyframe-based,
so some codecs may land slightly off on the first frame after a seek; the frame
*count* returned here is still exact and the order is guaranteed sequential.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def video_info(video_path: str | Path) -> dict[str, Any]:
    """Report decoder-side width/height/fps/frame_count (not ffprobe's view)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        cap.release()


def decode_window(
    video_path: str | Path,
    start_frame_no: int,
    end_frame_no: int,
) -> list[Any]:
    """Decode frames ``[start_frame_no, end_frame_no]`` inclusive, in order.

    Guarantees ``len(result) == end_frame_no - start_frame_no + 1`` so callers
    can map ``result[i]`` to ``frame_no = start_frame_no + i``; raises
    ``RuntimeError`` if the video can't be opened or decoding ends early.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame_no))
        frames: list[Any] = []
        wanted = int(end_frame_no) - int(start_frame_no) + 1
        for frame_no in range(int(start_frame_no), int(end_frame_no) + 1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(
                    f"decode ended early: expected frame_no {frame_no}, "
                    f"got {len(frames)}/{wanted}"
                )
            frames.append(frame)
        return frames
    finally:
        cap.release()