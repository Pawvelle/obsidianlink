"""Convert MineRL RGB frames into API image payloads.

Does not change Observation. The environment already stores POV as
``Observation.frame`` (typically ``(H, W, 3) uint8``).
"""

from __future__ import annotations

import base64
import io
from typing import Any


def frame_to_data_url(frame: Any, *, quality: int = 75) -> str:
    """Encode an RGB frame as a JPEG ``data:image/jpeg;base64,...`` URL."""
    image = frame_to_pil(frame)
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=int(quality), optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def frame_to_pil(frame: Any) -> Any:
    """MineRL POV ``ndarray (H, W, 3) uint8`` or an existing PIL image."""
    from PIL import Image

    if isinstance(frame, Image.Image):
        return frame
    shape = getattr(frame, "shape", None)
    if shape is None:
        raise TypeError(
            f"frame must be an RGB ndarray or PIL.Image, got {type(frame).__name__}"
        )
    if len(shape) != 3 or shape[-1] not in (3, 4):
        raise ValueError(f"frame must have shape (H, W, 3) or (H, W, 4); got {shape!r}")
    return Image.fromarray(frame)


def frame_summary(frame: Any) -> dict[str, Any]:
    """Metadata only. Never includes pixel bytes."""
    if frame is None:
        return {"present": False}
    shape = getattr(frame, "shape", None)
    dtype = getattr(frame, "dtype", None)
    mean = None
    try:
        mean = float(getattr(frame, "mean")())
    except Exception:
        mean = None
    return {
        "present": True,
        "shape": list(shape) if shape is not None else None,
        "dtype": str(dtype) if dtype is not None else type(frame).__name__,
        "mean": mean,
    }


__all__ = ["frame_summary", "frame_to_data_url", "frame_to_pil"]
