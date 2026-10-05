#!/usr/bin/env python3
"""Skip Lance imports in the installed stable-worldmodel package.

pylance 12.x wheels execute AVX2 (``vpermq``) while the extension is
loading. The Xeon E3-1270 v2 has AVX but not AVX2, so ``import lance``
is killed with SIGILL and cannot be caught. Push-T and LeWM encoding
do not use Lance. Set SWM_ENABLE_LANCE=1 only on a CPU with AVX2.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = (
    ROOT
    / ".venv"
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
PKG = SITE / "stable_worldmodel" / "data"

FORMATS_INIT = PKG / "formats" / "__init__.py"
DATA_INIT = PKG / "__init__.py"

OLD_FORMATS = "from . import lance  # noqa: F401\n"
NEW_FORMATS = """import os as _os

# AVX2-only wheel: importing lance SIGILLs on pre-Haswell CPUs.
if _os.environ.get("SWM_ENABLE_LANCE") == "1":
    from . import lance  # noqa: F401
"""

OLD_DATA = "from .formats.lance import LanceDataset, LanceWriter\n"
NEW_DATA = """import os as _os

if _os.environ.get("SWM_ENABLE_LANCE") == "1":
    from .formats.lance import LanceDataset, LanceWriter
else:
    LanceDataset = None  # Lance wheels require AVX2; see scripts/make_lance_optional.py
    LanceWriter = None
"""

SPT_DATA = SITE / "stable_pretraining" / "data" / "__init__.py"
OLD_SPT_VIDEO = """try:
    from .video import LanceVideoSegments, build_lance_video_dataset  # noqa: F401

    _VIDEO_AVAILABLE = True
except ImportError:
    _VIDEO_AVAILABLE = False

# Image (Lance-backed) — same cv2-guard rationale as video above.
try:
    from .images import LanceImageDataset, build_lance_image_dataset  # noqa: F401

    _IMAGE_AVAILABLE = True
except ImportError:
    _IMAGE_AVAILABLE = False
"""
NEW_SPT_VIDEO = """import os as _os

# `import lance` SIGILLs (AVX2) on pre-Haswell CPUs. ImportError cannot catch that.
if _os.environ.get("SWM_ENABLE_LANCE") == "1":
    try:
        from .video import LanceVideoSegments, build_lance_video_dataset  # noqa: F401

        _VIDEO_AVAILABLE = True
    except ImportError:
        _VIDEO_AVAILABLE = False

    try:
        from .images import LanceImageDataset, build_lance_image_dataset  # noqa: F401

        _IMAGE_AVAILABLE = True
    except ImportError:
        _IMAGE_AVAILABLE = False
else:
    _VIDEO_AVAILABLE = False
    _IMAGE_AVAILABLE = False
"""

KORNIA_IO = SITE / "kornia" / "io" / "io.py"
OLD_KORNIA = "import kornia_rs\n"
NEW_KORNIA = '''class _LazyKorniaRs:
    """Defer kornia_rs. Its wheel runs BMI2 (rorx) at import and SIGILLs here."""

    def __getattr__(self, name):
        import kornia_rs as _mod

        return getattr(_mod, name)


kornia_rs = _LazyKorniaRs()
'''


def _replace_once(path: Path, old: str, new: str) -> str:
    text = path.read_text()
    if new in text:
        return "already patched"
    if old not in text:
        raise SystemExit(f"{path} does not contain the expected import; package layout changed")
    path.write_text(text.replace(old, new, 1))
    return "patched"


def main() -> None:
    if not FORMATS_INIT.is_file() or not DATA_INIT.is_file():
        raise SystemExit(f"stable_worldmodel data package not found under {PKG}")
    print(FORMATS_INIT, _replace_once(FORMATS_INIT, OLD_FORMATS, NEW_FORMATS))
    print(DATA_INIT, _replace_once(DATA_INIT, OLD_DATA, NEW_DATA))
    print(SPT_DATA, _replace_once(SPT_DATA, OLD_SPT_VIDEO, NEW_SPT_VIDEO))
    print(KORNIA_IO, _replace_once(KORNIA_IO, OLD_KORNIA, NEW_KORNIA))


if __name__ == "__main__":
    main()
