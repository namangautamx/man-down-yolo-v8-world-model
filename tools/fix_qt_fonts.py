"""
Silence the opencv-python Qt font warning by giving Qt the fonts it looks for.

    python tools/fix_qt_fonts.py

The warning looks like this, repeated ~5 times at startup:

    QFontDatabase: Cannot find font directory .../site-packages/cv2/qt/fonts.
    Note that Qt no longer ships fonts. Deploy some ... or switch to fontconfig.

Cause: opencv-python bundles a Qt build whose QPA plugin expects a `fonts`
directory next to its `plugins` directory. Recent opencv wheels ship the
plugins but not the fonts, so Qt complains once per font it wanted. It is
cosmetic -- windows still open and text still renders -- but it buries real
output.

Things that do NOT fix it, all tested:
    QT_QPA_FONTDIR=/usr/share/fonts/...   ignored by this Qt build
    QT_QPA_PLATFORM=xcb                   unrelated
    QT_LOGGING_RULES=qt.qpa.fonts.warning=false
                                          the message is not logged under a
                                          category, so the rule misses it
Only `QT_LOGGING_RULES=*.warning=false` silences it, and that hides every other
Qt warning too -- a worse trade than just supplying the fonts.

This script symlinks the system fonts into the directory Qt wants. It is
idempotent, and safe to re-run after `pip install --force-reinstall
opencv-python`, which deletes the directory again.
"""

import os
import sys
from pathlib import Path

FONT_SOURCES = [
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/dejavu",
    "/usr/share/fonts/TTF",
    "/usr/share/fonts/truetype/liberation",
    "/usr/local/share/fonts",
    "/Library/Fonts",
]


def cv2_qt_dir():
    try:
        import cv2
    except ImportError:
        sys.exit("cv2 is not importable -- activate the venv first.")
    qt = Path(cv2.__file__).parent / "qt"
    if not qt.is_dir():
        sys.exit(f"No Qt directory at {qt}. This opencv build probably does not "
                 f"use Qt, so the warning cannot be coming from here.")
    return qt


def find_fonts():
    for d in FONT_SOURCES:
        p = Path(d)
        if p.is_dir():
            ttfs = sorted(list(p.glob("*.ttf")) + list(p.glob("*.otf")))
            if ttfs:
                return p, ttfs
    return None, []


def main():
    qt = cv2_qt_dir()
    target = qt / "fonts"
    src_dir, ttfs = find_fonts()

    if not ttfs:
        sys.exit("No system fonts found in any of:\n  " + "\n  ".join(FONT_SOURCES) +
                 "\nInstall some (e.g. `sudo apt install fonts-dejavu-core`) and re-run.")

    target.mkdir(parents=True, exist_ok=True)
    linked = skipped = 0
    for ttf in ttfs:
        dest = target / ttf.name
        if dest.is_symlink() or dest.exists():
            skipped += 1
            continue
        try:
            os.symlink(ttf, dest)
            linked += 1
        except OSError as e:
            print(f"  could not link {ttf.name}: {e}")

    print(f"source : {src_dir}")
    print(f"target : {target}")
    print(f"linked {linked} font(s), {skipped} already present "
          f"({len(list(target.iterdir()))} total)")
    print("\nRe-run the app -- the QFontDatabase warnings should be gone.")


if __name__ == "__main__":
    main()
