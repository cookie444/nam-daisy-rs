#!/usr/bin/env python3
"""Prefix a `.namb` container with its length for QSPI staging.

The firmware expects the model at QSPI offset 0 as:

    u32 little-endian byte length L
    L bytes of `.namb` container

Usage:
  python stage_model.py model.namb staged.bin

Then flash `staged.bin` to QSPI offset 0, e.g. with:
  dfu-util -a 0 -s 0x90040000:leave -D staged.bin
or via `probe-rs download --base-address 0x90000000 staged.bin`.
"""

import struct
import sys


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]
    blob = open(src, "rb").read()
    with open(dst, "wb") as f:
        f.write(struct.pack("<I", len(blob)))
        f.write(blob)
    print(f"staged {len(blob)} bytes of .namb -> {dst} ({len(blob) + 4} bytes)")


if __name__ == "__main__":
    main()
