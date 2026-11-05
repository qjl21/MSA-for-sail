# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))


def is_ppu15():
    """True on PPU0015 (890 / M890), False on PPU0010 (810E).

    Matches the arch resolution in fmha_ppu.jit: the torch compat layer reports
    sm_89 for PPU1.5 and sm_80 for PPU1.0.
    """
    arch = os.environ.get("MINFER_PPU_ARCH")
    if arch:
        return arch == "ppu_15"
    if not torch.cuda.is_available():
        return False
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return (prop.major, prop.minor) == (8, 9)


def fp8_test_dtypes(*dtypes):
    """Drop float8_e4m3fn on PPU0010; fp8 kernels are not validated there."""
    if is_ppu15():
        return list(dtypes)
    return [d for d in dtypes if d != torch.float8_e4m3fn]
