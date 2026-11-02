# Temporarily do this to avoid changing all imports in the repo
from sglang.srt.utils.common import *
from sglang.srt.utils.network import is_port_available  # noqa: F401

from sglang.srt.utils.sm80_fp8 import patch_triton_fp8e4nv

patch_triton_fp8e4nv()
