# SPDX-License-Identifier: Apache-2.0
"""Triton e4m3 (fp8e4nv) below sm_89: Triton refuses the type there ("fp8e4nv not supported in
this architecture") because the GPU has no e4m3 conversion instructions. Loads, stores and
bitcasts of e4m3 bytes need no hardware support, so allow the type everywhere and lower its
value conversions to integer code (as Triton already does for fp8e4b15) and its dots to fp16."""

import triton.language as tl

_PATCHED = False


def _below_sm89(sem) -> bool:
    arch = getattr(sem.builder.options, "arch", None)
    if not isinstance(arch, str) or not arch.startswith("sm"):
        return False
    try:
        return int(arch[2:].rstrip("a")) < 89
    except ValueError:
        return False


def _decode(sem, x, dst_ty):
    """e4m3fn bytes -> float32 (then dst)."""
    i32 = tl.int32
    c = lambda v: sem.scalar_constant(v, i32)  # noqa: E731
    b = sem.cast(sem.bitcast(x, tl.uint8), i32)
    sign = sem.and_(b, c(0x80))
    exp = sem.and_(sem.lshr(b, c(3)), c(0xF))
    man = sem.and_(b, c(7))
    normal = sem.or_(sem.shl(sem.add(exp, c(120), True), c(23)), sem.shl(man, c(20)))
    sub = sem.bitcast(sem.mul(sem.cast(man, tl.float32), sem.scalar_constant(2.0**-9, tl.float32), True), i32)
    mag = sem.where(sem.equal(exp, c(0)), sub, normal)
    f = sem.bitcast(sem.or_(mag, sem.shl(sign, c(24))), tl.float32)
    return sem.cast(f, dst_ty) if dst_ty.scalar != tl.float32 else f


def _encode(sem, x, dst_ty):
    """float -> e4m3fn bytes, round to nearest even, saturating at +-448."""
    i32 = tl.int32
    c = lambda v: sem.scalar_constant(v, i32)  # noqa: E731
    f = sem.cast(x, tl.float32)
    u = sem.bitcast(f, i32)
    sign = sem.and_(sem.lshr(u, c(24)), c(0x80))
    a = sem.bitcast(sem.and_(u, c(0x7FFFFFFF)), tl.float32)
    a = sem.minimum(a, sem.scalar_constant(448.0, tl.float32), tl.PropagateNan.NONE)
    au = sem.bitcast(a, i32)
    # normal: round the 23-bit mantissa to 3 bits (carry into the exponent is the right answer)
    r = sem.add(au, sem.add(c(0x7FFFF), sem.and_(sem.lshr(au, c(20)), c(1)), True), True)
    r = sem.and_(r, c(~0xFFFFF))
    normal = sem.or_(sem.shl(sem.sub(sem.lshr(r, c(23)), c(120), True), c(3)), sem.and_(sem.lshr(r, c(20)), c(7)))
    # subnormal (|x| < 2^-6): m = rne(|x| * 2^9), 8 rolls over to the smallest normal
    magic = sem.scalar_constant(12582912.0, tl.float32)
    s = sem.sub(sem.add(sem.mul(a, sem.scalar_constant(512.0, tl.float32), True), magic, True), magic, True)
    sub = sem.cast(s, i32)
    mag = sem.where(sem.less_than(a, sem.scalar_constant(2.0**-6, tl.float32)), sub, normal)
    out = sem.cast(sem.or_(mag, sign), tl.uint8)
    return sem.bitcast(out, dst_ty)


def patch_triton_fp8e4nv() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    try:
        from triton.backends.nvidia.compiler import CUDAOptions
        from triton.language.semantic import TritonSemantic
    except ImportError:
        return
    if "fp8e4nv" not in CUDAOptions.supported_fp8_dtypes:
        CUDAOptions.supported_fp8_dtypes = tuple(sorted(set(CUDAOptions.supported_fp8_dtypes) | {"fp8e4nv"}))

    cast, dot = TritonSemantic.cast, TritonSemantic.dot

    def _cast(self, input, dst_ty, fp_downcast_rounding=None):
        src = input.type.scalar
        dst = dst_ty.scalar
        if src != dst and _below_sm89(self):
            if src.is_fp8e4nv() and dst.is_floating() and not dst.is_fp8():
                return _decode(self, input, dst)
            if dst.is_fp8e4nv() and src.is_floating():
                if src.is_fp8():
                    input = cast(self, input, tl.float32)
                return _encode(self, input, dst)
            if src.is_fp8e4nv() and dst.is_fp8():
                return cast(self, _decode(self, input, tl.float32), dst_ty, fp_downcast_rounding)
        return cast(self, input, dst_ty, fp_downcast_rounding)

    def _dot(self, lhs, rhs, acc, input_precision, max_num_imprecise_acc, out_dtype):
        if _below_sm89(self) and (lhs.dtype.is_fp8() or rhs.dtype.is_fp8()):
            lhs = self.cast(lhs, tl.float16) if lhs.dtype.is_fp8() else lhs
            rhs = self.cast(rhs, tl.float16) if rhs.dtype.is_fp8() else rhs
            if lhs.dtype != rhs.dtype:
                lhs, rhs = self.cast(lhs, tl.float32), self.cast(rhs, tl.float32)
        return dot(self, lhs, rhs, acc, input_precision, max_num_imprecise_acc, out_dtype)

    TritonSemantic.cast = _cast
    TritonSemantic.dot = _dot
