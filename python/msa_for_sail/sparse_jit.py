# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""PPU-side JIT compilation entry for sparse_fmha kernels.

This module mirrors ``jit.py`` but is dedicated to the CuTe-free sparse
backend. All four kernels (build_k2q_csr, sparse_prepare_schedule,
sparse_fmha_fwd, sparse_fmha_combine) use the tvm_ffi/ninja route with the
hgcc device compiler, the sailify compat headers and the PPU codegen flags.

sparse_fmha_fwd is compiled per variant (like jit.py): each (kQpk, dtype-combo)
pair is rendered from csrc/sparse_fmha_fwd_variant.cu.jinja into its own
translation unit, cached under sparse_fmha_fwd_<variant>_<hash>/, and exported
as sparse_fmha_fwd_<variant>. ``compile_sparse_fmha_fwd`` prewarms all 6
variants through one parallel ninja invocation.
"""

import argparse
import hashlib
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .jit import (
    _ACTLIZE_INCLUDE,
    _gencode_flag,
    _get_compiler,
    _get_hgcc_flags,
    _get_link_libs,
    CACHE_BASE,
    get_ppu_arch,
)

_PACKAGE_DIR = Path(__file__).resolve().parent
_CSRC_DIR = _PACKAGE_DIR / "csrc"

_BUILD_K2Q_CSR_SRC = _CSRC_DIR / "build_k2q_csr.cu"

_build_k2q_csr_module = None
_build_k2q_csr_lock = threading.Lock()


def get_build_k2q_csr_module():
    """Get the JIT-compiled build_k2q_csr module.

    Thread-safe; cached after the first tvm_ffi load.
    """
    global _build_k2q_csr_module
    if _build_k2q_csr_module is not None:
        return _build_k2q_csr_module

    with _build_k2q_csr_lock:
        if _build_k2q_csr_module is not None:
            return _build_k2q_csr_module

        if not _BUILD_K2Q_CSR_SRC.exists():
            raise RuntimeError(f"build_k2q_csr source not found: {_BUILD_K2Q_CSR_SRC}")

        _do_compile_tvm_ffi(_BUILD_K2Q_CSR_SRC, "build_k2q_csr", [])
        _build_k2q_csr_module = _load_tvm_ffi_module(
            "build_k2q_csr", _BUILD_K2Q_CSR_SRC, []
        )
        return _build_k2q_csr_module


def compile_build_k2q_csr():
    """Compile (but do not call) the build_k2q_csr module."""
    get_build_k2q_csr_module()


_SPARSE_PREPARE_SCHEDULE_SRC = _CSRC_DIR / "sparse_prepare_schedule.cu"
_SPARSE_PREPARE_SCHEDULE_HEADERS = ["sparse_prepare_schedule.cuh"]
_SPARSE_PREPARE_SCHEDULE_MODULE = None
_SPARSE_PREPARE_SCHEDULE_LOCK = threading.Lock()


def get_sparse_prepare_schedule_module():
    """Get the JIT-compiled sparse_prepare_schedule module."""
    global _SPARSE_PREPARE_SCHEDULE_MODULE
    if _SPARSE_PREPARE_SCHEDULE_MODULE is not None:
        return _SPARSE_PREPARE_SCHEDULE_MODULE

    with _SPARSE_PREPARE_SCHEDULE_LOCK:
        if _SPARSE_PREPARE_SCHEDULE_MODULE is not None:
            return _SPARSE_PREPARE_SCHEDULE_MODULE
        _do_compile_tvm_ffi(
            _SPARSE_PREPARE_SCHEDULE_SRC,
            "sparse_prepare_schedule",
            _SPARSE_PREPARE_SCHEDULE_HEADERS,
        )
        _SPARSE_PREPARE_SCHEDULE_MODULE = _load_tvm_ffi_module(
            "sparse_prepare_schedule",
            _SPARSE_PREPARE_SCHEDULE_SRC,
            _SPARSE_PREPARE_SCHEDULE_HEADERS,
        )
        return _SPARSE_PREPARE_SCHEDULE_MODULE


def compile_sparse_prepare_schedule():
    """Compile (but do not call) the sparse_prepare_schedule module."""
    get_sparse_prepare_schedule_module()


_COMBINE_SRC = _CSRC_DIR / "sparse_fmha_combine.cu"

_COMBINE_MODULE = None
_COMBINE_LOCK = threading.Lock()

_FWD_VARIANT_TEMPLATE = _CSRC_DIR / "sparse_fmha_fwd_variant.cu.jinja"
_IS_PPU15 = get_ppu_arch() == "ppu_15"
_FWD_QPK_SPECS = (
    [(1, "qpk1"), (8, "qpk8"), (16, "qpk16")] if _IS_PPU15 else [(1, "qpk1")]
)
_FWD_DTYPE_TAG = (
    {0: "bf16", 1: "fp8"} if _IS_PPU15 else {0: "bf16"}
)
_FWD_VARIANT_SPECS = {
    f"{qpk_tag}_{dtype_tag}": {"kQpk": k_qpk, "kDtypeKey": dkey}
    for k_qpk, qpk_tag in _FWD_QPK_SPECS
    for dkey, dtype_tag in sorted(_FWD_DTYPE_TAG.items())
}

_FWD_HEADERS = [
    "sparse_fmha_fwd_entry.cuh",
    "sparse_fmha_fwd_impl.cuh",
    "sparse_fmha_fwd.cuh",
    "sparse_common.cuh",
    "fmha_common.hpp",
    "vec_dtypes.cuh",
    "acc_vreg_fraga.h",
]
_COMBINE_HEADERS = [
    "sparse_fmha_combine.cuh",
    "sparse_common.cuh",
]

_PPU_HASH_ONLY_HEADERS = [
    _ACTLIZE_INCLUDE / "ppu_include.hpp",
    _ACTLIZE_INCLUDE / "cute" / "arch" / "copy_aiu_base.hpp",
    _ACTLIZE_INCLUDE / "cute" / "arch" / "copy_ppu0010_aiu.hpp",
    _ACTLIZE_INCLUDE / "cute" / "atom" / "copy_traits_ppu0010_aiu.hpp",
    _ACTLIZE_INCLUDE / "cute" / "arch" / "mma_ppu0010.hpp",
    _ACTLIZE_INCLUDE / "cute" / "atom" / "mma_traits_ppu0010.hpp",
    _ACTLIZE_INCLUDE / "cute" / "arch" / "copy_ppu0015_aiu.hpp",
    _ACTLIZE_INCLUDE / "cute" / "atom" / "mma_traits_ppu0015.hpp",
]

_FWD_TEMPLATE = None
_FWD_VARIANT_MODULES = {}
_FWD_VARIANT_LOCKS = {}
_FWD_VARIANT_LOCKS_GUARD = threading.Lock()


def sparse_fwd_variant_name(qhead_per_kv, dispatch_key):
    """Variant tag for a (qhead_per_kv, dispatch key) request, e.g. "qpk8_fp8".

    qhead_per_kv 16/8 map to the packed M-dim variants; anything else falls to
    the generic head-loop variant "qpk1". On PPU0010 only the bf16 qpk1
    variant exists.
    """
    if dispatch_key not in _FWD_DTYPE_TAG:
        if not _IS_PPU15:
            raise ValueError(
                f"sparse fwd dispatch key {dispatch_key} is unavailable on "
                "PPU0010; fp8 sparse FMHA requires PPU0015"
            )
        raise ValueError(f"invalid sparse fwd dispatch key: {dispatch_key}")
    qpk_tag = (
        {16: "qpk16", 8: "qpk8"}.get(qhead_per_kv, "qpk1") if _IS_PPU15 else "qpk1"
    )
    return f"{qpk_tag}_{_FWD_DTYPE_TAG[dispatch_key]}"


def list_sparse_fwd_variants():
    """All fwd variant tags in deterministic order."""
    return tuple(_FWD_VARIANT_SPECS)


def _source_content_hash(cu_src: Path, headers: list[str]) -> str:
    """SHA-256 of the .cu source and all included headers.

    Bumping the hash invalidates the JIT cache when kernel source or dtype
    configuration changes, preventing stale .so reuse after edits.
    """
    h = hashlib.sha256()
    for path in [cu_src] + [_CSRC_DIR / "include" / name for name in headers] + [_CSRC_DIR / "tvm_ffi_utils.h"]:
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _do_compile_tvm_ffi(cu_src, cache_name, headers):
    """Generic compile: copy .cu + headers to cache, write ninja, compile .so."""
    src_hash = _source_content_hash(cu_src, headers)
    cache_dir = CACHE_BASE / f"{cache_name}_{src_hash}"
    so_path = cache_dir / f"{cache_name}.so"

    if so_path.exists():
        return

    logger = __import__("logging").getLogger(__name__)
    logger.info(f"JIT compiling {cache_name} module")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cu_name = cu_src.name
    shutil.copy2(cu_src, cache_dir / cu_name)

    for name in headers:
        src = _CSRC_DIR / "include" / name
        dst = cache_dir / name
        if not dst.exists() or dst.read_text() != src.read_text():
            shutil.copy2(src, dst)

    utils_src = _CSRC_DIR / "tvm_ffi_utils.h"
    utils_dst = cache_dir / "tvm_ffi_utils.h"
    if not utils_dst.exists() or utils_dst.read_text() != utils_src.read_text():
        shutil.copy2(utils_src, utils_dst)

    hgcc = _get_compiler()
    link_libs = _get_link_libs()

    obj = cache_dir / f"{cache_name}.o"
    hgcc_flags = _get_hgcc_flags(cache_dir, fmha=False)

    ninja_content = f"""ninja_required_version = 1.5

hgcc = {hgcc}
hgcc_flags = {hgcc_flags}
link_libs = {link_libs}

rule hgcc_compile
  command = $hgcc $hgcc_flags -c $in -o $out
  description = Compiling $in

rule hgcc_link
  command = $hgcc -shared $in -o $out $link_libs
  description = Linking $out

build {obj}: hgcc_compile {cache_dir / cu_name}
build {so_path}: hgcc_link {obj}
"""
    (cache_dir / "build.ninja").write_text(ninja_content)

    result = subprocess.run(
        ["ninja", "-j1"],
        cwd=str(cache_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{cache_name} compilation failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


def _load_tvm_ffi_module(cache_name, cu_src, headers):
    import tvm_ffi
    src_hash = _source_content_hash(cu_src, headers)
    so_path = CACHE_BASE / f"{cache_name}_{src_hash}" / f"{cache_name}.so"
    return tvm_ffi.load_module(str(so_path))


def _fwd_template():
    global _FWD_TEMPLATE
    if _FWD_TEMPLATE is None:
        import jinja2
        _FWD_TEMPLATE = jinja2.Template(
            _FWD_VARIANT_TEMPLATE.read_text(), keep_trailing_newline=True,
        )
    return _FWD_TEMPLATE


def _fwd_variant_hash(variant: str) -> str:
    """SHA-256 over the template, copied headers, hash-only ppu headers,
    variant tag and gencode arch.  Bumping the hash invalidates the variant
    cache when any of those change."""
    h = hashlib.sha256()
    for path in (
        [_FWD_VARIANT_TEMPLATE]
        + [_CSRC_DIR / "include" / name for name in _FWD_HEADERS]
        + [_CSRC_DIR / "tvm_ffi_utils.h"]
        + _PPU_HASH_ONLY_HEADERS
    ):
        h.update(path.read_bytes())
    h.update(variant.encode())
    h.update(_gencode_flag().encode())
    return h.hexdigest()[:16]


def _fwd_variant_cache_dir(variant: str) -> Path:
    return CACHE_BASE / f"sparse_fmha_fwd_{variant}_{_fwd_variant_hash(variant)}"


def _materialize_fwd_variant(variant: str):
    """Render the variant .cu and copy headers into its cache dir.

    Returns (cache_dir, so_path, needs_build).  Rendered/copied files are
    written only on content change so mtimes survive and ninja stays a no-op
    across prewarm reruns.  No cross-process locking: concurrent JIT compiles
    of the same variant in separate processes race exactly as before; run
    ``--compile-only sparse_fmha_fwd`` first on cold machines.
    """
    spec = _FWD_VARIANT_SPECS[variant]
    cache_dir = _fwd_variant_cache_dir(variant)
    so_path = cache_dir / f"sparse_fmha_fwd_{variant}.so"
    if so_path.exists():
        return cache_dir, so_path, False

    logger = __import__("logging").getLogger(__name__)
    logger.info(f"JIT materializing sparse_fmha_fwd_{variant}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cu_path = cache_dir / f"sparse_fmha_fwd_{variant}.cu"
    rendered = _fwd_template().render(
        variant_name=variant,
        entry_name=f"sparse_fmha_fwd_{variant}",
        wrapper_name=f"sparse_fmha_fwd_forward_{variant}",
        kQpk=spec["kQpk"],
        kDtypeKey=spec["kDtypeKey"],
    )
    if not cu_path.exists() or cu_path.read_text() != rendered:
        cu_path.write_text(rendered)

    for name in _FWD_HEADERS:
        src = _CSRC_DIR / "include" / name
        dst = cache_dir / name
        if not dst.exists() or dst.read_text() != src.read_text():
            shutil.copy2(src, dst)

    utils_src = _CSRC_DIR / "tvm_ffi_utils.h"
    utils_dst = cache_dir / "tvm_ffi_utils.h"
    if not utils_dst.exists() or utils_dst.read_text() != utils_src.read_text():
        shutil.copy2(utils_src, utils_dst)

    return cache_dir, so_path, True


def _compile_fwd_variant(variant: str) -> Path:
    """Compile one variant on demand (single-target ninja); returns so_path.

    The returned path comes from _materialize_fwd_variant so a source edit
    landing mid-call cannot split the compile dir and the load dir.
    """
    cache_dir, so_path, needs_build = _materialize_fwd_variant(variant)
    if not needs_build:
        return so_path

    logger = __import__("logging").getLogger(__name__)
    hgcc = _get_compiler()
    cu_name = f"sparse_fmha_fwd_{variant}.cu"
    obj = cache_dir / f"sparse_fmha_fwd_{variant}.o"
    hgcc_flags = _get_hgcc_flags(cache_dir, fmha=False)

    ninja_content = f"""ninja_required_version = 1.5

hgcc = {hgcc}
hgcc_flags = {hgcc_flags}
link_libs = {_get_link_libs()}

rule hgcc_compile
  command = $hgcc $hgcc_flags -c $in -o $out
  description = Compiling $in

rule hgcc_link
  command = $hgcc -shared $in -o $out $link_libs
  description = Linking $out

build {obj}: hgcc_compile {cache_dir / cu_name}
build {so_path}: hgcc_link {obj}
"""
    (cache_dir / "build.ninja").write_text(ninja_content)

    t0 = time.perf_counter()
    result = subprocess.run(
        ["ninja", "-j1"],
        cwd=str(cache_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"sparse_fmha_fwd_{variant} compilation failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    logger.info(f"JIT compiled sparse_fmha_fwd_{variant} in {time.perf_counter() - t0:.1f}s")
    return so_path


def _fwd_variant_lock(variant: str):
    with _FWD_VARIANT_LOCKS_GUARD:
        lock = _FWD_VARIANT_LOCKS.get(variant)
        if lock is None:
            lock = _FWD_VARIANT_LOCKS[variant] = threading.Lock()
        return lock


def get_sparse_fmha_fwd_module(variant_name: str):
    """Get the JIT-compiled sparse_fmha_fwd module for one variant.

    variant_name: tag from sparse_fwd_variant_name(), e.g. "qpk8_fp8".  The
    module exports sparse_fmha_fwd_<variant_name>.  Different variants compile
    concurrently; the same variant serializes on its own lock.
    """
    if variant_name not in _FWD_VARIANT_SPECS:
        raise ValueError(f"unknown sparse_fmha_fwd variant: {variant_name}")
    module = _FWD_VARIANT_MODULES.get(variant_name)
    if module is not None:
        return module
    with _fwd_variant_lock(variant_name):
        module = _FWD_VARIANT_MODULES.get(variant_name)
        if module is not None:
            return module
        so_path = _compile_fwd_variant(variant_name)
        import tvm_ffi
        module = tvm_ffi.load_module(str(so_path))
        _FWD_VARIANT_MODULES[variant_name] = module
        return module


def get_sparse_fmha_combine_module():
    global _COMBINE_MODULE
    if _COMBINE_MODULE is not None:
        return _COMBINE_MODULE
    with _COMBINE_LOCK:
        if _COMBINE_MODULE is not None:
            return _COMBINE_MODULE
        _do_compile_tvm_ffi(_COMBINE_SRC, "sparse_fmha_combine", _COMBINE_HEADERS)
        _COMBINE_MODULE = _load_tvm_ffi_module(
            "sparse_fmha_combine", _COMBINE_SRC, _COMBINE_HEADERS
        )
        return _COMBINE_MODULE


def compile_sparse_fmha_fwd():
    """Prewarm all fwd variants through one parallel ninja invocation."""
    missing = []
    for variant in _FWD_VARIANT_SPECS:
        cache_dir, so_path, needs_build = _materialize_fwd_variant(variant)
        if needs_build:
            missing.append((variant, cache_dir, so_path))
    if not missing:
        return

    logger = __import__("logging").getLogger(__name__)
    prewarm_dir = CACHE_BASE / "sparse_fmha_fwd_prewarm"
    prewarm_dir.mkdir(parents=True, exist_ok=True)
    hgcc = _get_compiler()
    hgcc_flags = _get_hgcc_flags(prewarm_dir, fmha=False)

    edges = []
    for variant, cache_dir, so_path in missing:
        cu = cache_dir / f"sparse_fmha_fwd_{variant}.cu"
        obj = cache_dir / f"sparse_fmha_fwd_{variant}.o"
        edges.append(
            f"build {obj}: hgcc_compile {cu}\n"
            f"  hgcc_flags = $hgcc_flags -I{cache_dir}\n"
            f"build {so_path}: hgcc_link {obj}\n"
        )
    ninja_content = f"""ninja_required_version = 1.5

hgcc = {hgcc}
hgcc_flags = {hgcc_flags}
link_libs = {_get_link_libs()}

rule hgcc_compile
  command = $hgcc $hgcc_flags -c $in -o $out
  description = Compiling $in

rule hgcc_link
  command = $hgcc -shared $in -o $out $link_libs
  description = Linking $out

{''.join(edges)}
"""
    (prewarm_dir / "build.ninja").write_text(ninja_content)

    jobs = int(os.environ.get("MINFER_FMHA_NINJA_JOBS", "0")) or min(8, os.cpu_count() or 1)
    t0 = time.perf_counter()
    result = subprocess.run(["ninja", f"-j{jobs}"], cwd=str(prewarm_dir))
    if result.returncode != 0:
        raise RuntimeError(
            "sparse_fmha_fwd prewarm failed; see ninja output above"
        )
    logger.info(
        f"JIT compiled {len(missing)} fwd variants in "
        f"{time.perf_counter() - t0:.1f}s "
        f"(skipped {len(_FWD_VARIANT_SPECS) - len(missing)} cached)"
    )


def compile_sparse_fmha_combine():
    get_sparse_fmha_combine_module()


def compile_all():
    """Compile all sparse kernels implemented so far."""
    compile_build_k2q_csr()
    compile_sparse_prepare_schedule()
    compile_sparse_fmha_fwd()
    compile_sparse_fmha_combine()


_COMPILE_TARGETS = {
    "build_k2q_csr": compile_build_k2q_csr,
    "sparse_prepare_schedule": compile_sparse_prepare_schedule,
    "sparse_fmha_fwd": compile_sparse_fmha_fwd,
    "sparse_fmha_combine": compile_sparse_fmha_combine,
    "all": compile_all,
}


def _main():
    parser = argparse.ArgumentParser(description="JIT compile sparse_fmha kernels")
    parser.add_argument(
        "--compile-only",
        choices=list(_COMPILE_TARGETS.keys()),
        required=True,
        help="Compile the named target without running tests",
    )
    args = parser.parse_args()
    _COMPILE_TARGETS[args.compile_only]()
    print(f"OK: {args.compile_only} compiled")


if __name__ == "__main__":
    _main()
