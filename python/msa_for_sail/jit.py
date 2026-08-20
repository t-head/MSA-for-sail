# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Per-variant lazy JIT compilation for FMHA varlen kernels.

Each FMHA variant (dtype x tile x sparse x page x split_kv x pack_factor) is compiled
independently on first use and cached to ~/.cache/minfer/msa_for_sail/.

To recompile after kernel changes: scripts/clear_fmha_cache.sh
"""

import functools
import itertools
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path

import jinja2

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _detect_ppu_arch():
    """Map the torch-reported compute capability to a PPU arch.

    The torch compat layer reports sm_89 for PPU1.5 devices and sm_80 for
    PPU1.0; anything else falls back to ppu_10 (same rule as DeepGemm).
    """
    try:
        import torch
        prop = torch.cuda.get_device_properties(torch.cuda.current_device())
        return "ppu_15" if (prop.major, prop.minor) == (8, 9) else "ppu_10"
    except Exception:
        return "ppu_15"


_ARCH_FLAGS = {
    "ppu_15": "-arch=ppu_15",
    "ppu_10": "-arch=ppu_10",
}


def get_ppu_arch():
    return os.environ.get("MINFER_PPU_ARCH") or _detect_ppu_arch()


def _gencode_flag():
    """Return the device arch flag for the PPU generation in use."""
    arch = get_ppu_arch()
    try:
        return _ARCH_FLAGS[arch]
    except KeyError:
        raise RuntimeError(
            f"Unknown PPU arch {arch!r}; expected one of {sorted(_ARCH_FLAGS)}"
        ) from None

def _compute_cache_base():
    explicit = os.environ.get("MINFER_FMHA_CACHE_DIR")
    if explicit:
        return Path(explicit)
    base = Path(os.path.expanduser("~/.cache/minfer/msa_for_sail"))
    # Different build configs get separate cache dirs to avoid conflicts
    suffix = ""
    if os.environ.get("GPU_TRACE") is not None:
        suffix += "_gpu_trace"
    if os.environ.get("FMHA_GMEM_CHECK") is not None:
        suffix += "_gmem_check"
    if suffix:
        base = base.parent / (base.name + suffix)
    return base


CACHE_BASE = _compute_cache_base()

# Kernel sources and ACTLIZE headers are shipped inside the package directory
# so that JIT compilation works from both editable and wheel installs.
_PACKAGE_DIR = Path(__file__).resolve().parent
_FMHA_VARLEN_DIR = _PACKAGE_DIR / "csrc"
_ACTLIZE_DIR = _PACKAGE_DIR / "actlize"
_ACTLIZE_INCLUDE = _ACTLIZE_DIR / "include"

_PACK_FACTORS = [1,2,4,6,8,16]
# _PACK_FACTORS = [1, 6]

# DLPack dtype codes (must match tvm_ffi_utils.h encode_dlpack_dtype)
_BFLOAT16_CODE = (4 << 16) | (16 << 8) | 1      # 266241
# fp8-native Q/K: fp8 feeds the TensorCell MMA directly (no dequantization);
# OnlyScore max_score prefill &Sparse, enforced in api.py.
_FLOAT8_E4M3FN_CODE = (12 << 16) | (8 << 8) | 1  # 788481

# dtype_in is both the compute and the gmem storage type: plain bf16 kernels,
# or the fp8 entry which keeps Q/K in fp8 end-to-end (fp8 TensorCell MMA,
# OnlyScore). OnlyScoreBF16 stores max_score as bf16; all other modes float32.
_MSA_SAIL_DISPATCH = [
    ("int64_t dtype_code", [
        (_BFLOAT16_CODE,      {"dtype_in": "ppu_bfloat16",   "cutlass_dtype_out": "cutlass::bfloat16_t"}),
        (_FLOAT8_E4M3FN_CODE, {"dtype_in": "__hg_fp8_e4m3", "cutlass_dtype_out": "cutlass::bfloat16_t"}),
    ]),
    ("int qo_tile_size", [
        ((16, 128),  {"tile_q": "_16",  "tile_kv": "_128", "thread_shape": "_1, _1, _1"}),
        ((64, 128),  {"tile_q": "_64",  "tile_kv": "_128", "thread_shape": "_1, _1, _1"}),
        ((128, 128), {"tile_q": "_128", "tile_kv": "_128", "thread_shape": "_1, _1, _1"}),
        ((16, 64),   {"tile_q": "_16",  "tile_kv": "_64",  "thread_shape": "_1, _1, _1"}),
    ]),
    ("bool single_wg", [
        ("true",  {"single_wg": "true"}),
        ("false", {"single_wg": "false"}),
    ]),
    ("int sparse_mode", [
        (0,    {"sparse_mode": "Sparse"}),
        (1,    {"sparse_mode": "Full"}),
        (2,    {"sparse_mode": "OnlyScore"}),
        (3,    {"sparse_mode": "OnlyScoreBF16"}),
        (None, {"sparse_mode": "Off"}),
    ]),
    ("int page_size", [
        (-1,  {"page_size": -1}),
        (128, {"page_size": 128}),
        # (256, {"page_size": 256}),
    ]),
    ("bool split_kv", [
        ("false", {"is_split_kv": "false"}),
        ("true",  {"is_split_kv": "true"}),
    ]),
    ("int pack_factor", [(i, {"pack_factor": i}) for i in _PACK_FACTORS]),
]

_MSA_SAIL_IMPOSSIBLE = lambda p: (
    (p.get("page_size") == -1 and p.get("sparse_mode") == "Sparse")
    or (p.get("dtype_in") == "__hg_fp8_e4m3"
        and p.get("sparse_mode") not in ("OnlyScore", "OnlyScoreBF16", "Sparse"))
)


def _dlpack_dtype_code(torch_dtype):
    """Encode a torch dtype as a DLPack int64 code."""
    import torch
    _map = {
        torch.float16: (2 << 16) | (16 << 8) | 1,
        torch.bfloat16: (4 << 16) | (16 << 8) | 1,
        torch.float32: (2 << 16) | (32 << 8) | 1,
        torch.float8_e4m3fn: (12 << 16) | (8 << 8) | 1,
        torch.float8_e5m2: (13 << 16) | (8 << 8) | 1,
    }
    return _map[torch_dtype]


def _variant_key_from_runtime(dtype_code, qo_tile_size, kv_tile_size,
                               single_wg, sparse_mode, page_size, split_kv, pack_factor):
    """Compute a variant key string from runtime parameters."""
    dims = _MSA_SAIL_DISPATCH

    def _match_idx(dim_values, runtime_val):
        # Convert Python bool to string to match dispatch table ("true"/"false")
        if isinstance(runtime_val, bool):
            runtime_val = "true" if runtime_val else "false"
        for idx, (match_val, _) in enumerate(dim_values):
            if match_val is None:
                return idx
            if match_val == runtime_val:
                return idx
        return len(dim_values) - 1

    runtime_vals = [dtype_code, (qo_tile_size, kv_tile_size), single_wg,
                    sparse_mode, page_size, split_kv, pack_factor]
    indices = []
    params = {}
    for (_, dim_values), rv in zip(dims, runtime_vals):
        idx = _match_idx(dim_values, rv)
        indices.append(idx)
        _, tparams = dim_values[idx]
        params.update(tparams)

    if _MSA_SAIL_IMPOSSIBLE(params):
        raise ValueError(f"Impossible FMHA variant combination: {params}")

    func_name = "msa_for_sail_" + "_".join(str(i) for i in indices)
    variant_name = "_".join(str(i) for i in indices)
    params["func_name"] = func_name
    params["variant_name"] = variant_name
    return variant_name, params


def _get_tvm_ffi_include():
    """Find TVM-FFI include directory."""
    try:
        import tvm_ffi
        tvm_dir = Path(tvm_ffi.__path__[0])
        inc = tvm_dir / "include"
        if inc.exists():
            return str(inc)
        inc2 = tvm_dir.parent / "include"
        if inc2.exists():
            return str(inc2)
    except ImportError:
        pass
    raise RuntimeError("Cannot find TVM-FFI include directory; install apache-tvm-ffi")


def _get_ppu_sdk_root():
    """PPU SDK root: runtime libraries for the link rule live here.

    The device compiler (bin/hgcc) and the runtime libraries (lib/) live
    under one SDK root in both container types, so this is the same lookup
    as the toolkit home; delegating keeps a single env-var table.
    """
    return _get_toolkit_home()


def _get_toolkit_home():
    """Toolkit root holding the device compiler.

    CUDA-compat containers set CUDA_HOME/CUDA_SDK and use their nvcc; SDK
    containers set PPU_* and use hgcc. The CUDA_* names are conversion-stable
    so the converted tree resolves the compat toolkit unchanged.
    """
    for var in ("CUDA_HOME", "CUDA_SDK", "PPU_SDK", "PPU_HOME", "PPU_PATH"):
        if var in os.environ:
            return os.environ[var]
    if os.path.isfile(os.path.join("/usr/local/PPU_SDK", "bin", "hgcc")):
        return "/usr/local/PPU_SDK"
    raise RuntimeError("Cannot find a device toolkit. Set PPU_SDK or CUDA_HOME.")


def _get_compiler():
    """Return the device compiler path under the toolkit root."""
    compiler = os.path.join(_get_toolkit_home(), "bin", "hgcc")
    if not os.path.isfile(compiler):
        raise RuntimeError(
            f"Device compiler not found: {compiler}. "
            "Set PPU_SDK (SDK container) or CUDA_HOME (CUDA-compat container)."
        )
    return compiler


def _get_link_libs():
    # Every deployment loads the .so through the PPU runtime, so the hggc
    # runtime libraries are the right link target in both containers; this
    # line must stay conversion-stable.
    home = _get_ppu_sdk_root()
    return f"-L{os.path.join(home, 'lib')} -lhggc_wrapper -lhggcrt1 -lhggc"


_ALL_VARIANTS_SO = CACHE_BASE / "_all_variants" / "all_variants.so"

def _get_hgcc_flags(cache_dir, fmha=True):
    tvm_include = _get_tvm_ffi_include()
    fmha_include = str(_FMHA_VARLEN_DIR / "include")
    if not (_ACTLIZE_INCLUDE / "ppu_include.hpp").is_file():
        raise RuntimeError(
            f"ACTLIZE headers not found at {_ACTLIZE_INCLUDE}.\n"
            "The submodule is not checked out; fetch it from the repository root:\n"
            "  git submodule update --init --recursive"
        )
    actlize_include = str(_ACTLIZE_INCLUDE)
    hgcc_flags = [
        "-O3", "-std=c++20",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        _gencode_flag(),
        "-static-global-template-stub=false",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
        "-DFLASHINFER_ENABLE_FP8_E5M2",
        "-DFLASHINFER_ENABLE_FP8_E8M0",
        "-DFLASHINFER_ENABLE_FP4_E2M1",
        "-DFLASHINFER_ENABLE_F16",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        f"-I{fmha_include}",
        f"-I{actlize_include}",
        f"-I{tvm_include}",
        f"-I{cache_dir}",
        "-use_fast_math",
        "-lineinfo",
        "-DNDEBUG",
        "-Xcompiler", "-fPIC",
        "-DUSE_PPU", "-DUSE_AIU=1",
    ]
    # The sailify compat headers are force-included so converted sources
    # resolve; the SDK compiler rejects the nvcc-only -Xptxas/-Xcudafe flags.
    hgcc_flags += [
        f"-I{_FMHA_VARLEN_DIR / '.ppu_compat'}",
        "-include", "compatible_wrapper.h",
    ]
    if os.environ.get("GPU_TRACE") is not None:
        hgcc_flags.append("-DGPU_TRACE_ENABLED")
    hgcc_flags.extend(
        [
            "-mllvm",
            "-ppu-max-vreg-count=256",
            "-mllvm",
            "-ppu-sink-matrix-addr=true",
            "-mllvm",
            "-ppu-max-alloca-byte-size=320",
            "-mllvm",
            "-ppu-sink-async-addr=true",
            "-mllvm",
            "-ppu-sink-load-addr=true",
            "-mllvm",
            "-ppu-sink-store-addr=true",
            "-mllvm",
            "-ppu-alloca-half-ldst-simplify=true",
            "-mllvm",
            "-ppu-volatile-yield=false",
            "-mllvm",
            "-sort-copy-before-coalesce=true",
            "-mllvm",
            "-enable-threadIdx-x-div32-always-uniform=true",
        ]
    )
    # hgcc_flags.extend(   # uncomment it for compilation debug
    #     [
    #         "-save-temps",
    #     ]
    # )
    if os.environ.get("FMHA_GMEM_CHECK") is not None:
        hgcc_flags.append("-DFMHA_GMEM_BOUNDS_CHECK")
    return " ".join(hgcc_flags)

class _VariantWrapper:
    """Wraps a TVM-FFI function to look like a module with .run()."""
    def __init__(self, fn):
        self._fn = fn

    def run(self, *args):
        return self._fn(*args)


class FMHAVariantManager:
    """Manages FMHA variant kernels. Loads from all_variants.so or per-variant .so."""

    def __init__(self):
        self._loaded = {}
        self._lock = threading.Lock()
        self._all_module = None
        self._all_module_checked = False
        self._inst_template = None
        self._run_template = None

    def _load_templates(self):
        if self._inst_template is None:
            with open(_FMHA_VARLEN_DIR / "fmha_sm80_inst.jinja") as f:
                self._inst_template = jinja2.Template(f.read())
            with open(_FMHA_VARLEN_DIR / "fmha_sail_variant_run.cu.jinja") as f:
                self._run_template = jinja2.Template(f.read())

    def _ensure_all_module(self):
        if not self._all_module_checked:
            self._all_module_checked = True
            if _ALL_VARIANTS_SO.exists():
                import tvm_ffi
                self._all_module = tvm_ffi.load_module(str(_ALL_VARIANTS_SO))

    def get_variant(self, dtype_code, qo_tile_size, kv_tile_size,
                    single_wg, sparse_mode, page_size, split_kv, pack_factor):
        variant_name, params = _variant_key_from_runtime(
            dtype_code, qo_tile_size, kv_tile_size,
            single_wg, sparse_mode, page_size, split_kv, pack_factor)

        cached = self._loaded.get(variant_name)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._loaded.get(variant_name)
            if cached is not None:
                return cached

            fn_name = f"run_{variant_name}"

            # Try all_variants.so
            self._ensure_all_module()
            if self._all_module is not None:
                try:
                    fn = getattr(self._all_module, fn_name)
                    self._loaded[variant_name] = _VariantWrapper(fn)
                    return self._loaded[variant_name]
                except AttributeError:
                    pass

            # Try per-variant .so
            cache_dir = CACHE_BASE / variant_name
            so_path = cache_dir / f"{variant_name}.so"
            if not so_path.exists():
                self._compile_only(variant_name, params)

            import tvm_ffi
            module = tvm_ffi.load_module(str(so_path))
            fn = getattr(module, fn_name)
            self._loaded[variant_name] = _VariantWrapper(fn)
            return self._loaded[variant_name]

    def _compile_only(self, variant_name, params):
        """Compile a variant without loading. Safe to call from subprocesses."""
        cache_dir = CACHE_BASE / variant_name
        so_path = cache_dir / f"{variant_name}.so"

        if so_path.exists():
            return

        logger.info(f"JIT compiling FMHA variant: {variant_name}")
        cache_dir.mkdir(parents=True, exist_ok=True)

        self._load_templates()

        inst_cu = cache_dir / f"fmha_sm80_inst_{variant_name}.cu"
        run_cu = cache_dir / f"fmha_sm80_run_{variant_name}.cu"

        inst_cu.write_text(self._inst_template.render(**params))
        run_cu.write_text(self._run_template.render(**params))

        for name in ["fmha_sail_params.h", "msa_profiling_interface.hpp",
                     "tvm_ffi_utils.h", "gmem_bounds_check.h"]:
            src = _FMHA_VARLEN_DIR / name
            dst = cache_dir / name
            if not dst.exists() or dst.read_text() != src.read_text():
                shutil.copy2(src, dst)

        self._write_ninja(cache_dir, variant_name, inst_cu, run_cu)
        self._run_ninja(cache_dir)

    def _write_ninja(self, cache_dir, variant_name, inst_cu, run_cu):
        hgcc = _get_compiler()
        link_libs = _get_link_libs()

        hgcc_flags = _get_hgcc_flags(cache_dir)

        so_path = cache_dir / f"{variant_name}.so"
        inst_obj = cache_dir / f"inst_{variant_name}.o"
        run_obj = cache_dir / f"run_{variant_name}.o"

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

build {inst_obj}: hgcc_compile {inst_cu}
build {run_obj}: hgcc_compile {run_cu}
build {so_path}: hgcc_link {inst_obj} {run_obj}
"""
        (cache_dir / "build.ninja").write_text(ninja_content)

    def _run_ninja(self, cache_dir):
        result = subprocess.run(
            ["ninja", "-j1"],
            cwd=str(cache_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"FMHA variant compilation failed:\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )


_variant_manager = FMHAVariantManager()


def get_fmha_variant(dtype_code, qo_tile_size, kv_tile_size,
                     single_wg, sparse_mode, page_size, split_kv, pack_factor):
    """Get a compiled FMHA variant module. Thread-safe, lazy compilation."""
    return _variant_manager.get_variant(
        dtype_code, qo_tile_size, kv_tile_size,
        single_wg, sparse_mode, page_size, split_kv, pack_factor)


# ============================================================================
# Plan kernel JIT (also needs sm_100a, can't be statically compiled)
# ============================================================================

_plan_module = None
_plan_lock = threading.Lock()


def _do_compile_plan():
    """Generate and compile plan kernel. No TVM loading."""
    cache_dir = CACHE_BASE / "plan"
    so_path = cache_dir / "fmha_sm80_plan.so"

    if so_path.exists():
        return

    logger.info("JIT compiling FMHA plan module")
    cache_dir.mkdir(parents=True, exist_ok=True)

    plan_cu = _FMHA_VARLEN_DIR / "fmha_sm80_plan.cu"
    shutil.copy2(plan_cu, cache_dir / "fmha_sm80_plan.cu")
    src = _FMHA_VARLEN_DIR / "msa_profiling_interface.hpp"
    dst = cache_dir / "msa_profiling_interface.hpp"
    if src.exists() and (not dst.exists() or dst.read_text() != src.read_text()):
        shutil.copy2(src, dst)

    hgcc = _get_compiler()
    link_libs = _get_link_libs()

    obj = cache_dir / "fmha_sm80_plan.o"

    hgcc_flags = _get_hgcc_flags(cache_dir, False)

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

build {obj}: hgcc_compile {cache_dir / "fmha_sm80_plan.cu"}
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
            f"FMHA plan compilation failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


def _compile_plan_only():
    """Compile plan kernel without loading. Safe for subprocesses."""
    _do_compile_plan()


def _compile_plan_module():
    """Compile and load plan kernel."""
    _do_compile_plan()
    import tvm_ffi
    so_path = CACHE_BASE / "plan" / "fmha_sm80_plan.so"
    return tvm_ffi.load_module(str(so_path))


def get_plan_fn():
    """Get the plan function. JIT compiles on first call."""
    global _plan_module
    if _plan_module is not None:
        return _plan_module

    with _plan_lock:
        if _plan_module is not None:
            return _plan_module
        _plan_module = _compile_plan_module()
        return _plan_module


# ============================================================================
# Prepare Plan Metadata kernel JIT
# ============================================================================

_prepare_metadata_module = None
_prepare_metadata_lock = threading.Lock()


def _do_compile_prepare_metadata():
    """Generate and compile prepare_plan_metadata kernel."""
    cache_dir = CACHE_BASE / "prepare_metadata"
    so_path = cache_dir / "prepare_plan_metadata.so"

    if so_path.exists():
        return

    logger.info("JIT compiling prepare_plan_metadata module")
    cache_dir.mkdir(parents=True, exist_ok=True)

    src_cu = _FMHA_VARLEN_DIR / "prepare_plan_metadata.cu"
    shutil.copy2(src_cu, cache_dir / "prepare_plan_metadata.cu")
    src = _FMHA_VARLEN_DIR / "msa_profiling_interface.hpp"
    dst = cache_dir / "msa_profiling_interface.hpp"
    if src.exists() and (not dst.exists() or dst.read_text() != src.read_text()):
        shutil.copy2(src, dst)

    # Copy required headers
    include_dir = _FMHA_VARLEN_DIR / "include"
    for name in ["prepare_plan_metadata.cuh", "utils.cuh"]:
        src = include_dir / name
        dst = cache_dir / name
        if src.exists() and (not dst.exists() or dst.read_text() != src.read_text()):
            shutil.copy2(src, dst)

    hgcc = _get_compiler()
    link_libs = _get_link_libs()

    obj = cache_dir / "prepare_plan_metadata.o"

    hgcc_flags = _get_hgcc_flags(cache_dir, False)

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

build {obj}: hgcc_compile {cache_dir / "prepare_plan_metadata.cu"}
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
            f"prepare_plan_metadata compilation failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


def _compile_prepare_metadata_module():
    """Compile and load prepare_plan_metadata kernel."""
    _do_compile_prepare_metadata()
    import tvm_ffi
    so_path = CACHE_BASE / "prepare_metadata" / "prepare_plan_metadata.so"
    return tvm_ffi.load_module(str(so_path))


def get_prepare_metadata_fn():
    """Get the prepare_plan_metadata function. JIT compiles on first call."""
    global _prepare_metadata_module
    if _prepare_metadata_module is not None:
        return _prepare_metadata_module

    with _prepare_metadata_lock:
        if _prepare_metadata_module is not None:
            return _prepare_metadata_module
        _prepare_metadata_module = _compile_prepare_metadata_module()
        return _prepare_metadata_module


# ============================================================================
# Sparse TopK Select kernel JIT
# ============================================================================

_sparse_topk_module = None
_sparse_topk_lock = threading.Lock()


def _do_compile_sparse_topk():
    cache_dir = CACHE_BASE / "sparse_topk"
    so_path = cache_dir / "sparse_topk_select.so"

    if so_path.exists():
        return

    logger.info("JIT compiling sparse_topk_select module")
    cache_dir.mkdir(parents=True, exist_ok=True)

    src_cu = _FMHA_VARLEN_DIR / "sparse_topk_select.cu"
    shutil.copy2(src_cu, cache_dir / "sparse_topk_select.cu")

    for name in ["tvm_ffi_utils.h", "msa_profiling_interface.hpp"]:
        src = _FMHA_VARLEN_DIR / name
        dst = cache_dir / name
        if not dst.exists() or dst.read_text() != src.read_text():
            shutil.copy2(src, dst)

    hgcc = _get_compiler()
    link_libs = _get_link_libs()

    obj = cache_dir / "sparse_topk_select.o"

    hgcc_flags = _get_hgcc_flags(cache_dir, False)
    # sparse_topk_select is arch-independent — the single PPU arch from
    # _gencode_flag(), which _get_hgcc_flags already includes, covers it.

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

build {obj}: hgcc_compile {cache_dir / "sparse_topk_select.cu"}
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
            f"sparse_topk_select compilation failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


def get_sparse_topk_module():
    """Get the sparse_topk_select module. JIT compiles on first call."""
    global _sparse_topk_module
    if _sparse_topk_module is not None:
        return _sparse_topk_module

    with _sparse_topk_lock:
        if _sparse_topk_module is not None:
            return _sparse_topk_module
        _do_compile_sparse_topk()
        import tvm_ffi
        so_path = CACHE_BASE / "sparse_topk" / "sparse_topk_select.so"
        _sparse_topk_module = tvm_ffi.load_module(str(so_path))
        return _sparse_topk_module


# ============================================================================
# Split-KV reduction kernel JIT
# ============================================================================

_reduction_module = None
_reduction_lock = threading.Lock()


def _do_compile_reduction():
    cache_dir = CACHE_BASE / "reduction"
    so_path = cache_dir / "fmha_sm80_reduction.so"

    if so_path.exists():
        return

    logger.info("JIT compiling fmha_sm80_reduction module")
    cache_dir.mkdir(parents=True, exist_ok=True)

    src_cu = _FMHA_VARLEN_DIR / "fmha_sm80_reduction.cu"
    shutil.copy2(src_cu, cache_dir / "fmha_sm80_reduction.cu")
    for name in ["gmem_bounds_check.h", "msa_profiling_interface.hpp"]:
        src = _FMHA_VARLEN_DIR / name
        dst = cache_dir / name
        if src.exists() and (not dst.exists() or dst.read_text() != src.read_text()):
            shutil.copy2(src, dst)

    hgcc = _get_compiler()
    link_libs = _get_link_libs()

    obj = cache_dir / "fmha_sm80_reduction.o"

    hgcc_flags = _get_hgcc_flags(cache_dir, False)

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

build {obj}: hgcc_compile {cache_dir / "fmha_sm80_reduction.cu"}
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
            f"fmha_sm80_reduction compilation failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


def get_reduction_module():
    """Get the split-KV reduction module. JIT compiles on first call."""
    global _reduction_module
    if _reduction_module is not None:
        return _reduction_module

    with _reduction_lock:
        if _reduction_module is not None:
            return _reduction_module
        _do_compile_reduction()
        import tvm_ffi
        so_path = CACHE_BASE / "reduction" / "fmha_sm80_reduction.so"
        _reduction_module = tvm_ffi.load_module(str(so_path))
        return _reduction_module
