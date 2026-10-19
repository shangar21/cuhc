"""Loads libcuhc.so (built by CMake into this directory) and declares its C API."""

import ctypes
import pathlib

_LIB_PATH = pathlib.Path(__file__).resolve().parent / "libcuhc.so"

if not _LIB_PATH.exists():
    raise ImportError(
        f"{_LIB_PATH} not found. Install the package with `pip install .`, or build the "
        "library in the source tree with:\n"
        "  cmake -S . -B build && cmake --build build -j"
    )

lib = ctypes.CDLL(str(_LIB_PATH))

_float_ptr = ctypes.POINTER(ctypes.c_float)

lib.cuhc_pairwise_distances.restype = ctypes.c_int
lib.cuhc_pairwise_distances.argtypes = [
    _float_ptr, ctypes.c_int,  # pointsA, numPointsA
    _float_ptr, ctypes.c_int,  # pointsB (NULL = same as A), numPointsB
    ctypes.c_int,              # numFeatures
    ctypes.c_int,              # metric id
    ctypes.c_float,            # metric parameter (Minkowski p)
    _float_ptr,                # output distances
]

lib.cuhc_linkage.restype = ctypes.c_int
lib.cuhc_linkage.argtypes = [
    _float_ptr, ctypes.c_int, ctypes.c_int,  # points, numPoints, numFeatures
    ctypes.c_int, ctypes.c_float,            # metric id, metric parameter
    ctypes.c_int,                            # linkage id
    ctypes.c_int,                            # algorithm id
    ctypes.POINTER(ctypes.c_double),         # output tree, (numPoints - 1) x 4
]

# Spatial pruning switch and statistics (tests and benchmarks).
lib.cuhc_set_pruning.restype = ctypes.c_int
lib.cuhc_set_pruning.argtypes = [ctypes.c_int]
lib.cuhc_prune_stats.restype = None
lib.cuhc_prune_stats.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)] * 2

PRUNE_MODES = {"auto": 0, "always": 1, "never": 2}


def set_pruning(mode):
    """Spatial pruning for mst, rnn and HDBSCAN: "auto", "always" or "never".

    The results are the same in all modes; only the speed changes.
    """
    check(lib.cuhc_set_pruning(PRUNE_MODES[mode]))


def prune_stats():
    """(tiles computed, tiles in total) by pruned passes since the last call."""
    computed, total = ctypes.c_ulonglong(), ctypes.c_ulonglong()
    lib.cuhc_prune_stats(ctypes.byref(computed), ctypes.byref(total))
    return computed.value, total.value

# Test hook: Ward through rnn rounds with explicit hand-over settings.
lib.cuhc_ward_rnn_with_options.restype = ctypes.c_int
lib.cuhc_ward_rnn_with_options.argtypes = [
    _float_ptr, ctypes.c_int, ctypes.c_int,  # points, numPoints, numFeatures
    ctypes.c_float, ctypes.c_int, ctypes.c_int,  # min progress, dense below, allow dense
    ctypes.POINTER(ctypes.c_double),         # output tree, (numPoints - 1) x 4
]

lib.cuhc_hdbscan_tree.restype = ctypes.c_int
lib.cuhc_hdbscan_tree.argtypes = [
    _float_ptr, ctypes.c_int, ctypes.c_int,  # points, numPoints, numFeatures
    ctypes.c_int, ctypes.c_float,            # metric id, metric parameter
    ctypes.c_int, ctypes.c_float,            # min_samples, alpha
    ctypes.c_int,                            # core distance method (0 auto, 1 fused, 2 radix)
    ctypes.POINTER(ctypes.c_double),         # output tree, (numPoints - 1) x 4
    _float_ptr,                              # output core distances (may be NULL)
]

_int64_ptr = ctypes.POINTER(ctypes.c_int64)
lib.cuhc_condense_tree.restype = ctypes.c_int64
lib.cuhc_condense_tree.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int64, ctypes.c_int64,  # tree, n, min_cluster_size
    ctypes.c_int64,                                                   # output capacity
    _int64_ptr, _int64_ptr, ctypes.POINTER(ctypes.c_double), _int64_ptr,
]

# cudaErrorMemoryAllocation
CUDA_ERROR_OUT_OF_MEMORY = 2

lib.cuhc_error_string.restype = ctypes.c_char_p
lib.cuhc_error_string.argtypes = [ctypes.c_int]


def check(status):
    """Raises RuntimeError if a C API call returned a non-zero status."""
    if status != 0:
        raise RuntimeError(f"cuhc: {lib.cuhc_error_string(status).decode()} (code {status})")


def float_ptr(array):
    return array.ctypes.data_as(_float_ptr)


def double_ptr(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def int64_ptr(array):
    return array.ctypes.data_as(_int64_ptr)
