import pathlib
import sys

# Use the package straight from the source tree (python/cuhc, with libcuhc.so
# built into it by CMake).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

# CUHC_TEST_PRUNING=always (or never) runs the whole suite with spatial
# pruning forced on (or off). Results must not change.
import os  # noqa: E402

if os.environ.get("CUHC_TEST_PRUNING"):
    from cuhc import _lib  # noqa: E402

    _lib.set_pruning(os.environ["CUHC_TEST_PRUNING"])
