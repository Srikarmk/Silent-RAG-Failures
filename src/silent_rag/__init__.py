"""Silent RAG failures: a benchmark for label-free detection of retrieval degradation."""

import os
import sys

# On macOS, faiss and torch each ship their own OpenMP runtime, and running both in one
# process segfaults (exit 139, no error message) once they compute in parallel. One
# OpenMP thread avoids it. Set here, on import, before either library loads; setdefault,
# so an explicit setting still wins. No effect on Linux or Windows.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
