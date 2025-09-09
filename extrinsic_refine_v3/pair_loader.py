
from io_utils import load_clouds_from_dir
def load_paired_dirs(orbbec_dir, aria_dir, orbbec_glob="*.ply", aria_glob="*.ply", limit=None):
    O, Ofiles = load_clouds_from_dir(orbbec_dir, orbbec_glob, limit)
    A, Afiles = load_clouds_from_dir(aria_dir, aria_glob, limit)
    if len(O) != len(A):
        raise ValueError(f"Pair count mismatch: {len(O)} Orbbec vs {len(A)} Aria")
    return O, A, Ofiles, Afiles
