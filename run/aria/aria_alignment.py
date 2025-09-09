# world_align.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from projectaria_tools.core import mps
from projectaria_tools.core.sensor_data import TimeQueryOptions

# ---------- Basics: load timestamps & providers ----------
def _load_tracking_times(mps_root: Path) -> np.ndarray:
    csv = mps_root / "slam" / "closed_loop_trajectory.csv"
    df = pd.read_csv(csv, usecols=["tracking_timestamp_us"])
    ts = df["tracking_timestamp_us"].to_numpy(np.int64)
    # Ensure strictly increasing & unique (some CSVs can have duplicates at the ends)
    return np.unique(ts)

def _provider(mps_root: Path) -> mps.MpsDataProvider:
    paths = mps.MpsDataPathsProvider(str(mps_root))
    return mps.MpsDataProvider(paths.get_data_paths())

def _device_positions(provider: mps.MpsDataProvider, times_us: np.ndarray) -> np.ndarray:
    pts = []
    for t in times_us:
        pose = provider.get_closed_loop_pose(int(t * 1000), TimeQueryOptions.CLOSEST)
        T = pose.transform_world_device
        p = np.asarray(T.translation(), dtype=np.float64).reshape(3)
        pts.append(p)
    return np.vstack(pts)

# ---------- Time matching with tolerance ----------
def _match_times(tA: np.ndarray, tU: np.ndarray, tol_us: int = 2000) -> np.ndarray:
    # Two-pointer linear match for nearly identical sequences (fast & robust)
    i = j = 0
    pairs = []
    while i < len(tA) and j < len(tU):
        dt = int(tA[i]) - int(tU[j])
        if abs(dt) <= tol_us:
            pairs.append((int(tA[i]), int(tU[j])))
            i += 1; j += 1
        elif dt < 0:
            i += 1
        else:
            j += 1
    return np.array(pairs, dtype=np.int64)

# ---------- Fixed-scale Umeyama (Procrustes) ----------
def _umeyama(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find R, t that best map A -> B (no scale):  R @ A_i + t ~ B_i
    """
    muA = A.mean(axis=0)
    muB = B.mean(axis=0)
    A0 = A - muA
    B0 = B - muB
    H = A0.T @ B0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = muB - R @ muA
    return R, t

def _to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = t
    return T

# ---------- Optional robust refinement (trim outliers once) ----------
def _refine_once(A: np.ndarray, B: np.ndarray, keep_ratio: float = 0.9):
    R, t = _umeyama(A, B)
    pred = (A @ R.T) + t
    err2 = np.sum((pred - B) ** 2, axis=1)
    k = max(10, int(len(A) * keep_ratio))
    idx = np.argsort(err2)[:k]
    R2, t2 = _umeyama(A[idx], B[idx])
    pred2 = (A @ R2.T) + t2
    rmse = float(np.sqrt(np.mean(np.sum((pred2 - B) ** 2, axis=1))))
    return R2, t2, rmse, k

# ---------- Public: estimate T_U<-A for one recording ----------
def estimate_T_U_from_A(
    single_root: Path,
    multi_root: Path,
    max_pairs: int = 2000,
    tol_us: int = 2000,
    robust_trim_ratio: float = 0.9,
):
    """
    single_root: MPS folder for the single-run output of THIS recording (A)
    multi_root:  MPS folder for the Multi-SLAM output of THIS recording (U)
    Returns: (T_U_from_A: 4x4, diag: dict)
    """
    tA = _load_tracking_times(single_root)
    tU = _load_tracking_times(multi_root)
    pairs = _match_times(tA, tU, tol_us=tol_us)
    if len(pairs) < 10:
        raise RuntimeError(f"Matched only {len(pairs)} timestamps (<10). Check inputs / tolerance.")

    # Subsample uniformly for speed if very long
    if len(pairs) > max_pairs:
        step = len(pairs) // max_pairs + 1
        pairs = pairs[::step]

    provA = _provider(single_root)
    provU = _provider(multi_root)
    tA_sel = pairs[:, 0]
    tU_sel = pairs[:, 1]
    PA = _device_positions(provA, tA_sel)  # world_A device origins
    PU = _device_positions(provU, tU_sel)  # world_U device origins

    # One-shot robust fit
    R, t, rmse_m, used = _refine_once(PA, PU, keep_ratio=robust_trim_ratio)
    T = _to_T(R, t)

    diag = {
        "num_pairs_total": int(len(pairs)),
        "num_pairs_used": int(used),
        "rmse_m": rmse_m,
    }
    return T, diag

# ---------- Multi-recording driver: produce all transforms from ref ----------
def compute_transforms_from_ref(
    singles: dict[str, Path],
    multis: dict[str, Path],
    ref_key: str,
    out_json: Path,
):
    """
    singles: {"rec1": Path(...single-run mps root...),
              "rec2": Path(...),
              "rec3": Path(...)}
    multis:  {"rec1": Path(...multi-SLAM mps root for the SAME recording...),
              "rec2": Path(...),
              "rec3": Path(...)}
    ref_key: which entry in the dict is your 'first' recording (e.g., "rec1")
    out_json: where to save results as JSON

    Returns: payload dict (and writes JSON)
    """
    T_U_from_A = {}
    diagnostics = {}

    # 1) per-recording: estimate T_U<-A
    for k in singles.keys():
        T, diag = estimate_T_U_from_A(singles[k], multis[k])
        T_U_from_A[k] = T.tolist()
        diagnostics[k] = diag

    # 2) pairwise from 'ref' recording: T_Ai<-Aref = (T_U<-Ai)^(-1) @ T_U<-Aref
    ref_T_U_from_A = np.array(T_U_from_A[ref_key], dtype=np.float64)
    pairwise_from_ref = {}
    for k in singles.keys():
        if k == ref_key:
            continue
        T_U_from_Ak = np.array(T_U_from_A[k], dtype=np.float64)
        T_Ak_from_U = np.linalg.inv(T_U_from_Ak)
        T_Ak_from_Aref = T_Ak_from_U @ ref_T_U_from_A
        pairwise_from_ref[f"T_{k}_from_{ref_key}"] = T_Ak_from_Aref.tolist()

    payload = {
        "T_U_from_A": T_U_from_A,
        "pairwise_from_ref": pairwise_from_ref,
        "diagnostics": diagnostics,
        "note": "All matrices are 4x4 row-major. Apply as X_dst = T_dst<-src @ X_src.",
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    return payload
