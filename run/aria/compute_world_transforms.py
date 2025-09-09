from pathlib import Path
from aria_alignment import compute_transforms_from_ref

def run_three_recordings():
    base = Path("../../../data")

    # Example recording folder names (replace with your real names)
    recs = {
        "rec1": "20250206_Testing",
        "rec2": "20250519_Testing",
        "rec3": "20250519_Testing",
    }

    names = {
        "20250206_Testing": "20250206_Testing",
        "20250519_Testing": 'b0bb3408-2f28-46d0-bf48-ad2892708a1c',
        "20250519_Testing": '929c8945-8451-4134-9de7-e483f2aa8f54'
    }
    # SINGLE-RUN outputs (what you already have today)
    singles = {
        k: base / v / "Aria" / f"mps_{names[v]}_vrs"  # <-- adjust if your layout differs
        for k, v in recs.items()
    }

    # MULTI-SLAM outputs (same three recordings, produced by one job)
    # Adjust these paths to wherever your Multi-SLAM job wrote them.
    multis = {
        k: base / v / "Aria" / "MultiSLAM" / f"mps_{names[v]}_vrs"
        for k, v in recs.items()
    }

    out_json = base / "world_frame_transforms.json"
    payload = compute_transforms_from_ref(
        singles=singles,
        multis=multis,
        ref_key="rec1",
        out_json=out_json,
    )
    print(f"Wrote transforms to: {out_json}")
    for k, info in payload["diagnostics"].items():
        print(f"[{k}] pairs_used={info['num_pairs_used']}, rmse={info['rmse_m']:.003f} m")

if __name__ == "__main__":
    run_three_recordings()
