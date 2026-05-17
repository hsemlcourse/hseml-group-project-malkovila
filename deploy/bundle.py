"""Pack minimal artifacts needed for HF Spaces / Render deployment."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "deploy" / "dist"

ARTIFACTS = [
    ROOT / "models" / "final_lgbm_cp2.joblib",
    ROOT / "models" / "text_tfidf_svd_artifacts.joblib",
    ROOT / "models" / "text_svd_meta.json",
    ROOT / "data" / "processed" / "split_meta.json",
    ROOT / "report" / "tables" / "permutation_importance.csv",
]


def bundle() -> None:
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)

    for src in ARTIFACTS:
        if not src.exists():
            print(f"[SKIP] {src.relative_to(ROOT)}")
            continue
        dst = DIST / src.name
        shutil.copy2(src, dst)
        print(f"[COPY] {src.relative_to(ROOT)} -> deploy/dist/{dst.name}")

    print(f"\nBundle ready: {DIST}  ({sum(f.stat().st_size for f in DIST.iterdir()) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    bundle()
