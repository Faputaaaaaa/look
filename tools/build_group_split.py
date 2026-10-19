"""Generate an opt-in grouped split; does not modify legacy split.json."""
import argparse
import json
from pathlib import Path
from quality.manifest import load_manifest
from quality.split import build_group_split, check_group_isolation


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--heldout-sku", action="append", default=[], help="repeat for multiple completely unseen SKUs")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    manifest = load_manifest(a.manifest)
    split = build_group_split(manifest, a.heldout_sku, a.val_fraction, a.seed)
    check_group_isolation(manifest, split)
    if a.out.resolve() == (Path(__file__).resolve().parent.parent / "split.json").resolve():
        ap.error("refusing to overwrite existing legacy split.json")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print({p: len(split[p]) for p in ("train", "val", "test_unseen_sku")})
    print("class instance counts:", split["class_instance_counts"])
    for warning in split["coverage_warnings"]:
        print("WARNING:", warning)


if __name__ == "__main__":
    main()
