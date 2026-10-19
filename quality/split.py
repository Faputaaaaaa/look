"""Group-isolated train/val split plus hard-heldout SKU split."""
import random
import itertools
import math
from collections import defaultdict

from .manifest import validate_manifest


def build_group_split(manifest, heldout_skus=(), val_fraction=0.2, seed=42):
    validate_manifest(manifest)
    if not (0 < val_fraction < 1):
        raise ValueError("val_fraction must be strictly between zero and one")
    heldout = set(heldout_skus)
    known_skus = {r["sku_id"] for r in manifest["images"]}
    if heldout - known_skus:
        raise ValueError(f"unknown heldout SKUs: {sorted(heldout-known_skus)}")
    groups = defaultdict(list)
    for r in manifest["images"]:
        groups[r["capture_group"]].append(r)
    seen_groups, unseen_groups = [], []
    for group_id, records in sorted(groups.items()):
        is_unseen = [r["sku_id"] in heldout for r in records]
        if any(is_unseen) and not all(is_unseen):
            raise ValueError(f"capture_group {group_id!r} mixes seen and heldout SKU")
        (unseen_groups if all(is_unseen) else seen_groups).append(group_id)
    if len(seen_groups) < 2:
        raise ValueError("at least two non-heldout capture groups are required")
    # Best-effort group-level stratification: unlike a single random draw, avoid
    # a validation partition containing no defects whenever feasible. No images
    # or patches from a capture_group may cross train/val boundaries.
    n_val = min(len(seen_groups) - 1, max(1, round(len(seen_groups) * val_fraction)))
    per_group = {}
    for group_id in seen_groups:
        rows = groups[group_id]
        per_group[group_id] = {cls: sum(d["class_name"] == cls for r in rows for d in r["defects"])
                               for cls in manifest["classes"]}
    group_presence = {cls: sum(per_group[g][cls] > 0 for g in seen_groups) for cls in manifest["classes"]}
    totals = {cls: sum(per_group[g][cls] for g in seen_groups) for cls in manifest["classes"]}

    def penalty(val_group_ids):
        val = set(val_group_ids)
        score = 0.0
        for cls in manifest["classes"]:
            val_count = sum(per_group[g][cls] for g in val)
            total = totals[cls]
            if group_presence[cls] >= 2:
                if val_count == 0 or val_count == total:
                    score += 1000  # failure to represent a feasible class in both groups
                score += abs(val_count / total - val_fraction) * 10
            elif group_presence[cls] == 1 and val_count:
                score += 200  # keep sole examples in train, not val
        if sum(sum(per_group[g].values()) for g in val) == 0:
            score += 100  # do not validate on normal images alone when avoidable
        return score

    # Exhaustive choices for small datasets; bounded seeded search for large.
    n_combinations = math.comb(len(seen_groups), n_val)
    if n_combinations <= 4096:
        candidates = itertools.combinations(sorted(seen_groups), n_val)
    else:
        rng = random.Random(seed)
        candidates = (rng.sample(seen_groups, n_val) for _ in range(1024))
    val_groups = set(min(candidates, key=lambda ids: (penalty(ids), tuple(sorted(ids)))))
    out = {"seed": seed, "split_type": "group_isolated_with_unseen_sku_holdout", "train": [], "val": [], "test_unseen_sku": [],
           "heldout_skus": sorted(heldout), "group_field": "capture_group"}
    for r in manifest["images"]:
        part = "test_unseen_sku" if r["sku_id"] in heldout else "val" if r["capture_group"] in val_groups else "train"
        out[part].append(r["image_id"])
    for part in ("train", "val", "test_unseen_sku"):
        out[part].sort()
    by_id = {r["image_id"]: r for r in manifest["images"]}
    out["class_instance_counts"] = {part: {
        cls: sum(d["class_name"] == cls for id_ in out[part] for d in by_id[id_]["defects"])
        for cls in manifest["classes"]} for part in ("train", "val", "test_unseen_sku")}
    out["coverage_warnings"] = [f"{cls}: train/val group coverage insufficient; add more independently captured examples"
        for cls in manifest["classes"]
        if not out["class_instance_counts"]["train"][cls] or not out["class_instance_counts"]["val"][cls]]
    assert not (set(out["train"]) & set(out["val"]))
    return out


def check_group_isolation(manifest, split):
    validate_manifest(manifest)
    by_id = {r["image_id"]: r for r in manifest["images"]}
    partition_of = {}
    for partition in ("train", "val", "test_unseen_sku"):
        for image_id in split.get(partition, []):
            if image_id not in by_id or image_id in partition_of:
                raise ValueError(f"unknown/duplicated image in split: {image_id}")
            partition_of[image_id] = partition
    if set(partition_of) != set(by_id):
        raise ValueError("split does not cover all manifest images")
    group_partitions = defaultdict(set)
    sku_partitions = defaultdict(set)
    for image_id, partition in partition_of.items():
        r = by_id[image_id]
        group_partitions[r["capture_group"]].add(partition)
        sku_partitions[r["sku_id"]].add(partition)
    if any(len(x) != 1 for x in group_partitions.values()):
        raise ValueError("capture_group leaks across splits")
    heldout_skus = set(split.get("heldout_skus", []))
    if heldout_skus - set(sku_partitions):
        raise ValueError("heldout_skus lists an absent SKU")
    for sku, parts in sku_partitions.items():
        if sku in heldout_skus and parts != {"test_unseen_sku"}:
            raise ValueError(f"heldout SKU {sku!r} leaks into train/val")
        if sku not in heldout_skus and "test_unseen_sku" in parts:
            raise ValueError(f"unlisted SKU {sku!r} is incorrectly marked as unseen")
    return True
