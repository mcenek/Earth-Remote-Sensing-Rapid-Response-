"""Build the complete V11 inventory from its already published native counts.

No inference, mask changes or new dataset access occurs here.
"""
import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode

SITE = Path(__file__).resolve().parents[1] / "dist"
RUN_ID = "GTM_Model7_hard_v11_20261025_candidate"
GROUPS = [
    ("strong", "Strong match", "Labeled plume; native IoU at least 50%."),
    ("partial", "Partial match", "Labeled plume; native IoU from 10% to below 50%."),
    ("weak", "Weak trace", "Some correct plume pixels, but native IoU below 10%."),
    ("miss", "Missed plume", "No correct plume pixels. Marks elsewhere can still be false alarms."),
    ("false", "False alarm", "Reviewed no-plume scene with at least one marked native pixel."),
    ("clear", "Correct clear", "Reviewed no-plume scene with no marked native pixels."),
]


def category(tp, fp, fn):
    if tp + fn == 0:
        return "false" if fp else "clear"
    if tp == 0:
        return "miss"
    iou = tp / (tp + fp + fn)
    return "strong" if iou >= .5 else "partial" if iou >= .1 else "weak"


def main():
    catalog_file = SITE / "data/research-catalog.json"
    catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
    entry = next(e for e in catalog["experiments"] if e["id"] == RUN_ID)
    names = {key: label for key, label, _ in GROUPS}
    rows, totals, hashes = [], {k: 0 for k in ("tp", "fp", "fn", "tn")}, set()
    for scene in entry["scenes"]:
        m = scene["nativeMetrics"]
        tp, fp, fn, tn = [m[k] for k in totals]
        assert all(isinstance(n, int) and n >= 0 for n in (tp, fp, fn, tn))
        assert sum((tp, fp, fn, tn)) == m["evaluated_pixels"] > 0
        assert scene["predictionAvailable"] and scene["decisionThreshold"] == .65
        key = category(tp, fp, fn)
        assert (tp + fn > 0) == (scene["label"] == "PLUME")
        for asset in ("rgb", "predictionImage", "truthImage", "disagreementImage"):
            assert (SITE / scene[asset]).is_file(), scene[asset]
        hashes.add(scene["checkpointHash"])
        for k in totals:
            totals[k] += m[k]
        parts = [p.strip() for p in scene["name"].split("|")]
        iou = tp / (tp + fp + fn) if tp + fp + fn else None
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        for key_metric, value in (("iou", iou), ("precision", precision), ("recall", recall)):
            if value is not None:
                assert abs(value - m[key_metric]) < 1e-12
        rows.append({
            "id": scene["id"], "shortId": scene["id"][5:13], "name": scene["name"],
            "country": parts[1], "date": scene["date"], "label": scene["label"],
            "category": key, "categoryLabel": names[key], "cutoff": .65,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "iou": iou,
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "validPixels": m["evaluated_pixels"], "rgb": scene["rgb"],
            "predictionImage": scene["predictionImage"], "truthImage": scene["truthImage"],
            "disagreementImage": scene["disagreementImage"],
            "viewerUrl": "index.html?" + urlencode({"experiment": RUN_ID, "scene": scene["id"], "layout": "panels"}) + "#compare",
        })
    assert len(rows) == len({s["id"] for s in rows}) == 198
    assert len(hashes) == 1
    assert sum(s["label"] == "PLUME" for s in rows) == 46
    for key, value in totals.items():
        assert value == entry["nativeAggregate"][key]
    groups = [{"id": key, "label": label, "rule": rule,
               "count": sum(s["category"] == key for s in rows)} for key, label, rule in GROUPS]
    rank = {g[0]: i for i, g in enumerate(GROUPS)}
    rows.sort(key=lambda r: (rank[r["category"]], -(r["iou"] or 0), -r["fp"], r["country"], r["date"], r["id"]))
    payload = {"runId": RUN_ID, "total": len(rows), "plumes": 46, "noPlumes": 152,
               "cutoff": .65, "checkpointSha256": next(iter(hashes)),
               "catalogSha256": hashlib.sha256(catalog_file.read_bytes()).hexdigest(),
               "scope": "Exposed development observations used for model selection; not independent validation.",
               "groups": groups, "nativeAggregate": entry["nativeAggregate"], "predictions": rows}
    (SITE / "data/v11-predictions.json").write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    with (SITE / "data/v11-predictions.csv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"total": len(rows), "groups": {g["id"]: g["count"] for g in groups}, "nativeCounts": totals}))


if __name__ == "__main__":
    main()
