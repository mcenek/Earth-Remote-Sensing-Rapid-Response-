#!/usr/bin/env python3
"""Package saved Model7 scene exports for the static research site.

The source manifests and image bytes are not modified. Every scene is kept;
interactive numeric grids are included for six explicitly identified examples.
Usage: python tools/export_recent_results.py PATH_TO_RESEARCH_WORKTREE
"""
import copy
import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image
SITE = Path(__file__).resolve().parents[1] / "dist"
RUNS = [
    ("GTM_Model7_hard_v11_20261025_candidate", "Our U-Net · v11", "20261025"),
    ("GTM_Model7_hard_v11_20261025_baseline", "Earlier U-Net · v8", "20261025"),
    ("GTM_Model7_released_v15_20261026_candidate", "Released MARS-S2L · reference", "20261026"),
]
INTERACTIVE_IDS = {"53ce2910", "9123854a", "cde9d349", "df4817f5", "40c50f2a", "eaf034e7"}
SCENE_ASSETS = ("rgb", "reference90Rgb", "truthImage", "predictionImage", "probabilityImage", "disagreementImage")
DEFAULT_SCENE = "MARS_cde9d349-a49f-4c55-ae06-33acc04acf64"


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    worktree = Path(sys.argv[1]).resolve()
    source_root = worktree / "outputs" / "GTM_viewer_bundles"
    target = SITE / "assets" / "research"
    target.mkdir(parents=True, exist_ok=True)
    entries, samples, copied, outlines = [], {}, {}, {}

    def materialize_bytes(data, suffix):
        digest = hashlib.sha256(data).hexdigest()
        rel = Path("assets/research") / (digest + suffix.lower())
        destination = SITE / rel
        if not destination.exists():
            destination.write_bytes(data)
        elif hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Existing asset hash mismatch: {destination}")
        copied[digest] = len(data)
        return rel.as_posix()

    def materialize(source):
        return materialize_bytes(source.read_bytes(), source.suffix)

    def outline_for(truth_path):
        # Preserve the original binary mask, while rendering its boundary on top
        # of predictions so the annotation does not hide a correctly marked plume.
        if truth_path in outlines:
            return outlines[truth_path]
        with Image.open(SITE / truth_path) as source:
            image = source.convert("RGBA")
            width, height = image.size
            mask = image.getchannel("A").tobytes()
        alpha = bytearray(width * height)
        for index, present in enumerate(mask):
            if not present:
                continue
            x, y = index % width, index // width
            edge = x == 0 or y == 0 or x == width - 1 or y == height - 1
            if not edge:
                edge = any(mask[near] == 0 for near in (index - 1, index + 1, index - width, index + width))
            alpha[index] = 240 if edge else 30
        outline = Image.new("RGBA", (width, height), (0, 225, 230, 0))
        outline.putalpha(Image.frombytes("L", (width, height), bytes(alpha)))
        buffer = io.BytesIO()
        outline.save(buffer, format="PNG")
        outlines[truth_path] = materialize_bytes(buffer.getvalue(), ".png")
        return outlines[truth_path]

    for run_id, label, dated in RUNS:
        root = (source_root / run_id).resolve()
        manifest = root / "experiment.json"
        entry = json.loads(manifest.read_text(encoding="utf-8"))
        entry = copy.deepcopy(entry)
        entry.pop("reportUrl", None)
        entry["name"] = label
        entry["status"] = "EXPOSED DEVELOPMENT SET · saved predictions"
        entry["description"] = (
            "198 previously reviewed development observations: 46 plume and 152 no-plume. "
            "These scenes were used for model selection; this is not independent accuracy. "
            "The released model has training-location overlap, so the comparison is not controlled."
        )
        entry["notes"] = "MARS-S2L binary plume annotations. Results do not measure emission rate or establish US-wide performance."
        for scene in entry["scenes"]:
            assert scene["id"].startswith("MARS_")
            short = scene["id"][5:13]
            for key in SCENE_ASSETS:
                if scene.get(key):
                    source = (root / scene[key]).resolve()
                    if not source.is_relative_to(root) or not source.is_file():
                        raise ValueError(f"Missing or unsafe {key}: {source}")
                    scene[key] = materialize(source)
            if scene.get("truthImage"):
                scene["truthOutlineImage"] = outline_for(scene["truthImage"])
            if short in INTERACTIVE_IDS and scene.get("grid"):
                source = (root / scene["grid"]).resolve()
                if not source.is_relative_to(root) or not source.is_file():
                    raise ValueError(f"Missing or unsafe grid: {source}")
                scene["grid"] = materialize(source)
            else:
                scene.pop("grid", None)
            if run_id == RUNS[0][0] and short in INTERACTIVE_IDS:
                samples[short] = {k: scene.get(k) for k in ("id", "name", "rgb", "truthImage", "truthOutlineImage", "predictionImage", "disagreementImage", "nativeMetrics")}
        if len(entry["scenes"]) != 198:
            raise ValueError(f"Unexpected scene count for {run_id}")
        entries.append(entry)

    catalog = {
        "schemaVersion": 1,
        "activeExperimentId": RUNS[0][0],
        "presentation": {"defaultView": "compare", "experiments": [
            {"id": run_id, "label": label, "sceneId": DEFAULT_SCENE} for run_id, label, _ in RUNS
        ]},
        "local": {"bundleDirectory": "Published saved snapshot", "warnings": []},
        "live": {"serverName": "Published research snapshot"},
        "experiments": entries,
    }
    data_dir = SITE / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "research-catalog.json").write_text(json.dumps(catalog, separators=(",", ":")), encoding="utf-8", newline="\n")
    (data_dir / "progress-cases.json").write_text(json.dumps(samples, separators=(",", ":")), encoding="utf-8", newline="\n")
    receipt = {"runs": [r[0] for r in RUNS], "scenes_per_run": 198, "interactive_grids_per_run": len(INTERACTIVE_IDS),
               "unique_assets": len(copied), "total_unique_bytes": sum(copied.values()),
               "source_manifest_sha256": {run_id: hashlib.sha256((source_root / run_id / "experiment.json").read_bytes()).hexdigest() for run_id, _, _ in RUNS}}
    (data_dir / "export-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
