#!/usr/bin/env python3
"""Local metadata-only original-scale feasibility audit; no imagery or network."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH_COLUMNS = ("S2_t0_path", "S2_pre_path", "S2_pre_pre_path", "S2_plume_label_path")


def coordinate_label(row):
    lat, lon, label = float(row["latitude"]), float(row["longitude"]), int(row["label"])
    if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("Invalid coordinate")
    if label not in (0, 1):
        raise ValueError("Nonbinary label")
    return lat, lon, label


def proposed_member_path(path):
    """Explicit alias hypothesis, never an implicit replacement in acquisition."""
    prefix = "data/original_scale/train/s2/"
    if not path.startswith(prefix) or any(x in path.split("/") for x in ("..", ".", "")):
        raise ValueError(f"Not an original-scale training S2 path: {path}")
    return "data/original/geo/train/s2/" + path[len(prefix):]


def audit(original, frozen_manifest, candidates, survey):
    original_ids = [str(r["id"]) for r in original]
    frozen_ids = {str(i) for i in candidates["candidate_ids"]}
    frozen = [r for r in frozen_manifest if str(r["id"]) in frozen_ids]
    if len(frozen) != len(frozen_ids):
        raise ValueError("Frozen candidate IDs do not uniquely resolve")
    groups = {coordinate_label(r): len(r["row_ids"]) for r in candidates["coordinate_groups"]}
    if dict(Counter(coordinate_label(r) for r in frozen)) != groups:
        raise ValueError("Frozen manifest and candidate coordinate groups disagree")
    s2 = [r for r in original if "S2" in r["available_sensor"].split(",") and r.get("S2_t0_path")]
    if len(s2) != len({str(r["id"]) for r in s2}):
        raise ValueError("Duplicate S2 original manifest IDs")
    mapped = [r for r in s2 if coordinate_label(r) in groups]
    counts = Counter(coordinate_label(r) for r in mapped)
    frozen_index = {str(r["id"]): r for r in frozen}
    collisions = [r for r in original if str(r["id"]) in frozen_index]
    disagreements = sum(coordinate_label(r) != coordinate_label(frozen_index[str(r["id"])]) for r in collisions)
    path_to_row = {}
    for row in mapped:
        for column in PATH_COLUMNS:
            if row.get(column):
                path_to_row[proposed_member_path(row[column])] = row
    hits = []
    for entry in survey["results"]:
        member = entry.get("first_member")
        if member in path_to_row:
            row = path_to_row[member]
            hits.append({"archive": entry["archive"], "archive_bytes": entry["archive_bytes"],
                         "first_member": member, "original_id": int(row["id"]),
                         "latitude": float(row["latitude"]), "longitude": float(row["longitude"]),
                         "label": int(row["label"]),
                         "expected_members_under_alias_hypothesis": {c: proposed_member_path(row[c]) for c in PATH_COLUMNS},
                         "proven_members": [member], "payload_or_quality_verified": False})
    hits.sort(key=lambda r: (r["archive"], r["original_id"]))
    return {
        "schema_version": "ersrr.methaneunion.original-mapping.v1",
        "status": "feasible_new_metadata_cohort_only_not_acquisition_authorization",
        "access": {"network": False, "raster_payload": False, "released_test_manifest": False, "model_outcomes": False},
        "mapping_rule": "Exact finite float64 latitude + longitude + binary label, within original_scale geo train S2; IDs are not cross-scale identifiers.",
        "frozen": {"rows": len(frozen), "coordinate_label_groups": len(groups),
                   "distinct_spatial_coordinates": len({k[:2] for k in groups}),
                   "labels": dict(Counter(str(r["label"]) for r in frozen))},
        "original": {"manifest_rows": len(original), "cross_sensor_excess_id_rows": len(original_ids) - len(set(original_ids)), "s2_rows": len(s2), "mapped_rows": len(mapped),
                     "mapped_coordinate_label_groups": len(counts), "labels": dict(Counter(str(r["label"]) for r in mapped)),
                     "rows_with_four_nonempty_paths": sum(all(r.get(c) for c in PATH_COLUMNS) for r in mapped)},
        "unsafe_id_join": {"overlapping_ids": len(collisions), "coordinate_or_label_disagreements": disagreements,
                           "matching_coordinate_and_label": len(collisions) - disagreements},
        "changed_group_counts": [{"latitude": k[0], "longitude": k[1], "label": k[2], "frozen_rows": groups[k],
                                  "original_rows": counts[k]} for k in sorted(groups) if counts[k] != groups[k]],
        "missing_groups": [list(k) for k in sorted(set(groups) - set(counts))],
        "alias_hypothesis": {"manifest": "data/original_scale/train/s2/", "archive": "data/original/geo/train/s2/",
                             "globally_verified": False},
        "first_member_matches": hits,
        "deterministic_pilot": hits[0] if hits else None,
        "pilot_bounds": {"first_member_compressed_size_known": False, "first_member_uncompressed_size_known": False,
                         "suggested_maximum_network_bytes": 16777216,
                         "suggested_maximum_decompressed_bytes": 67108864,
                         "rule": "One first-member header/payload pilot only; start at existing 256 KiB prefix size and expand under hard bounds. Stop if limits fail; never fall back to full archive. These caps are proposed, not size estimates or authorization."},
        "limitations": [
            "This defines a new original-scale cohort, not recovery of the frozen cross-scale sample identities.",
            "Coordinate and label do not establish acquisition date, plume event identity, crop offset, or one-to-one sample correspondence.",
            "A first TAR member confirms one path only. Other temporal members and masks may be anywhere or absent, including beyond the release tail.",
            "Manifests contain no shape, dtype, band order, radiometric units, geotransform, or payload size; those remain unverified.",
            "Original-scale name does not prove native pixel scale. Local legacy crop source uses S2 32 px crops resized to 224 and claims 10 m GSD, while raw acquisition uses 20 m products; neither proves these release TIFFs.",
            "Binary scene labels and plume paths are supplied; mask semantics, co-registration, temporal observability, and independent geographic grouping require separate adjudication.",
        ],
    }


def provenance(path):
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def markdown(report):
    f, o, j = report["frozen"], report["original"], report["unsafe_id_join"]
    lines = ["# MethaneUnion original-scale mapping audit", "", "Local metadata only; no downloads, imagery, test manifests, or model outcomes opened.", "",
             f"The frozen {f['rows']} rows represent {f['coordinate_label_groups']} coordinate-label groups but only {f['distinct_spatial_coordinates']} distinct coordinates.", "",
             f"Exact coordinate+label mapping gives **{o['mapped_rows']} original-scale S2 rows**, {o['labels'].get('1', 0)} positive and {o['labels'].get('0', 0)} negative, covering {o['mapped_coordinate_label_groups']} groups. All {o['rows_with_four_nonempty_paths']} have four nonempty manifest paths. This is a new cohort requiring a separate protocol.", "",
             f"Reusing IDs is unsafe: {j['overlapping_ids']} overlap, but {j['coordinate_or_label_disagreements']} disagree in coordinates or label. {len(report['changed_group_counts'])} coordinate-label groups change sample count.", ""]
    pilot = report["deterministic_pilot"]
    if pilot:
        lines += [f"The observed first member of **{pilot['archive']}** matches original training ID {pilot['original_id']} (label {pilot['label']}) after the explicit path alias hypothesis:", "", f"`{pilot['first_member']}`", "",
                  "This proves one path exists, not that a complete training example or usable mask exists. Proposed expected members:", ""]
        lines += [f"- `{p}`" for p in pilot["expected_members_under_alias_hypothesis"].values()]
    lines += ["", "A possible next step is one first-member pilot capped at 16 MiB network / 64 MiB decompressed, starting at 256 KiB. Member size is absent from the saved survey, so success within those caps is unknown. Do not download the full archive automatically.", "", "## Limits", ""]
    if "existing_probe_metadata" in report:
        probe = report["existing_probe_metadata"]
        lines[-3:-3] = ["", f"An independent existing probe already stopped at its {probe['network_cap']:,}-byte cap with status `{probe['status']}`. It did not obtain a complete member. The larger cap above is only a hypothetical planning bound, not an instruction to extend that stopped probe."]
    lines += [f"- {s}" for s in report["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stem", type=Path, default=ROOT / "reports/acquisition/methaneunion_original_mapping")
    args = parser.parse_args()
    original_path = ROOT / ".research/external/MethaneUnion-release/datasets/geo_split/original_scale/train.csv"
    frozen_path = ROOT / ".research/external/MethaneUnion-release/datasets/geo_split/480m_GSD/train.csv"
    candidate_path = ROOT / "reports/acquisition/methaneunion_novel_s2_candidates.json"
    survey_path = ROOT / "reports/acquisition/methaneunion_remaining_archive_prefix_survey.json"
    with original_path.open(newline="", encoding="utf-8-sig") as handle:
        original = list(csv.DictReader(handle))
    with frozen_path.open(newline="", encoding="utf-8-sig") as handle:
        frozen = list(csv.DictReader(handle))
    report = audit(original, frozen, json.loads(candidate_path.read_text()), json.loads(survey_path.read_text()))
    report["inputs"] = {name: provenance(path) for name, path in [("original_manifest", original_path), ("frozen_manifest", frozen_path), ("candidates", candidate_path), ("prefix_survey", survey_path)]}
    probe_path = ROOT / "reports/acquisition/methaneunion_one_member_probe.json"
    if probe_path.exists():
        report["existing_probe_metadata"] = json.loads(probe_path.read_text(encoding="utf-8"))
        report["inputs"]["existing_probe_metadata"] = provenance(probe_path)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    args.output_stem.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_stem.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"mapped_rows": report["original"]["mapped_rows"], "first_member_matches": len(report["first_member_matches"]), "output_stem": str(args.output_stem)}))


if __name__ == "__main__":
    main()
