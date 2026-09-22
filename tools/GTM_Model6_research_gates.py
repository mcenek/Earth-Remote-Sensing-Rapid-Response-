"""Run a bounded, read-only readiness gate for the active GTM_Model6 study.

This command deliberately does not train, download imagery, open held-out labels,
or overwrite an experiment.  It records what the current local cohort and frozen
pilot artifacts can support before another run is authorized.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL6 = ROOT / "research" / "GTM_Model6"
REPORTS = ROOT / "reports" / "research"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def available_ram_gib() -> float | None:
    """Return Windows available physical memory without importing ML runtimes."""

    if sys.platform.startswith("win"):
        class Memory(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("load", ctypes.c_ulong),
                ("total", ctypes.c_ulonglong),
                ("available", ctypes.c_ulonglong),
                ("page_total", ctypes.c_ulonglong),
                ("page_available", ctypes.c_ulonglong),
                ("virtual_total", ctypes.c_ulonglong),
                ("virtual_available", ctypes.c_ulonglong),
                ("extended", ctypes.c_ulonglong),
            ]

        memory = Memory()
        memory.length = ctypes.sizeof(memory)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            return float(memory.available / 2**30)
    return None


def _legacy_check() -> dict[str, Any]:
    path = ROOT / "tools" / "GTM_Model0_legacy_baseline.py"
    spec = importlib.util.spec_from_file_location("gtm_model0_legacy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load legacy checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check()


def _inventory() -> dict[str, Any]:
    """Read the existing bounded Model6 cohort; no network or new files."""

    sys.path.insert(0, str(MODEL6))
    from GTM_Model6_plumes import inventory  # type: ignore[import-not-found]

    items, excluded = inventory(32)
    datasets: dict[str, int] = {}
    positive_groups: set[str] = set()
    groups: set[str] = set()
    partitions: dict[str, int] = {}
    for item in items:
        row = item["row"]
        dataset = str(row["dataset"])
        datasets[dataset] = datasets.get(dataset, 0) + 1
        groups.add(str(row["group"]))
        key = f"{dataset}/fold_{row['partition']}"
        partitions[key] = partitions.get(key, 0) + 1
        if row["label"] in ("PLUME", "POSITIVE_ONLY"):
            positive_groups.add(str(row["group"]))
    return {
        "records": len(items),
        "datasets": datasets,
        "connected_groups": len(groups),
        "positive_groups": len(positive_groups),
        "partitions": dict(sorted(partitions.items())),
        "excluded": len(excluded),
        "emit_records_are_positive_only": all(
            item["row"]["label"] == "POSITIVE_ONLY"
            for item in items
            if item["row"]["dataset"] == "EMIT"
        ),
    }


def _pilot_summary(run_id: str) -> dict[str, Any]:
    path = ROOT / "outputs" / "GTM_Model6" / run_id / "GTM_Model6_plume_summary.json"
    value = _load(path)
    mean = value.get("mean", {})
    return {
        "run_id": run_id,
        "summary_sha256": _sha(path),
        "mean": {
            "reviewed_MARS_pixel_precision": mean.get("reviewed_MARS_pixel_precision"),
            "reviewed_MARS_pixel_IoU": mean.get("reviewed_MARS_pixel_IoU"),
            "MARS_negative_mean_predicted_fraction": mean.get("MARS_negative_mean_predicted_fraction"),
            "EMIT_positive_support_recall": mean.get("EMIT_positive_support_recall"),
            "EMIT_precision": mean.get("EMIT_precision"),
            "EMIT_false_positive_rate": mean.get("EMIT_false_positive_rate"),
            "nationally_validated": mean.get("nationally_validated"),
        },
    }


def build_report() -> dict[str, Any]:
    config_path = MODEL6 / "GTM_Model6_config.json"
    legacy = _legacy_check()
    inventory = _inventory()
    pilots = [_pilot_summary("GTM_Model6_plume_01"), _pilot_summary("GTM_Model6_plume_evidence_02")]
    report = {
        "schema_version": 1,
        "scope": "bounded read-only research readiness gate; no training, download, or held-out label opening",
        "config_sha256": _sha(config_path),
        "available_ram_gib": available_ram_gib(),
        "legacy_contract": {
            "verified_hashes": legacy["verified_hashes"],
            "runtime_load_verified": legacy["runtime_load_verified"],
            "inference_performed": legacy["inference_performed"],
            "tensorflow_installed": legacy["tensorflow_installed"],
            "available_ram_gib_at_check": legacy["available_ram_gib"],
        },
        "current_model6_inventory": {
            **inventory,
            "negative_cap": 32,
            "interpretation": "capped readiness inventory; the completed pilots used their frozen 173-record manifest",
        },
        "completed_pilots": pilots,
        "gates": {
            "legacy_artifact_identity": bool(legacy["verified_hashes"]),
            "legacy_runtime_inference_available": bool(
                legacy["runtime_load_verified"]
            ),
            "quantitative_emit_target_available_in_current_model6_cohort": False,
            "cafo_supervised_methane_rotation_ready": False,
            "user_authorized_for_bounded_implementation": True,
            "native_quantitative_emit_products_available": True,
            "native_paired_sentinel_mapper_ready": False,
            "new_real_pilot_data_ready": False,
            "promotion_candidate_available": False,
        },
        "native_quantitative_emit": {
            "available": True,
            "us_acquisitions": [
                "20241020T170504",
                "20241130T180310",
                "20250922T204933",
            ],
            "products": ["CH4ENH", "CH4SENS", "CH4UNCERT", "CH4PLM"],
            "paired_mapper_ready": False,
            "reason": (
                "The three native quantitative sets are currently used by the EMIT-only diagnostic. "
                "The aligned Sentinel cohort is not established: six cached L2A files include crops that "
                "miss or truncate plume peaks, and no frozen Sentinel/native-EMIT parent manifest proves "
                "complete support, timing, grid alignment and independent splits."
            ),
        },
        "decision": "Do not launch another Model6 sweep on the current nine EMIT-positive cohort. Resolve quantitative EMIT target/provenance and CAFO controls first; retain the existing pilots as failed engineering evidence.",
        "limitations": [
            "The current Model6 EMIT records use positive-only footprint support; exterior pixels are unknown.",
            "The 36 reviewed MARS positives are concentrated in two connected groups.",
            "The local CAFO inventory has no date-specific plume labels or reviewed facility negatives.",
            "Native quantitative EMIT products exist for three US acquisitions, but the paired Sentinel mapper contract is not yet met.",
            "Legacy checkpoint inference is not runtime-verified in this environment.",
        ],
    }
    return report


def _markdown(report: dict[str, Any]) -> str:
    inventory = report["current_model6_inventory"]
    lines = [
        "# GTM_Model6 bounded research gates",
        "",
        f"- Scope: {report['scope']}",
        f"- Decision: **{report['decision']}**",
        f"- Available RAM at gate: {report['available_ram_gib']:.2f} GiB" if report["available_ram_gib"] is not None else "- Available RAM at gate: unavailable",
        "",
        "## Current local cohort",
        "",
        f"- Records: {inventory['records']} ({inventory['datasets']}); negative cap: {inventory['negative_cap']}",
        f"- Connected groups: {inventory['connected_groups']}; positive groups: {inventory['positive_groups']}",
        f"- Excluded during bounded inventory: {inventory['excluded']}",
        f"- EMIT labels positive-only: {inventory['emit_records_are_positive_only']}",
        "",
        "## Native quantitative EMIT availability",
        "",
        "- Native CH4ENH/CH4SENS/CH4UNCERT/CH4PLM sets are available for 20241020T170504, 20241130T180310 and 20250922T204933.",
        "- They are currently an EMIT-only diagnostic resource. A paired Sentinel/native-EMIT mapper manifest is not ready: the six cached L2A crops include missed or truncated plume peaks and do not yet prove complete support, timing, grid alignment and independent splits.",
        "- The data-readiness gates below are not permission gates; bounded implementation/testing is authorized.",
        "",
        "## Gates",
        "",
    ]
    for key, value in report["gates"].items():
        lines.append(f"- `{key}`: **{value}**")
    lines += ["", "## Completed pilot summaries", "", "| Run | MARS pixel precision | MARS IoU | Negative activation | EMIT positive support recall |", "|---|---:|---:|---:|---:|"]
    for pilot in report["completed_pilots"]:
        mean = pilot["mean"]
        lines.append(
            f"| {pilot['run_id']} | {mean['reviewed_MARS_pixel_precision']:.4%} | {mean['reviewed_MARS_pixel_IoU']:.4%} | {mean['MARS_negative_mean_predicted_fraction']:.4%} | {mean['EMIT_positive_support_recall']:.4%} |"
        )
    lines += ["", "## Limitations", ""]
    lines.extend(f"- {value}" for value in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stem", type=Path, default=REPORTS / "GTM_Model6_research_gates_2026-09-22_v2")
    args = parser.parse_args()
    json_path = args.output_stem.with_suffix(".json")
    md_path = args.output_stem.with_suffix(".md")
    if json_path.exists() or md_path.exists():
        raise SystemExit(f"Refusing to overwrite existing gate artifacts: {json_path} / {md_path}")
    report = build_report()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
