#!/usr/bin/env python3
"""Read-only MARS baseline import and CUDA probe. Never opens training data."""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/marss2l-code"
REVISION = "f7d264c2c845dfba1cb27f76ef6026275f8d8758"
MODULES = (
    ("marss2l.dataframe_image_plumes", ("process_images_and_plumes", "read_csv_images", "read_csv_plumes")),
    ("marss2l.loaders", ("DatasetPlumes",)),
    ("marss2l.loss", ("DeltaXCH4Loss",)),
    ("marss2l.models", ("SegmentationModelMARSS2L",)),
    ("marss2l.mars_sentinel2.plumesimulation", ("PlumeSimulator",)),
    ("marss2l.mars_sentinel2.transmittance_to_ch4", ("TransmittanceCH4InterpolationFromDict",)),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, help="Installed isolated site-package target")
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled", DO_NOT_TRACK="1")
    if args.overlay:
        if not args.overlay.is_dir():
            parser.error(f"Overlay absent: {args.overlay}")
        sys.path.insert(0, str(args.overlay.resolve()))
    sys.path.insert(0, str(SOURCE.resolve()))
    result = {"source": str(SOURCE), "revision_expected": REVISION, "imports": {}, "base": {}}
    try:
        head = (SOURCE / ".git/HEAD").read_text(encoding="utf-8").strip()
        actual = (SOURCE / ".git" / head.removeprefix("ref: ")).read_text(encoding="utf-8").strip()
        result["revision_actual"] = actual
        if actual != REVISION:
            raise ValueError(f"Source revision mismatch: {actual}")
        for module, _ in MODULES:
            path = SOURCE / (module.replace(".", "/") + ".py")
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        import torch
        import numpy
        result["base"] = {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numpy": numpy.__version__,
        }
        if not (sys.version_info[:2] == (3, 11) and torch.__version__ == "2.11.0+cu128" and numpy.__version__ == "2.4.4"):
            raise ValueError("Base Python/Torch/NumPy versions changed")
        for name, attributes in MODULES:
            try:
                package = importlib.import_module(name)
                for attribute in attributes:
                    getattr(package, attribute)
                path = Path(package.__file__).resolve()
                if not path.is_relative_to(SOURCE.resolve()):
                    raise ValueError(f"Module imported outside pinned source: {path}")
                result["imports"][name] = {"ok": True, "path": str(path)}
            except Exception as error:
                result["imports"][name] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        if result["imports"]["marss2l.mars_sentinel2.transmittance_to_ch4"]["ok"]:
            physics = importlib.import_module("marss2l.mars_sentinel2.transmittance_to_ch4")
            lookup = physics.TransmittanceCH4InterpolationFromDict()
            amf = float(lookup.air_mass_factor(30.0, 10.0))
            if not math.isfinite(amf) or amf <= 0:
                raise ValueError(f"Invalid air mass factor: {amf}")
            result["physics"] = {
                "source_lut": str(Path(physics.LUT_PUBLIC_FILE).resolve()),
                "source_lut_amf_count": len(lookup.amf_arr),
                "background_concentration": float(lookup.background_concentration),
                "air_mass_factor_sza30_vza10": amf,
                "marshsi_version": metadata.version("marshsi"),
            }
        result["overlay_versions"] = {}
        for line in (HERE / "requirements-minimal-locked.txt").read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            name, expected = line.split("==", 1)
            try:
                actual = metadata.version(name)
            except metadata.PackageNotFoundError:
                actual = None
            if actual != expected:
                result["overlay_versions"][name] = {"expected": expected, "actual": actual}
        result["ok"] = all(item["ok"] for item in result["imports"].values()) and not result["overlay_versions"] and result["base"]["cuda_available"]
    except Exception as error:
        result["fatal"] = f"{type(error).__name__}: {error}"
        result["ok"] = False
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
