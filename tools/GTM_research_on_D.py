"""Run the research tools with data, results, caches and temporary files on D:."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STORE = Path(os.environ.get("GTM_STORAGE_ROOT", "D:/Research/REMOTE-SENSING/ersrr-ordinal-win")).resolve()


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in {"baseline", "review", "viewer"}:
        print("Usage: python tools/GTM_research_on_D.py {baseline|review|viewer} [tool arguments]")
        return 2
    task, arguments = sys.argv[1], sys.argv[2:]
    data = STORE / "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs"
    for relative in ("outputs", "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs"):
        if (ROOT / relative).resolve() != (STORE / relative).resolve():
            raise RuntimeError(f"Active research path is not on the selected drive: {relative}")
    environment = os.environ.copy()
    settings = {"HF_HOME": "cache/huggingface", "HF_HUB_CACHE": "cache/huggingface/hub",
                "TORCH_HOME": "cache/torch", "UV_CACHE_DIR": "cache/uv",
                "PIP_CACHE_DIR": "cache/pip", "TEMP": "tmp", "TMP": "tmp"}
    for variable, relative in settings.items():
        path = STORE / relative
        path.mkdir(parents=True, exist_ok=True)
        environment[variable] = str(path)
    environment.update(WANDB_MODE="disabled", DO_NOT_TRACK="1", HF_HUB_DISABLE_TELEMETRY="1",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    if task in {"baseline", "review"}:
        overlay = STORE / "runtime/mars-baseline"
        if not overlay.is_dir():
            raise FileNotFoundError(f"Training runtime is not installed: {overlay}")
        environment["PYTHONPATH"] = str(overlay)
        script = ROOT / ("tools/GTM_Model8_mars_baseline.py" if task == "baseline" else "tools/GTM_Model8_mars_review.py")
        if "--data-root" not in arguments:
            arguments += ["--data-root", str(data / "publication-v1/external/MARS-S2L")]
        if task == "baseline" and "--output" not in arguments:
            arguments += ["--output", str(STORE / "outputs/GTM_Model8_mars_baseline_20261030/fit")]
    else:
        script = ROOT / "tools/GTM_Model0_local_viewer.py"
    print(json.dumps({"task": task, "storage_root": str(STORE), "code_root": str(ROOT)}), flush=True)
    return subprocess.run([sys.executable, "-u", str(script), *arguments], cwd=ROOT, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
