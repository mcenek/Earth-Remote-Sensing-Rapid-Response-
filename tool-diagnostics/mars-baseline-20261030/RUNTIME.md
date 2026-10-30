# MARS baseline import overlay

The cached UNEP source is at commit `f7d264c2c845dfba1cb27f76ef6026275f8d8758`.
Its package metadata requires Python 3.12. The six baseline modules probed here
parse on the existing Python 3.11.14 interpreter. An unrelated plotting module
uses Python 3.12 f-string syntax and must stay outside this import path.

`requirements-minimal-locked.txt` is the transitive closure of the baseline's
eager import graph in the official `environment/requirements-test.lock`, minus
packages already in the local CUDA environment. It is derived by
`freeze_overlay.py`, which stops at installed packages and preserves the
Python 3.11 override below. It also adds `win32-setctime==1.2.0`, which the
Linux generated official lock omitted but `loguru` requires on Windows. The
source's broad package requirements also name
`cyclopts`, `wandb`, and `torchmetrics`, which are for its official CLI/trainer
and are not imported by the local baseline trainer. `geopandas`, Earth Engine,
CloudSEN12, and Azure packages are present because the source eagerly imports
producer or cloud paths while importing the dataset, even though local FIT
training does not call those paths. No fake package modules are used.

The official lock has `marshsi==0.1.6`, which requires Python 3.12. The
Python 3.11 compatible release is `0.1.4`. The source declares `marshsi>=0.1.1`.
The active dictionary based transmittance class uses the pinned source JSON,
plus `marshsi.lut.air_mass_factor` and
`marshsi.quantification.BACKGROUND_CONCENTRATION`. The source-only comparison
in `marshsi_physics_comparison.json` shows that `marshsi/lut.py`,
`marshsi/quantification.py`, and the packaged NetCDF LUT are byte identical
between the installed 0.1.4 and official lock's 0.1.6 wheel. All `marshsi`
symbols referenced by the baseline are in those identical files. This proves
the `marshsi` physics path is unchanged despite the Python metadata override.
The 0.1.6 wheel was read from PyPI and hash verified, never installed or
executed.

Run `check_pypi_compat.py` before installation to check all pinned release
metadata and Windows Python 3.11 wheels without fetching package binaries.
Use `--no-deps --only-binary=:all:` when installing the lock into an isolated
target. Keep the existing base Torch, NumPy, rasterio, and other installed
packages untouched. Then run `probe_runtime.py --overlay <target>` from the
repo's `.venv/Scripts/python.exe`. The probe checks source revision, imports,
package versions, CUDA, and Python version. It reads no training rasters.

The lock was derived from a Python 3.12 environment and cannot prove Python
3.11 compatibility by itself. Successful wheel metadata checks, import probe,
and a real trainer smoke run are still required.
