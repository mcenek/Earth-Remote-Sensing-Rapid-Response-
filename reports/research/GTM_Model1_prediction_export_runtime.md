# GTM Model 1 prediction export runtime

Date: 2026-09-18

The available Windows interpreter is `C:/Users/joshu/.pyenv/pyenv-win/versions/3.11.9/python.exe` with `ersrr-ordinal-win/.venv/Lib/site-packages` on `PYTHONPATH`. Native NumPy and rasterio imports succeed when executed with the approved native-DLL environment. Installed packages include NumPy 2.4.4, rasterio 1.4.4, tifffile 2026.3.3, and PyTorch 2.11.0+cu128. TensorFlow is absent, but the exact `compact_resunet_v1` artifact loaded successfully with Keras 3.14.0 and the Torch backend. Only lightweight Keras/HDF5 dependencies were added; no TensorFlow was installed.

The legacy checkpoint files were found only in the sibling checkout and remain read-only inputs:

| artifact | bytes | SHA-256 |
|---|---:|---|
| `checkpoint.weights.h5` | 10,898,424 | `97B7BA336C80702D2337E8E11C252449C5D9FE919E855ABEA18853957E2FDA11` |
| `checkpoint_new.weights.h5` | 11,717,992 | `37921517E99A9A4AC7CFB829A96C15A09924FFC57AC072C0507785712DD40759` |
| `dice.weights.h5` | 9,631,136 | `46DD72ADE3FEF969FDDEB02BE4949B72377A17C90D76CC1D67E024B384C4F396` |
| `x_max.npy` | 132 | `895DAE4C18D571BF7CA9E26E13DBEBCBE89C114494E56748B01802ABDBEEB1F0` |

The legacy two-head checkpoint remains unavailable. GTM_Model1 therefore preserves the exact compact_resunet_v1 artifact identity from the sibling checkout. It loaded with input `(128,128,5)` and output `(128,128,1)`. Its declared preprocessing is `log1p` on B2/B3/B4/B11/B12 followed by config z-score normalization; its output is a plume probability mask with decision threshold 0.01. No score table or scene-level value was converted into a spatial mask.

`outputs/GTM_Model1_viewer_export/` contains two real 256×256 six-band US validation source exports with model inference and:

- `rgb.tif`: display-stretched Sentinel-2 B4/B3/B2 RGB;
- `emit_reference.tif`: original sixth band with `-9999` converted to NaN;
- `valid_mask.tif`: finite, non-`-9999` EMIT validity mask;
- Canonical model outputs: `probability_{compact,physics}_256_native.tif` and `mask_{compact,physics}_256_native.tif`, stitched native-grid inference; earlier 128px/full-footprint intermediate files are superseded and must not be used;
- `display_*_3857.tif`: shared 128×128 EPSG:3857 RGB, probability, reference mask, and valid mask layers;
- `overlay_compact_3857.json` and `overlay_physics_3857.json`: separate canonical model arrays on the shared display grid; the unqualified intermediate JSON is superseded;
- `metadata.json`: source SHA-256, band order, CRS, native bounds, WGS84 Leaflet bounds, valid fraction, acquisition filename date, and explicit reference semantics;
- `manifest.json`: two reference cases and two prediction cases.

The EMIT layer is labeled as the aligned paired reference from the source TIFF and is not described as independent truth. Truth masks use `EMIT_CH4 > 300 ppm*m` only where finite and non-`-9999`; nodata remains unknown through `valid_mask`. The two US validation tiles are WGS84 and have verified US geographic coordinates. Their filename dates provide Sentinel acquisition dates, but the filenames do not contain an EMIT product/date identifier, so any Sentinel/EMIT date gap remains unknown.

Final inference uses four native 128×128 windows stitched over each 256×256 source tile, preserving the source approximately-20m grid; the 128×128 EPSG:3857 display grid is a visualization resampling. Compact and physics model layers are both retained, with compact as the manifest default. The compact prediction ranges / count of native pixels at probability ≥0.5 are: T14RMU `0.0493–0.9538`, `49,794`; T13RFQ `0.0026–0.9224`, `25,759`. The physics ranges / counts are: T14RMU `0.0238–0.9805`, `34,022`; T13RFQ `0.0021–0.9701`, `28,005`. Artifact thresholds are compact `0.01` and physics `0.05`; these are research outputs, and the calibration reports very low specificity, so masks should not be presented as operational alerts.
