# Tessera engineering checks

These are preserved research harnesses, separate from the scaffold's unit tests.

- `controller_verify_scaffold.py` repeats the offline synthetic CLI flow and independent ranking, export, abstention, and pin checks. A November 7 controller review reported ten passing unit tests plus saved-file CLI checks. It established no methane or real-facility performance.
- `smoke.py` checks a saved Sentinel-2 crop with independently supplied raster files and a matching receipt. It requires `--crop-dir`, `--receipt`, and `--output-dir`; the crop and receipt are not published here.
- `cambridge_live/fetch.py` is a bounded downloader for the public Cambridge 2024 example. Its Source Cooperative URLs are public. Running it requires an explicit `--cap-bytes` limit. It writes into ignored `cambridge_live/local-run/`.
- `cambridge_live/check.py` checks a local Cambridge 2024 tile through GeoTessera 0.11.0, then exercises the scaffold CLI. It requires the local cache and download receipt from `fetch.py`; it blocks network calls after loading them.

The historical Cambridge 2024 and GeoTessera 0.11.0 smoke checks are engineering evidence only. Dataset bodies, caches, environment details, download receipts, and generated exports are excluded from this source publication. None of these scripts was run during this publication audit, and no live request was made.
