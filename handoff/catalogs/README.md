# Model8 required asset catalog

The compressed JSONL lists the public provider files required by the existing frozen Model8 experiment. It does not contain images or model weights. The adjacent JSON receipt records the catalog hash, source catalog hashes, object count, and total size.

- Provider: `UNEP-IMEO/MARS-S2L`.
- Immutable revision: `c26b1d7e31a0c5241fa37c9140802622c215eb32`.
- Required assets: 67,687 files, 32,607,815,901 bytes.
- Scope: frozen FIT, validation and reserved evaluation inputs; FIT methane patterns; provider metadata; pinned released reference.
- Provider data and weights license: CC BY-NC-SA 4.0. The catalog includes the provider's `README.md`.

Use `tools/GTM_Model8_fetch_assets.py` to check an existing cache before transfer. Download only missing files. Supply an explicit cumulative network byte limit. File sizes and content hashes are checked before a download is published. The source snapshot is separate under `handoff/vendor/`.

This catalog preserves the earlier experiment's required objects. It does not select new scenes, expand training to all US imagery, or make the reserved scenes independent of earlier project work.
