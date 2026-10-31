# MARS data cache on the campus server

The current baseline needs only part of MARS. Keep the rest of the released dataset available for later experiments in the same cache. The extra download must reuse the baseline files and must not change its training or test split.

## Dataset and size

- Provider: `UNEP-IMEO/MARS-S2L` on Hugging Face.
- Fixed revision: `c26b1d7e31a0c5241fa37c9140802622c215eb32`.
- Complete release: **199,226 files, 96,041,292,441 bytes** (96.04 GB).
- Existing baseline catalog: **67,687 files, 32,607,815,901 bytes**.
- Additional file content needed after the baseline download: **63,433,476,540 bytes**.
- Additional response-body transfer cap: **66,000,000,000 bytes**, including retries. The separate baseline cap stays at 35,000,000,000 bytes.
- The full release includes 5.80 GB under `data/Not Used`, supporting tables, and released reference models. These files are archived for completeness. They are not added to training.

The PC had all 187,014 files in its earlier image catalog, totaling 89.02 GB, at the October 30 inventory. Compared with the complete provider release, it had 187,025 files totaling 89.54 GB. The difference includes the provider's unused archive and supporting files. This was a presence and size check, not a new content-hash audit of the whole PC cache.

The complete catalog was obtained through the provider's paginated recursive file tree. A single dataset-info response did not contain all files. The compressed catalog and its receipt are under `handoff/catalogs/GTM_MARS_full_release_c26b1d7e.*`. The catalog's SHA-256 is `97f510b871f9a99942422978c71a252bfb90551042c9476d5742bba238200be7`.

## Start the supplemental cache job

Use the research branch and an existing user-owned data directory. On the campus account, the cache is `~/gtm-model8/data/MARS-S2L` and the log is `~/gtm-model8/full-release-acquisition.log`.

```bash
cd ~/Earth-Remote-Sensing-Rapid-Response-
python3 -u tools/GTM_Model8_fetch_assets.py \
  --data-root ~/gtm-model8/data/MARS-S2L \
  --catalog handoff/catalogs/GTM_MARS_full_release_c26b1d7e.jsonl.gz \
  --profile full_release --wait-for-cache 21600 \
  --download --max-network-bytes 66000000000
```

Run this in the separate tmux session `gtm-mars-full`. It waits up to six hours for the current acquisition to release its cache claim. It never removes another job's claim. It then checks existing files against provider content hashes and fetches only missing or corrupt files. Do not launch a second copy or delete the claim to bypass the wait.

The supplemental job keeps its own receipt and cumulative transfer ledger:

```text
.gtm_model8_fetch_profile-full_release_receipt.json
.gtm_model8_fetch_profile-full_release_transfer.jsonl
.gtm_model8_fetch_profile-full_release_progress.json
```

The original baseline receipt and transfer budget remain separate. Its waiting training launcher can still proceed when the original catalog is complete. Do not mark the full cache complete until its receipt says `complete`, with zero remaining, corrupt, or failed files.

## Use the cache for research

The download adds file availability. It does not alter the frozen baseline manifest, reserved sites, checkpoint selection, thresholds, or model inputs. Holdout labels remain for evaluation. The provider's `Not Used` folder needs a separate review before any use. A larger training experiment needs its own split and plan.

Keep the provider README and attribution. The data and released weights use CC BY-NC-SA 4.0. Data files stay outside Git. The compressed file catalog contains paths, sizes, pinned public URLs, and expected hashes only.
