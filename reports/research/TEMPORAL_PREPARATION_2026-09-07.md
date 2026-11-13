# Research preparation without imagery

The useful preparation boundary is a tested workflow and an explicit data contract. Additional architecture variants or synthetic accuracy tuning would not resolve the scientific uncertainty. No imagery acquisition is needed to run the preparation tools below.

## Prepared components

- `tools/validate_temporal_manifest.py`: local metadata preflight for sample identity, partition declarations, geographic/event group overlap, exact coordinate overlap, provenance and annotation availability. Optional file existence checks do not read raster payloads or certify imagery quality. Different coordinates can still belong to the same site; geography needs source-based adjudication.
- `EarthRemoteSensingRapidResponse/temporal_attention_experiment.py`: an in-memory training engine with separate dense and scene stages. It restores the dense checkpoint chosen by validation dense loss before scene fitting. All-invalid targets are excluded from the scene objective. It accepts only explicit train/validation batches and rejects overlapping sample IDs.
- `tools/rehearse_temporal_experiment.py`: a bounded CPU rehearsal of both actual candidate architectures, with separate generated training and validation tensors, a missing dense annotation, an unavailable reference and an invalid target. The generated partitions are software fixtures, not statistically independent satellite evaluation sets.
- `configs/temporal_attention_comparison_draft.json`: comparison design, fixed candidate identities, seed pairing, selection rules and remaining decisions. It is explicitly a draft; no real experiment should treat it as a frozen protocol.

## Run the rehearsal

From the research checkout, with the existing PyTorch environment:

```powershell
$env:PYTHONPATH = (Resolve-Path '.venv/Lib/site-packages').Path
python tools/rehearse_temporal_experiment.py --output-dir reports/experiments/temporal_rehearsal_2026-09-07
```

The destination must not already exist. Each run records source hashes, runtime configuration, selected epochs, histories and final status. It neither overwrites the previous synthetic diagnostic nor exports a scientific checkpoint. A failed run leaves its failure status for inspection. Re-running requires a new directory and should have a specific verification reason.

## Remaining work once data arrives

The illustrative `configs/temporal_manifest_example.json` contains fictional identifiers, coordinates and nonexistent paths. Run `python tools/validate_temporal_manifest.py configs/temporal_manifest_example.json` to exercise the metadata contract; adding `--check-files` correctly rejects it. A metadata pass is explicitly reported with `imagery_validated: false`. Provenance must identify a particular observation/product instance, not just the provider name.

Verify one complete example before implementing the product-specific loader: observed band order (never the legacy stack containing the EMIT label as an input), reflectance scale/offset, native GSD, coordinate transforms, temporal dates, reference eligibility, cloud/nodata QA and annotation meaning. Compute native physics from verified inputs; the synthetic deficit channel is not an MBMP implementation.

Then establish geographic/event groups, audit overlap with past evaluation sites, assign and hash partitions, and freeze the draft's remaining decisions before evaluation. A training-only resource pilot should set crop/batch sizes and equal optimization budgets. The in-memory runner is intended for that small pilot; it does not yet provide streaming raster loading, augmentation, distributed training, resume checkpoints, calibration or a final test evaluator.

Thresholds and acceptance margins depend on the intended operating constraint and how many independent negatives are available. Leaving them explicitly unresolved now avoids pretending a tiny cohort can establish a stringent false-positive rate. No test outcomes should be inspected to choose them.

## User action

No immediate user action is required for this preparation. An eventual dataset location or an efficient acquisition route is the remaining external input. GPU rental and large downloads are unnecessary for these software checks.

## Observed verification

The focused suite passed 51 tests on 2026-09-07. Both actual candidate architectures completed the three-epoch dense / three-epoch scene CPU rehearsal in `reports/experiments/temporal_rehearsal_2026-09-07/`. Each partition had four synthetic samples: two eligible for dense supervision, three for scene supervision, one missing annotation and one invalid scene. The counters matched those expectations for both models.

The CLI example passed metadata-only validation and failed file-presence validation as intended. A repeat into the existing rehearsal directory raised `FileExistsError`, and all existing artifact bytes remained unchanged. No satellite imagery, network acquisition or held-out methane outcomes were used. These checks establish software behavior only.
