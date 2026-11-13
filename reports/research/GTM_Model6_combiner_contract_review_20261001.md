# Model6: preparing the four models and later combiner

> Public source publication note: Referenced generated viewer pages, images, local outputs, and runtime receipts are retained as research pointers but are omitted from this Git publication. Those links require the original local artifacts.

October 1, 2026. This is an engineering rehearsal, not a new methane result.

## What changed

There is now a repeatable check for the training schedule Martin proposed. It reads the existing Model6 observation manifest and keeps the same parent IDs, sites, connected groups, partitions and image grids. It schedules the four context models so the later combiner only learns from predictions made without training on the predicted site. It checks the saved artifacts before accepting those predictions.

The models do not need five unrelated source datasets. The four branches can use the same paired observations, with different input window sizes. The combiner needs a different *form* of training data: their predictions on excluded groups, together with the corresponding valid reference labels. Its final evaluation groups must remain excluded from every fitting stage.

The existing real model still averages four branch logits. No real learned combiner has been trained. The rehearsal uses explicitly artificial branch score files and a tiny fixed logistic combiner to exercise the bookkeeping. That toy combiner is not a decision about the final research architecture.

## What was verified

The [public-command E2E receipt](../../output/GTM_Model6_combiner_contract_20261001/e2e_v2/e2e.json) records **43 passing checks**: ten successful command calls, 32 deliberate contract rejections and one protected-output refusal. The acceptance cases and E2E runner were written before implementation.

- Ten artificial parents in five groups produce 16 inner branch jobs plus four final branch jobs. The 40 saved tiles comprise 32 out-of-fold training predictions and eight outer evaluation predictions. These are scheduled/stub jobs, not 20 trained U-Nets.
- Own-parent, sibling-site, outer-site, shared-acquisition and partition leakage are rejected. Swapped branches, final-fit substitution, coupled branch losses, missing predictions, inconsistent grids/central regions/support, nonfinite scores and stale hashes are also rejected.
- An identical replay produces byte-identical rehearsal output. Removing both outer evaluation target files changes nothing. Changing outer scores changes outputs but leaves fitted weights unchanged; changing inner training scores changes the fit.
- A balanced zero-score fixture gives the independently expected result: zero weights and bias, and probability 0.5 for both the combiner and fixed-mean baseline. Unsupported pixels remain null.
- The real R5 manifest adapter preserves outer fold 1: 164 parents in 11 recorded groups, with 99 fit and 65 evaluation parents. It generates a metadata-only schedule and refuses synthetic fitting on that real-data manifest. No imagery, saved prediction arrays or checkpoint tensors are loaded.
- The existing Model6 status command now exposes this preparation tool while retaining the failed R5 status and reporting that the real learned merger and scientific readiness remain incomplete.

Only Python standard-library JSON, hashes and small scalar calculations were used. Available RAM before execution was 1.82 GiB. The baseline checked 297,576 artifact bytes; the complete latest E2E package, including isolated faulty copies, is 14,852,883 bytes. There was no model runtime, GPU job, imagery transfer, new browser/server, publication or third-party message. Hashes of the R5 manifest/results and inspected source/config files were unchanged across verification.

## Run it again

From the active repository with Python 3.11 or newer:

```powershell
python tools/GTM_Model6_combiner_contract_e2e.py --output-dir output/GTM_Model6_combiner_contract_replay
```

The output directory must be new. This creates the fixture, runs the public CLI, preserves each rejection and writes `e2e.json`. The prior `e2e_v1` directory is retained as the initial 37-check result; `e2e_v2` is the complete result including status integration and the additional cases.

To repeat only the synthetic combiner using the saved fixture:

```powershell
python tools/GTM_Model6_combiner_contract.py rehearse --manifest output/GTM_Model6_combiner_contract_20261001/e2e_v2/fixture/manifest.json --plan output/GTM_Model6_combiner_contract_20261001/e2e_v2/fixture/plan.json --bundle output/GTM_Model6_combiner_contract_20261001/e2e_v2/fixture/bundle.json --output output/GTM_Model6_combiner_replay.json
```

Other public modes are `plan` and `check`; use `--help` for arguments. Plans are pinned to manifest, config and checker code hashes. Regenerate a plan after changing any of those inputs.

## Limits and next integration

This checks one canonical central 16x16 tile per parent, not full-scene mosaics. It checks declared group membership and producer records; it cannot independently prove that a producer honestly recorded its training history or that the geographic grouping is complete. JSON/artifacts are bounded to 4 MiB each and 8 MiB per checked bundle, with at most 256 parents and ten partitions.

The real training runner does not yet consume this nested schedule or export its per-job provenance/score format. That is the next integration before a real learned-combiner experiment, after the paired-data gates are satisfied. The checker does not approve data or resolve EMIT units, background-label validity, quantitative QA, Sentinel radiometry, missing reference images or independent positive/negative coverage. No methane accuracy metric was computed. R5 remains failed reused-development evidence.

## Exact artifacts

- [Acceptance and failure cases](GTM_Model6_combiner_contract_acceptance_20261001.md)
- [CLI](../../tools/GTM_Model6_combiner_contract.py) and [E2E runner](../../tools/GTM_Model6_combiner_contract_e2e.py)
- [Synthetic schedule](../../output/GTM_Model6_combiner_contract_20261001/e2e_v2/fixture/plan.json), [artifact bundle](../../output/GTM_Model6_combiner_contract_20261001/e2e_v2/fixture/bundle.json), and [toy output](../../output/GTM_Model6_combiner_contract_20261001/e2e_v2/receipts/rehearsal.json)
- [Existing R5 metadata schedule](../../output/GTM_Model6_combiner_contract_20261001/e2e_v2/r5_metadata_only/plan.json)
- [Model6 status receipt](../../output/GTM_Model6_combiner_contract_20261001/e2e_v2/receipts/model6_status.json)

E2E receipt SHA256: `eee98b767f9ffb684b7503a51387695cda02144fea0a389f9ee9c1d3e9bb4523`.

CLI SHA256: `d9dd9bacca7da45d60740eeedb3004624194edc819c76099c9b6d9a6b8f9a484`.

E2E runner SHA256: `2ab99cf9244275b07aebb6533490510738ff25875e6cea3176d923e259395611`.
