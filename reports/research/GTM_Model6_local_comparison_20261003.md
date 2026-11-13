# Local model comparison, October 3, 2026

> Public source publication note: Referenced generated viewer pages, images, local outputs, and runtime receipts are retained as research pointers but are omitted from this Git publication. Those links require the original local artifacts.

We trained two small CPU models and compared 12 methods on the same cached images. Neither new model improved plume overlap at the fixed cutoff. Requiring three of the four saved neural branches to agree reduced false-positive pixels, but it also removed real plume pixels. These are useful diagnostic results. They do not establish a working methane detector.

[Open the visual comparison](http://127.0.0.1:8766/bundle-assets/GTM_Model6_local_cpu_comparison_20261003/comparison.html). Use the Scene menu to compare all three positive cases and three selected controls. Orange marks predictions. Cyan marks the reference boundary. Stripes mark pixels outside the evaluated output.

## What ran

- Two newly fitted, class-balanced logistic models: temporal SWIR contrast alone, and all 16 cached Sentinel features. These have no learned spatial context. The extra evidence channel was excluded.
- The four saved R5 window models, the saved fused result, arithmetic probability averaging, fixed three-of-four and four-of-four agreement, and constant positive/negative controls.
- The original 99 training scenes and 65 evaluation scenes. All methods used the same 503,044 evaluated pixels, including 1,513 positive pixels. Only three evaluation scenes contain plumes, and all belong to one connected positive group.
- Training used 15,934 positive and 50,688 negative sampled pixels. Both fits converged in four updates. Fitting, scoring, export, and input hash checks completed in 5.307 seconds.
- No imagery or weights downloaded. No GPU or package installation. The current interpreter has NumPy and Pillow, but does not have Torch or TensorFlow. Free memory before the run was approximately 3.1 GiB.

This uses the existing MARS binary-mask supervision. It is not new quantitative EMIT training, upscaling, or the proposed three-CNN model. The original labels and partition were inherited, not independently relabeled. The evaluation fold has already been examined in previous research.

## Results on identical pixels

IoU measures the overlap between predicted and reference plume masks. Precision is the fraction of predicted positive pixels that match the reference. Recall is the fraction of reference plume pixels recovered. All values below use the predeclared cutoff of 0.50.

| Method | IoU | Precision | Recall | False-positive pixels |
| --- | ---: | ---: | ---: | ---: |
| Saved R5 mean | 2.15% | 2.25% | 32.52% | 21,387 |
| S128 branch | 1.55% | 1.58% | 47.85% | 45,232 |
| Probability mean | 2.17% | 2.28% | 32.52% | 21,124 |
| At least three branches agree | 3.41% | 3.83% | 23.60% | 8,952 |
| All four branches agree | 2.06% | 5.36% | 3.24% | 866 |
| New SWIR-only model | 0.66% | 0.66% | 62.59% | 142,092 |
| New 16-feature model | 0.54% | 0.54% | 56.31% | 157,161 |

Three-of-four agreement removes 58.14% of R5's false-positive pixels, but loses 135 of its 492 true-positive pixels. Every one of the 62 negative scenes still has some activation. This is not a validated scene-level detection rule. Four-of-four agreement suppresses most output and misses almost all of the plume pixels.

The two simple models recover more reference pixels mainly by marking much more background. Their scores are not calibrated probabilities. Fixed-cutoff performance does not prove that all linear models, feature sets, or training schemes will perform this way. No threshold or hyperparameter was changed after inspecting these results.

## Visual observations

All six selected cases were inspected in the local browser. Selection was fixed before scoring: all positives plus the two largest false alarms for R5 and each new model, with duplicates removed. It is a diagnostic selection, not a representative sample.

- November 1 positive: R5 overlaps part of the plume. S128 has better overlap on this one image, but more background activation. Both new models also mark roads and surrounding surfaces. Requiring more agreement progressively erases the plume.
- June 4 positive: R5 and S128 miss the reference plume entirely. The new models overlap some of it but also mark large parts of the facility and surrounding surfaces. Agreement does not recover the missing plume.
- November 21 positive: R5 overlaps part of the reference. S128 and the simple models mark broader horizontal surface features. Three-of-four agreement reduces some of these errors and some true output.
- January 25 and October 27 controls: the new models mark broad areas despite the empty reference masks. Agreement makes the false alarms smaller but does not remove them.
- November 26 control: the 16-feature model marks much of the facility. R5 and three-of-four agreement activate less, but still produce false positives.

The saved arrays and rendered images agree. The weak whole-run result is not explained by a missing prediction overlay. Saved browser screenshots are in [the verification folder](assets/GTM_Model6_local_comparison_20261003/).

## Verification and repeatability

The [protocol](GTM_Model6_local_comparison_protocol_20261003.md), failure cases, and public-command E2E runner were written before implementation and the real run. All 11 recorded E2E checks passed. They include a known zero-feature solution, deterministic replay, rejection of group leakage and changed inputs, refusal to overwrite results, and fitting with evaluation targets removed.

The real run matched all 325 original scene/branch confusion counts. A separate saved-output audit checked all 780 scene/method confusion counts, all 12 aggregate totals, all 65 prediction file hashes, the six viewer grids and their overlay masks, and all 76 report assets through the local server. The new entry loads in the normal viewer. Native and displayed IoU match for the inspected first scene. No browser console errors were observed.

- [Saved-output verification receipt](assets/GTM_Model6_local_comparison_20261003/verification.json)
- [E2E receipt](assets/GTM_Model6_local_comparison_20261003/e2e.json)
- [Native results](../../outputs/GTM_viewer_bundles/GTM_Model6_local_cpu_comparison_20261003/results.json)
- [Model parameters and fit membership](../../outputs/GTM_viewer_bundles/GTM_Model6_local_cpu_comparison_20261003/models.json)
- [Input hashes and runtime](../../outputs/GTM_viewer_bundles/GTM_Model6_local_cpu_comparison_20261003/provenance.json)

From the repository root, with the current Python environment:

```powershell
python tools/GTM_Model0_local_viewer.py --port 8766 --name Research-PC
```

The server is already running for this session. Open the comparison link above. The new 16-feature model is also available under Full research archive as `Local CPU baseline | all 16 features | October 3`. The presentation defaults stay on the existing curated examples.

To repeat the experiment or E2E checks, use a new output directory:

```powershell
python tools/GTM_Model6_local_comparison.py --output outputs/GTM_viewer_bundles/GTM_Model6_local_cpu_comparison_20261003_replay --minutes 5
python tools/GTM_Model6_local_comparison_e2e.py --artifact-dir output/GTM_Model6_local_comparison_20261003/e2e_replay
```

Original benchmark files retained their hashes. All 65 new native outputs and both fitted models are saved locally. Nothing was promoted, pushed, or written to the external research journal.

## Implication for the next experiment

Keep these as inexpensive controls for the next real training run. The immediate lesson is to measure false activation on controls alongside plume overlap. A more restrictive merger alone does not solve the problem.

Do not use this repeatedly examined positive group to select a new architecture. The proposed three-CNN experiment still needs a defined paired airborne dataset, verified EMIT supervision and a separate set of positive and negative groups. The current machine can support bounded CPU comparisons; larger training also needs an appropriate runtime and memory. These runs leave those scientific requirements unchanged.
