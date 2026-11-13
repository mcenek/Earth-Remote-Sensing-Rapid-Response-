# Model 7: missed-plume validation study

> Public source publication note: Referenced generated viewer pages, images, local outputs, and runtime receipts are retained as research pointers but are omitted from this Git publication. Those links require the original local artifacts.

October 26, 2026. This study did not reach useful 70% plume F1. None of the three new models passed the selection checks. Keep v11 as the research candidate; it is still not ready for a US-wide methane survey.

## Open the actual predictions

[Three-model comparison, including every assessment scene](http://127.0.0.1:8766/bundle-assets/GTM_Model7_validation_v12_assessment_candidate/_versions/cb380f19c0644e98b03ea8f3a1f879e8/index.html)

[Assessment predictions in the map viewer](http://127.0.0.1:8766/?catalog=all&experiment=GTM_Model7_validation_v12_assessment_candidate&scene=MARS_e963d08d-2f2b-436c-946d-c87ec871cfb3#compare)

These private links require access to the existing Tailscale network and the viewer computer to remain on. For use on this computer, replace the host with `http://127.0.0.1:8766`. No training runs when you open a result.

1. Select a scene in the comparison report. All 144 observations remain available, including clear scenes and complete misses.
2. Select **Errors against annotation**. Teal marks correct overlap, purple marks missed plume pixels, and pink marks extra prediction pixels. Gray pixels are outside scored support.
3. Select **Continuous score** to inspect the saved model scores. These scores are not calibrated methane probabilities or concentrations.
4. Use the three map links to inspect the same scene in each model. Model switching keeps the scene. The frozen cutoff is 0.70 for these failed diagnostics.
5. Read the whole-run results before using an individual good example. For example, Kazakhstan `e963d08d` reaches 89.7% F1 in the fraction model, but the complete assessment result is only 39.7%.

The viewer registry now contains nine new bundles: control, native-label loss and native-label loss plus ratio, each for selection validation, locked assessment and old-site regression. They are marked as failed research comparisons. The earlier viewer default was not replaced.

## Validation that was fixed before fitting

The protocol is `docs/GTM_Model7_validation_v12_protocol_20261026.md`. Its SHA256 is `133879978a506ab2145f29aa97514045c5a453eb56927cc74982158b465fe48f`.

| Role | Plume observations | Clear observations | Sites | Linked groups | Use |
|---|---:|---:|---:|---:|---|
| FIT | 188 | 482 | 171 | 26 | Training and normalization only |
| SELECT | 39 | 105 | 27 | 16 | Select checkpoint and cutoff |
| ASSESS | 41 | 103 | 26 | 13 | Evaluate after all choices were frozen |
| Old regression | 46 | 152 | 55 | Historical set | Compare with previous work; never select a new model |

All dates and classes from one physical site stay together. Sites are also joined when they are within 25 km, share target/reference acquisitions or observation identity, or share spectral image hashes. These linked groups do not cross the new FIT/SELECT/ASSESS boundaries. Normalization uses FIT only. The old regression set is outside the selection process; it is not claimed to meet the new three-way separation rules.

There are two material limits:

- All these sites were available to earlier project experiments. They were withheld from these fresh fits, but they are not pristine independent test sites.
- The group constraints left all 97 US-positive observations from the original training set in FIT. SELECT and ASSESS contain no US-positive observations. The old regression set has only three. This study cannot establish US-wide performance.

The split was assigned from metadata, without using prediction quality. No poor scene was removed or moved later. Assessment and old-regression arrays were opened only after all three model choices were frozen. Previously consumed final-test arrays were not used.

## What changed

Three matched models were trained from fresh weights, each for 4,800 updates. They share the same initial common weights, seed, scene draws, rotations/reflections, optimizer, full image inputs and training budget. Each batch contains four real plume and four real clear observations. Sampling chooses sites uniformly within each class, then observations within a site. Normalization is fitted again using only the new FIT set.

| Model | Change |
|---|---|
| Fresh control | Existing 24-channel temporal U-Net; a 20 m training cell is positive if any of its four native label pixels is positive |
| Native-label loss | Same model, but train against the fraction of native positive pixels in each cell; this removes implicit label expansion |
| Native-label loss + ratio | Same fractional loss plus a validity-aware current/reference B12/B11 feature computed from existing raw bands |

The control and fraction models each have 121,937 parameters. The ratio model has 122,081; its added first-layer input starts at zero, while the other initial weights match. These are compact temporal U-Net comparisons, not a new implementation of Martin's three independently trained CNN stages.

All use AdamW, learning rate 0.0005, weight decay 0.0001, positive BCE weight 2, pooled Dice weight 0.1 and gradient clipping at 1. No mixed precision or TF32. The input is the full 100 by 100 grid. Predictions are repeated onto the native label grid for scoring, without dilation, filling, flooding or label-guided cropping. Fractional labels do not create finer-resolution spectral information.

The FIT-only audit found that the old target rule expands labeled area by 8.46%. The ideal block-constant 20 m representation can still achieve about 97% F1 on those FIT labels. Thus the grid rule is worth fixing, but it cannot explain most complete plume misses by itself.

## Selection and results

Each model had four fixed checkpoints and 22 fixed cutoffs. None of the 88 choices per model passed all selection checks: precision at least 70%, recall at least 30%, clear-pixel false-positive rate at most 0.1%, clear-scene activation at most 15%, plus the recovery check in the protocol. The useful target was stricter: F1 at least 70% and recall at least 65%, with the precision and clear-scene checks also passing on SELECT and ASSESS.

Because none qualified, the predeclared failure fallback was used for each displayed diagnostic: update 1,200, cutoff 0.70. No model was promoted. These fallback scores are not the maximum achievable F1 among the candidates. The highest unconstrained SELECT score was 46.68% for the ratio model at update 3,600/cutoff 0.80; it still failed precision and clear-scene checks. No candidate reached 70% SELECT F1, even without the guards.

Locked ASSESS results at the frozen diagnostic settings:

| Model | Pixel F1 | Precision | Recall | Complete plume misses | Clear scenes activated | Clear-pixel false-positive rate |
|---|---:|---:|---:|---:|---:|---:|
| Fresh control | 34.32% | 76.28% | 22.14% | 27 / 41 | 10 / 103 | 0.0684% |
| Native-label loss | 39.69% | 69.53% | 27.78% | 21 / 41 | 21 / 103 | 0.1301% |
| Native-label loss + ratio | 38.99% | 73.64% | 26.52% | 25 / 41 | 14 / 103 | 0.0733% |

These pixel precision and F1 totals include false-positive pixels from both plume and clear observations. A complete miss means zero correct plume pixels. A clear-scene activation means any positive prediction on its scored support, not a confirmed new methane source.

Compared with the control on ASSESS, the fraction model marks 37.62% more pixels, gains 25.45% more correct plume pixels, but adds 76.74% more false-positive pixels. That is not a sufficient fix for the requested problem. The ratio model helps some real plume traces but also completely misses plumes the control finds.

At the common 0.50 display cutoff, training F1 rises to about 67-68%, while SELECT stays much lower. This is evidence that the fits learn training patterns, but do not transfer reliably to held-out sites. It does not establish a unique cause such as architecture, reference quality or limited diversity.

Old regression F1 is 31.04%, 33.45% and 42.52%, respectively. Historical v11 scored 53.06% on those same 198 observations, with precision 71.48% and recall 42.19%. V11 used a larger training set and a different seed/sampler, so this is a regression comparison, not an isolated architecture experiment. It remains an exposed-development score.

## Visual sanity check

The main agent inspected all 41 ASSESS-positive and all 39 SELECT-positive observations, the ten largest ASSESS clear-scene false alarms, and representative old-regression cases including all three US positives and large clear-scene errors. Contact sheets preserve the review; all other scenes remain in the interactive report.

- `e963d08d`, Kazakhstan: all three trace the plume reasonably well. The fraction model gives F1 89.70%, precision 98.19%, recall 82.56%, with 37 extra native pixels. This is one example, not an overall validation score.
- `0428ae40`, Algeria: all three miss the labeled plume; the ratio model marks 656 incorrect pixels elsewhere. The reference also has a visible broad color/haze difference. It needs a separate reference-quality review, not removal from this result.
- `38251004`, Algeria: the fraction model improves a real plume trace to 78.2% F1 while keeping few extra pixels.
- `658b1f01`, Algeria: all three miss the complete plume. Several offshore and Yemen examples also remain complete misses.
- `e1ff8c4f`, labeled clear, Algeria: the models mark surface blobs, adding 636, 1,064 and 792 pixels. These are scored as false alarms against the released labels; they are not verified methane detections.
- Old US `9123854a`: the ratio model recovers much more labeled plume than the fresh control. Other countries still have complete misses, so the US example does not justify national deployment.

The failures are present in saved score maps and masks. They are not explained by a missing viewer overlay. The report and main map show the same source-bound scene and counts.

## Verification and repeatable artifacts

Root folder: `outputs/GTM_Model7_validation_v12_20261026`.

- `data/`: frozen split, source-linked manifest and boundary receipt.
- `audit/fit_label_audit.json`: FIT-only label-area and representation audit. `audit/crossrole_check/data/` reproduces the split after matching target/reference observation-key namespaces; membership is unchanged.
- `preflight/receipt.json`: failure controls and real-data feed-forward/backward checks completed before fitting.
- `fit_*` and `replay_*`: four checkpoints per model, training histories, fresh CPU replays. Saved scores, support and native counts reproduce exactly.
- `selection/`: all 264 measured choices, frozen failed choices and all-model freeze receipt.
- `evaluation_*`: all 1,458 saved model/scene predictions across the three evaluation roles, plus exact fresh-process CPU replays.
- `review_*`: original TIFF reconstruction checks, exact native count recomputation, report pages and viewer exports. 486 scene reconstructions were checked across the roles.
- `http_*.json`: 10,215 served assets checked against the saved bundles, including mask alpha, thresholds and counts.
- `browser/`: actual screenshots and browser check receipt for scene/model switching, score/error modes and reload.
- `tailscale_receipt.json`: private HTTPS report matches local bytes, with normal certificate verification.
- `summary.json`, `selection_precision_recall.png`: results, complete-miss IDs and the precision/recall tradeoff. Group-bootstrap intervals in the summary are descriptive for these historically exposed groups, not population guarantees.
- `completion_receipt.json`: hashes bind the protocol, implementation and verification artifacts.

To repeat the existing saved-artifact verification from the repository root, without training or changing choices:

```powershell
$runRoot = 'outputs/GTM_Model7_validation_v12_20261026'
$pythonExe = '.venv/Scripts/python.exe'
foreach ($role in @('dev', 'assessment', 'regression')) {
    & $pythonExe tools/GTM_Model7_validation_v12_infer.py --root $runRoot --split $role --verify
    & $pythonExe tools/GTM_Model7_prediction_publication_e2e.py --review "$runRoot/review_$role" --output "$runRoot/http_$role.json"
}
```

To reproduce a full new study, use a new output directory, preserve the protocol, and run in this order: metadata-only split; FIT-only audit; preflight; the three fits and their CPU replays; selection freeze; inference for each role; inference replays; review recipes; renderer; contacts; HTTP and actual browser review. Do not retrain in the existing directories or choose weights after reading ASSESS.

## What to do next

1. Reserve additional paired US plume and clear sites prospectively, before training. Keep nearby sites and shared acquisitions together. The current cache cannot supply a pristine new positive-site test.
2. Audit reference atmosphere, cloud support and temporal matching using a written rule. Review label and reference quality without dropping cases merely because the model misses them.
3. In a new protocol, test the cached exogenous wind inputs or another justified change on FIT/SELECT only. Methane flux metadata must not be an input: it tracks the label and would leak the answer. Do not start another threshold sweep on this now-exposed ASSESS set.
4. Keep the measurement claim precise. These runs use binary MARS-S2L annotations and Sentinel-2 L1C bands. They do not demonstrate quantitative NASA EMIT methane-map upscaling. Cached EMIT pilot pairs have footprint and temporal limitations, and raw enhancement units still need verification. EMIT is satellite data, not airborne data.

No new imagery was downloaded, no national inference was run, and no result was pushed to GitHub or written to the shared research journal in this study.
