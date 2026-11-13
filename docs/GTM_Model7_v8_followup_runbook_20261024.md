# U-Net follow-up, 24 October 2026

> Public source publication note: Referenced generated viewer pages, images, local outputs, and runtime receipts are retained as research pointers but are omitted from this Git publication. Those links require the original local artifacts.

The new site-balanced fit recovers more plume pixels on development data. It also creates more false alarms. Keep v7 as the selected baseline. Both new methods remain visible as diagnostic comparisons. Neither is ready for a US survey.

## What was done

Used the same 958 real training observations: 268 plume and 690 no-plume. Used the same 198 exposed development observations: 46 plume and 152 no-plume, at 55 sites in 12 countries. Development observations stayed out of gradient updates. These project locations have already been evaluated. They are not an independent test.

1. Averaged eight rotated and reflected views of the frozen v7 model. No weights changed. The inverse transforms and original-view predictions were checked against saved arrays.
2. Trained one site-balanced U-Net for 4,800 updates. Kept the architecture, seed, loss, full-scene inputs, optimizer and 50/50 plume-clear batches fixed. Changed only the sampling rule: select a physical site uniformly within each class, then select an observation at that site. Positive US draws fell from 50.0% to 20.6%. All 958 training observations were sampled.

The site-balanced model used the same current image, reference image, signed difference and absolute difference as v7. Its selected checkpoint is update 2,400 at cutoff 0.75. Selection used four checkpoints and 22 cutoffs under the written protocol. The baseline is update 1,200 at cutoff 0.70. The comparison therefore measures the complete predeclared selection procedure, not the isolated sampler at matched checkpoints and cutoffs.

## Same development set

| Measure | Frozen v7 | Eight-view mean, selected cutoff 0.65 | Site-balanced candidate |
|---|---:|---:|---:|
| Plume overlap, IoU | 28.13% | 31.16% | 36.24% |
| Plume-pixel precision | 65.18% | 66.67% | 66.49% |
| Plume-pixel recall | 33.10% | 36.91% | 44.34% |
| Mean positive-site recall | 29.75% | 31.77% | 37.59% |
| Recover at least 10% of plume pixels | 23/46 | 24/46 | 24/46 |
| Clear observations with any positive pixel | 20/152 | 18/152 | 23/152 |
| Clear-pixel false-positive rate | 0.08489% | 0.08760% | 0.09546% |

Both candidates pass the ordinary development eligibility limits. Both fail the stricter replacement rule, which requires improved mean positive-site recall, nondecreasing IoU and no increase in clear-pixel false-positive rate. Do not relax this rule after seeing results. Keep the recorded v7 selection unchanged.

At the same 0.70 cutoff, orientation averaging raises IoU to 29.51% and lowers clear-pixel false alarms to 0.07327%, but mean positive-site recall falls to 29.22%. CPU model-only time for the same 198 observations rises from 1.74 seconds to 13.19 seconds, about 7.6 times. Times exclude source reads, export and verification.

## Actual visual findings

All 46 plume cases were inspected in paired contact sheets. The largest clear-scene false alarms and representative clear observations were also inspected. Live browser checks covered a US improvement, a Kazakhstan miss, a China false alarm, and mask, score and error modes.

- The large US scene dated 2024-10-23 improves from 45.89% to 78.99% overlap, with 241 extra pixels. This strong example is not the overall result. The US totals cover only three positive observations. Their pooled recall improves, but the third observation loses some recall.
- Some Jordan, Libya and Venezuela shapes improve. Several entire plumes remain missed in Kazakhstan, Libya, Syria and Turkmenistan. The candidate regresses on some Iran and Russia observations.
- New clear blobs appear over China and Libya, and existing Syria false detections expand. Terrain and surface changes remain visible failure patterns. They are present in saved predictions, not missing overlay controls.
- The reference audit did not find a systematic date or label swap. Haze and brightness differences remain in individual pairs. Four development-positive countries are absent from training; this coverage gap does not explain every miss.

These are binary MARS-S2L Sentinel-2 L1C plume annotations. They are not quantitative NASA EMIT reconstruction. Repeating a 20 m prediction on the native 10 m annotation grid does not establish learned 10 m detail. No-plume annotations do not prove zero emission. A CAFO coordinate is not a dated plume label.

## Open the predictions

- [Site-balanced side-by-side comparison](http://127.0.0.1:8766/bundle-assets/GTM_Model7_site_v8_20261024_candidate/_versions/8dae8d3317d54aaebdc2d6e38033f844/index.html)
- [Site-balanced candidate on the interactive map](http://127.0.0.1:8766/?catalog=all&experiment=GTM_Model7_site_v8_20261024_candidate&scene=MARS_9123854a-64f6-4791-9255-63ad109e0896#compare)
- [Orientation diagnostic comparison](http://127.0.0.1:8766/bundle-assets/GTM_Model7_orientation_v8_20261024_candidate/_versions/3a418e898d074b569dfdfde1187df971/index.html)

Use Scene to select a case. Use Display to select the mask, score or errors. Teal shows correct overlap. Purple shows missed plume pixels. Pink shows extra predictions. The links above each image row open that exact case in the map viewer. If a just-published model is missing from the map catalog, use Refresh now before selecting it. A catalog refresh was needed in this check; no viewer-source fix is claimed.

## Verification and evidence

Failure cases and acceptance rules were recorded before implementation in the orientation and site-balanced v8 protocols. Real-batch preflight passed. Fresh CPU processes replayed every saved checkpoint across all 198 development observations exactly. The selected candidate's frozen inference was independently replayed with zero probability error. Each published source was reconstructed from original files. Native counts and threshold masks were independently checked. HTTP verification passed 2,774 served assets per paired comparison, 5,548 total.

Artifacts are under `outputs/GTM_Model7_orientation_v8_20261024` and `outputs/GTM_Model7_site_balanced_v8_20261024`. Each contains a recipe, measurements, replay receipts, source/native-count verification, HTTP checks and actual contact sheets. Browser screenshots include improvements and failures. The site-balanced checkpoint SHA256 is `d592e371babcb5e788c69caa4b9c9853d4c04c2338155a203bf75640ad117e21`.

To repeat the selected inference check from the repository, without training or selection:

```powershell
.venv/Scripts/python.exe tools/GTM_Model7_broader_v7_inference.py --freeze outputs/GTM_Model7_site_balanced_v8_20261024/selection/freeze.json --evaluation-manifest outputs/GTM_Model7_broader_dev_20261024/dev_data/combined_manifest.json --output outputs/GTM_Model7_site_balanced_v8_20261024/inference --verify
.venv/Scripts/python.exe tools/GTM_Model7_prediction_publication_e2e.py --review outputs/GTM_Model7_site_balanced_v8_20261024/review --output outputs/GTM_Model7_site_balanced_v8_20261024/http_recheck.json
```

## Next research boundary

No final-test arrays were opened in this pass. The earlier 17-observation clear-site test is already consumed and stays out of further tuning. There are no certifiably unused positive paired MARS sites left in the audited cache. Before claiming new-site plume performance, reserve and review genuinely new positive and clear locations, including US conditions. Future experiments can use exposed development failures for controlled improvement, with a new written protocol, but must retain that exposure label. No new imagery was downloaded, no external document was edited and no commit or push was made in this pass.
