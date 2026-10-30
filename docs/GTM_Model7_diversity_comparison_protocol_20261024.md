# Fixed comparison for the cached diversity supplement

Written before fitting this supplement. Do not change this protocol after inspecting its predictions. This pass does not start training.

Failures to prevent: test-guided selection; treating international fitting data as independent US validation; adding sensor or radiometry changes while claiming a dataset-only improvement; hiding empty masks or false alarms; selecting visually attractive training examples as proof of transfer; interpreting a producer cloud mask as proof of perfect atmospheric conditions.

Use `diverse_training/combined_manifest.json` in the October 24 non-Texas readiness output. Its SHA-256 is `b7d44b43e113baa98d61353750a8721d8297ab6a9d94fd6e74ca0ae50da75836`. Run the existing `GTM_Model7_train_repaired_mars_log_v3.py` trainer without changing its model definitions, optimizer, log1p radiometry, seed, site/class sampling or 2,400-update budget. Train both models. Keep original baseline outputs unchanged. Record all source hashes, software versions, draw schedule and train-only normalizer.

The four-real-crop learning controls must qualify before the full fit. A failure stops the run, as the trainer already enforces. Reject nonfinite data, loss or gradients. Keep its CPU-time bounds and save intermediate results. Do not silently increase the budget on failure.

Use the existing strict DEV rule in `GTM_Model7_dev_calibration_protocol.md`, unchanged, for milestone and cutoff selection. Also report the raw 0.50-cutoff results for a comparable baseline. Publish all 86 DEV cases for each model with current image, reference, annotation, model score, prediction and errors. Reconstruct source inputs, independently recount native pixels and replay saved checkpoints in a fresh process before publication. Inspect misses, clear scenes and the separately recorded haze/brightness warning examples. Those visual warnings do not authorize relabeling or data removal based on model results.

The DEV selection criteria are an experimental gate, not deployment acceptance. Stop promotion if recall, precision or clear-scene rejection fails, or if actual outputs mostly follow terrain/fields rather than plume annotations. No statewide run is permitted by a good fitting loss alone. Do not adjust using the already-consumed Kansas/Mississippi evaluation scenes. No claim of fresh geographic generalization can be made without a new independent labeled collection.

Future viewer IDs are reserved as `GTM_Model7_diverse_20261024_dev_stable_residual` and `GTM_Model7_diverse_20261024_dev_siamese`. Do not create catalog entries until actual saved predictions and independent verification exist. The present input-review page contains only training inputs and source annotations.
