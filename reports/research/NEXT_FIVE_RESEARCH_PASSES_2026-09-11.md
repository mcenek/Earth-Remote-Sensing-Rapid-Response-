# Next five research passes

Planning date: 2026-09-11. These are hypothetical work packages, not completed experiments or a promise that each fits into one session. Follow Martin's sequence: known-facility acquisition, five approximately 20%-training/80%-validation rotations, strongest defensible Sentinel-2 result, then TESSERA and fusion. Nationwide application remains the eventual objective.

## Starting position

The local inventory has 14,155 facility records, of which 12,703 are listed Active. Its association with Huang is unconfirmed. A draft groups the active records into 4,244 geographic components and five balanced folds; the largest component contains 1,770 facilities. Three bounded Sentinel-2 catalog queries returned lower-cloud candidate scenes, but those crops have not been acquired or validated. Temporal candidates, training utilities and evaluation utilities exist; their tests and synthetic rehearsals are not real-data performance results.

## Pass 1 — Establish the research question and trustworthy cohort

**Question:** What exactly will a positive prediction mean, and which source records can support it?

- Confirm the coordinate inventory's source/version, relationship to Huang, datum, location uncertainty, activity dates and geographic scope. Retain source IDs and raw provenance; inspect duplicate coordinates rather than deleting distinct facilities.
- Define facility retrieval and methane validation separately. Known facility presence supports facility-recovery measurement; independently verified, time-compatible plume evidence supports methane accuracy. If plume evidence is unavailable, report facility association only.
- Select a study year/window consistent with inventory provenance and the later annual TESSERA comparison. The current 2024 assumption is provisional.
- Inspect the largest geographic components and agree on crop footprints and geographic separation before freezing folds. All dates from a facility remain in its group. Preserve Martin's one-fold-train/four-fold-validation direction.
- Specify the controls: reviewed non-CAFO examples for facility classification; independently supported no-plume observations where methane false-positive rates are claimed. Keep other emitters and unlabeled locations distinct.
- Write the metric definitions, spatial matching rule, observation denominator, model-selection budget and final confirmation strategy before examining model results. No nationwide inference from an Iowa-only development cohort.

**Deliverable:** versioned cohort specification, label dictionary, source/coordinate audit, draft acquisition list and justified split design.

**Proceed when:** the task's labels and location assumptions are defensible. Otherwise resolve the source/label gap; do not manufacture methane truth from facility presence.

## Pass 2 — Acquire a small pilot and verify the complete data path

**Question:** Can we consistently obtain usable observations at the chosen facilities?

- Begin with roughly 25–50 locations spanning multiple geographic groups, facility types and coordinate-quality levels. This is a pipeline pilot, not a performance benchmark. Select locations and image dates by predeclared coverage/quality criteria, not model success.
- Start with one eligible observation per location and a compatible reference where temporal inference requires it. Add dates only after the first complete path works.
- Use bounded crop extraction and reuse shared source tiles. Record attempted/retained samples, transfer bytes, failures and expected expansion cost. Stop on persistent path/product errors rather than scanning unrelated archives.
- Verify facility visibility, crop-level cloud/SCL and nodata, band order, units/scale/offset, geotransforms, native SWIR resolution and temporal alignment. L2A assets require their own adapter; do not silently feed them to a frozen L1C model.
- Produce visual contact sheets and inspect representative successful and rejected crops. Attach facility-control review and any available plume evidence separately.

**Deliverable:** small inspected pilot dataset, reproducible acquisition/preprocessing commands, hashes, QA report and measured scaling estimate.

**Proceed when:** representative examples pass the data contract and the acquisition budget is practical. A small engineering pilot passing does not justify a model-accuracy claim.

## Pass 3 — Establish a real-data baseline

**Question:** What can a simple, properly evaluated method recover from these observations?

- Expand acquisition in bounded tranches only after the pilot succeeds, aiming for the verified cohort rather than assuming every inventory row is usable. Record missing coverage without moving facilities between folds to improve balance or results.
- For facility classification, compare a simple regularized baseline with a compact image CNN using reviewed facility/control labels. If sufficient date-specific plume labels exist, separately evaluate an input-compatible frozen methane reference and train the temporal CNN under the plume contract.
- Run one diagnostic rotation first; check labels, gradients, prediction maps, invalid-scene handling, runtime and memory. Then execute all five approximately 20/80 rotations after fixing engineering failures.
- Fit preprocessing, tuning and thresholds within each 20% training pool using internal geographic splits. Evaluate its outer 80% only after its model/rule is fixed.
- Report known-facility recovery among observable locations, abstentions, spatial matching errors and candidate workload. Report classification precision/AP only with appropriate controls, and plume AP/IoU only with appropriate plume truth.

**Deliverable:** first real baseline report, out-of-training predictions with facility/group IDs, qualitative errors and resource measurements.

**Proceed when:** the baseline is reproducible and the error analysis separates data failures, facility confusion, methane observability and model limitations. If labels or coverage are inadequate, return to Pass 1 or 2.

## Pass 4 — Find the best defensible Sentinel-2 configuration

**Question:** Which specific change improves the agreed task without simply adding false alerts or exploiting the location split?

- Freeze a short candidate list and equal tuning budgets. Prioritize a clean CNN baseline, temporal information where relevant, and the already prepared compact temporal-attention comparison where the available labels support it.
- Change one factor at a time: input/date selection, temporal comparison or attention. Keep preprocessing, site grouping, supervision and training budget matched within each comparison.
- Use inner training-group validation for tuning. Report all five outer rotations, uncertainty across geographic groups, sensitivity to seeds and the corresponding false-alert/review workload. Four validation predictions for one facility are not four independent facilities.
- If outer-rotation results select the global winner, label that selection as development evidence. Reserve a genuinely new geographic region for subsequent confirmation instead of rebranding the same rotations as an untouched final test.
- Freeze the selected Sentinel-2 model, preprocessing, operating rule and artifacts. Keep the simpler model if the more complex one does not deliver a meaningful gain under the prespecified criteria.

**Deliverable:** a bounded ablation report and frozen Sentinel-2 baseline, including negative results and limitations.

**Proceed when:** the baseline selection is justified by the agreed metrics. If no candidate helps, retain the simpler baseline and still conduct the planned representation comparison rather than continuing open-ended architecture search.

## Pass 5 — Repeat with TESSERA, test fusion and prepare regional screening

**Question:** Does TESSERA improve the same task and cohort beyond the frozen Sentinel-2 baseline?

- Verify regional/year availability and pin the TESSERA version. Use a separate compatible environment. Begin with a small embedding retrieval pilot before expanding.
- Compare three arms: frozen Sentinel-2 baseline; frozen TESSERA embeddings plus a small classifier; Sentinel-2 plus TESSERA feature fusion. Use identical facility/control labels, rotation assignments, temporal scope and tuning budgets.
- Perform the primary comparison on common coverage, and report each method's full coverage and excluded facilities separately. Fit normalization/dimensionality reduction only within the training pool.
- Treat annual embeddings as retrospective representations. They must not provide later-in-year information to a purported real-time, date-specific plume experiment. Any benefit in facility recognition is reported separately from verified methane sensitivity.
- Estimate regional inference throughput, storage and candidate-review burden. Run a bounded regional screening demonstration only after the method and inference contract are ready; audit a sample of detections and misses. Use a new region for confirmation if suitable labels exist.

**Deliverable:** matched Sentinel-2/TESSERA/fusion comparison, representative maps, coverage accounting and an evidence-based US-scale feasibility plan.

**Proceed to nationwide work when:** regional coverage, runtime, data availability and review requirements are understood. A nationwide map alone is not evidence of methane accuracy; known-CAFO recovery is one endpoint, with non-inventory detections remaining unclassified until investigated.

## Expected outcome of these five passes

A trustworthy acquisition path, a real baseline, a bounded architecture comparison, an honest answer about whether TESSERA adds value, and a justified next step toward broader screening. The outcome may be that a simpler model is best, or that labels remain the binding limitation. Neither should be concealed by additional architecture searches or synthetic demonstrations.

Companion documents: `MARTIN_CAFO_TESSERA_DIRECTION_2026-09-11.md`, `CAFO_COORDINATE_AUDIT_2026-09-11.md`, and `configs/cafo_tessera_research_plan.json`.
