# Model6 branch and combiner rehearsal: acceptance before implementation

Recorded October 1, 2026, before the new CLI or its E2E runner was written.

## Purpose and existing integration

Martin's plan is four independently fitted 16/32/64/128 context models, followed by a separately fitted combiner. The current mapper uses independent branch losses and an arithmetic logit mean. Its R5 checkpoint and development predictions are not inner out-of-fold training data for a new combiner.

The smallest missing piece is a schedule and artifact contract. Consume the existing mapper manifest's `training_observations` and `observations`, preserving `id`, `site`, `group`, `partition`, `target_scene`, `reference_scene`, `crs`, `affine`, and `shape`. Do not load the raster inventory or import Torch. Read branch sizes and the 16-pixel prediction window from the existing Model6 config.

For one outer evaluation fold, schedule four branches for each remaining inner held fold. Each inner fit excludes both the outer fold and the inner prediction fold. The combiner fits only those inner out-of-fold scores. Four final branches fit the outer training pool and predict the outer evaluation fold. The combiner's evaluation labels are not read. All parent observations, sites and shared acquisitions must remain within their recorded connected group and partition.

Only one canonical central 16x16 tile per parent is rehearsed. This is not a full-scene exporter or a replacement model trainer. Synthetic branch files are declared score stubs, not trained U-Nets. A tiny fixed logistic combiner exercises data flow only; its form is not the selected scientific merger architecture.

## Failure cases to establish first

1. A prediction's own parent, sibling site/group, or outer evaluation group appears in its branch training membership.
2. A parent, site, shared acquisition, or duplicate input hash is assigned to different connected groups; a group crosses partitions.
3. Final outer-fit predictions are reused as inner out-of-fold features, or a branch uses the wrong inner fold.
4. One branch is swapped, duplicated or omitted; its recorded context size, independent-loss mode, or checkpoint does not agree with the schedule.
5. CRS, affine transform, parent grid dimensions or central output region differ between the manifest and branch predictions.
6. Missing predictions/files silently become zeros, or scores have a wrong shape, nonfinite value, wrong score type, or inconsistent support.
7. Source manifest, plan, checkpoint, provenance, prediction or used training-target hashes are stale. Metadata must still be checked after a fault is deliberately rehashed.
8. An artifact path escapes its bundle, or a new report overwrites an existing file.
9. A real-data manifest is presented to the synthetic-only fit command. A contract pass is never a data-readiness or scientific-success pass.
10. Outer evaluation labels accidentally influence fitting or predictions. Removing them must leave the rehearsal byte-identical. Changing outer scores must leave the combiner weights unchanged, while changing fit scores must affect the fit.

## Public-command E2E acceptance

Write the E2E runner before implementation. It will create a deterministic fixture with ten artificial parents, five groups/partitions, and two parents per group, using the existing manifest keys. Expected schedule: 16 inner branch fits plus four final branch fits; 32 inner prediction tiles and eight outer prediction tiles. Preserve the fixture, exact invocations, rejection receipts and a final machine-readable result.

Call the public CLI to plan, check and rehearse. Run an identical replay and compare output bytes. Use a balanced, zero-score variant as an independent oracle: all fitted weights and bias remain zero and both combined and mean probabilities equal 0.5. Deliberately inject the failures above in isolated copies. Every rejected input must return a nonzero exit code with a specific reason. Missing evaluation targets must not be accessed. Existing output protection must preserve its original hash.

Finally run the plan command on the existing R5 manifest, without reading any imagery, checkpoint tensors or predictions, to demonstrate integration with actual project metadata. Label that schedule diagnostic and scientifically blocked. Keep all original R5 results and data-contract files unchanged.

Before adding status integration, require the existing public GTM_Model6.py status command to expose the preparation CLI while retaining learned_merger_trained=false, training_ready=false, and the failed R5 status. The R5 adapter check must preserve its recorded outer fold 1, with 99 fit parents and 65 evaluation parents. Extend fault execution to cover a missing checkpoint, stale provenance, final-fit score substitution, nonfinite supported scores and changed inner-fit scores.

## Resource and claim boundaries

Use Python standard-library JSON, hashing and small scalar calculations only. Bound JSON sizes, parent count and total prediction tiles; no worker pool, browser, server, GPU, model import, model download or paid service. The tiny command is a lightweight metadata job, not a replacement for the existing memory gates on model execution. No accuracy metric, real methane training, quantitative upscaling, source-origin claim, external journal edit, message, commit or push is part of this rehearsal.
