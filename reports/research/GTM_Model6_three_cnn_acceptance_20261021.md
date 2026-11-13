# Three-CNN local acceptance

> Public source publication note: Referenced generated viewer pages, images, local outputs, and runtime receipts are retained as research pointers but are omitted from this Git publication. Those links require the original local artifacts.

Date: October 21, 2026. **Software check passed. No measured methane result is established.**

## What is ready

CNN 1 and CNN 2 learn separate spatial features from Sentinel inputs. Their references are airborne and EMIT masks, respectively. CNN 3 learns from their frozen features using EMIT masks. Prediction needs Sentinel imagery only. The complete model still runs both frozen branches. The earlier four-window experiment remains historical evidence.

The public command covers data checks, bounded training, saved predictions, checkpoint replay, viewer export, and publication. Inputs have five bands in a fixed order. Unknown target cells remain unknown. Separate site groups are required for branch fitting, combination fitting, validation, and final test. Fitting uses eight right-angle and mirror transforms.

This version predicts plume probability on the input's 20 m grid. Loss and native scores use exact averages on the reference grid. It does not predict methane concentration, emission rate, or a confirmed source point. Independent finer methane measurements are still needed to validate fine detail.

## Verification

- The complete public workflow passed **79 checks** on a new 14-scene artificial dataset. It trained both branches and the combiner, then loaded all saved checkpoints again. All six paired test predictions replayed with maximum absolute difference **0.0**. Both branch weight hashes remained unchanged during combination training. [Receipt](assets/GTM_Model6_three_cnn_20261021/training_e2e.json).
- Deployment ran with only weights and training receipts while the original training manifest and arrays were unavailable. Its four scenes contained no target arrays. Both shared EMIT scenes matched the paired predictions within the declared tolerance. Unlabeled scenes have no reported methane accuracy.
- Five public inference checks rejected a target field, a hidden target array, a changed file hash, falsely declared measured provenance, and mixed measured coordinate systems. [Receipt](assets/GTM_Model6_three_cnn_20261021/input_only_rejections.json).
- The existing live viewer E2E passed **68 checks**, including publication updates and incomplete-publication rejection. The default presentation did not change. [Receipt](assets/GTM_Model6_three_cnn_20261021/live_viewer_e2e.json).
- The browser review covered the mask overlay, reference, original input, CNN 2/CNN 3 switching, heatmap, side-by-side view, negative control, input-only result, reload, and existing presentation. It caught and fixed a display selector that did not update the rendered layer. [Visual receipt](assets/GTM_Model6_three_cnn_20261021/browser_review.json).

The synthetic combiner did not outperform the synthetic EMIT branch. This check establishes working software, not a benefit from airborne data. Do not use its scores as methane research results or select a model from them.

## Visible evidence

The orange prediction follows a curved artificial signal. Cyan shows the coarser reference boundary. This is not measured satellite data.

![Synthetic CNN 3 prediction and reference](assets/GTM_Model6_three_cnn_20261021/fusion_mask.png)

[CNN 2 probability heatmap](assets/GTM_Model6_three_cnn_20261021/emit_heatmap.png) | [Negative control](assets/GTM_Model6_three_cnn_20261021/negative.png) | [Input-only prediction](assets/GTM_Model6_three_cnn_20261021/unlabeled.png)

## Repeat and use on the VM

Use the [training and viewer runbook](../../docs/GTM_Model6_three_cnn_runbook.md). A fresh engineering run is:

```bash
python tools/GTM_Model6_three_cnn_e2e.py \
  --output outputs/GTM_Model6/GTM_Model6_three_cnn_e2e_repeat --epochs 6
```

That command now also includes the five input-only refusal checks. The verified run and exact logs are retained locally under `outputs/GTM_Model6/GTM_Model6_three_cnn_e2e_20261021_v2/`. Large arrays and checkpoints remain outside Git. The small receipts and screenshots above can travel with the code.

Local verification used the existing WSL Python 3.11.14 and TensorFlow 2.21.0 environment on CPU. No imagery, model weights, or packages were downloaded. The shared imagery budget is unchanged. The GUI work was done and checked by the primary agent.

## Before real training

1. Restore working SSH and verify TensorFlow GPU access on the VM.
2. Identify the intended airborne product and its observation meaning.
3. Approve time-matched Sentinel and measured methane pairs. Resolve radiometry, support, independent controls, and group exclusions. Do not change approval fields to bypass these checks.
4. Train and visually inspect CNN 2 first. Then train CNN 1 and CNN 3 on the frozen manifest. Compare on the same held-out EMIT scenes. Use later ablations to distinguish extra fitting data and model capacity from an airborne-feature benefit.

The VM environment, measured-data training, and measured-map reprojection on a real approved cohort remain unverified. A larger machine does not resolve those data requirements. CAFOs remain a separate source-association check, not methane mask labels.
