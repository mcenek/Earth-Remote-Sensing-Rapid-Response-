# GTM Model6 review packet, 22 September 2026

The [follow-up report](../../GTM_Model6_followup_verification_2026-09-22.md) explains the methods, scored support, visual failures and data gates. This directory holds a small, reviewable copy of the results. Full checkpoints, 65-case viewer bundles and raw Sentinel/EMIT imagery stay in ignored local `outputs/` and acquisition directories.

- [Prediction review](r5_visual_review.png): all three held plume cases and two highest false-positive controls from the reused development fold.
- [R5 results](r5_report.md), [machine-readable metrics](r5_results.json) and [predeclared protocol](r5_protocol.json): four independent S16/S32/S64/S128 branches and their mean. The result failed the quality gate.
- [Registered observed Sentinel/EMIT data](georgia_registered_observations.png), [QA](georgia_qa.md) and [acquisition receipt](georgia_acquisition.json): one new near-time crop. It has no model prediction or approved quantitative training target.

The interactive [local viewer](../../../../GTM_Viewer/GTM_Model0_bundle_guide.md) requires its local run bundles, which are excluded from Git. The committed figures provide visual evidence without publishing the imagery cache or model checkpoints.
