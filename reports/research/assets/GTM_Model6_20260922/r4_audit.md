# Mapper saved-prediction audit

Run: `GTM_Model6_mapper_mars_pilot_20260922_r4`. Checkpoint and all 65 saved predictions are unchanged.

- Frozen native metrics recompute: True.
- Old viewer support mismatches: 65 scenes.
- Positive/negative scenes: 3/62.
- Pooled native IoU: 0.0000%.

The original viewer metadata scored the full clear scene, while inference covered only the central 88x88 region of a 200x200 source. A 128-pixel context and 16-pixel output tile leave a 56-pixel margin. The old 64/128-pixel margin descriptions were wrong. The corrected export uses the exact saved support for both metrics and display. Unsupported area is unknown, not a true negative.

This correction does not turn the all-negative output into a successful model. The run contains 3 positive and 62 negative scenes; its original mean-per-scene aggregate was not a pooled IoU. A new training-only diagnosis is required before attributing this failure to data scarcity or architecture.

This is reused development evidence. It provides no EMIT enhancement, methane super-resolution, source localization or nationwide validation.
