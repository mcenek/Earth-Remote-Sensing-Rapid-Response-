# Experiment viewer update: acceptance before implementation

Visual direction, updated after Joshua's concept handoff: match the charcoal US map and connected comparison concepts under `design/map-concepts-2026-10-02`. Compact navigation, teal selection, legible off-white labels and a dominant imagery workspace. Orange predictions and cyan reference boundaries retain their scientific meaning. Root personally implements and verifies the UI.

Content order: experiment and scene selection, aligned maps and native scores, then provenance and diagnostics. Connection state and optional producer-reported progress remain visible without crowding the images.

Interaction: restrained selection/focus transitions, deliberate map fitting, and background refresh that preserves the current investigation. Respect reduced motion. New data must not cause an unexpected scene switch.

## Failure cases to address

- A fresh checkout shows blank maps, zero-height image containers or unavailable local-only assets.
- A selected scene's title changes while an older scene's imagery or scores remain visible.
- Broken image/grid requests look like valid empty predictions or successful zero-valued results.
- One invalid experiment prevents every other experiment from loading.
- Polling transfers the entire catalog unchanged, overlaps requests or continues without backoff during an outage.
- A producer writes a partial manifest or publishes before all assets exist, causing a previously visible result to disappear.
- An update silently changes the selected run, scene, map position, threshold, layer choice or search filter.
- Same-name revised assets stay in the browser cache after a new publication.
- A remote backend requires edits to workstation paths or exposes a public service by default.
- Historical saved results are mislabeled as a currently running or newly successful experiment.
- Duplicate portable/full outputs create duplicate observation markers, or pixel-only images are placed on invented geographic footprints.
- Map filters, dates, clusters, model switching or same-scene viewer links lose the selected observation or camera.
- A missing online basemap hides local evidence; illustrative concept hatching is mistaken for actual prediction support.
- Back/reload navigation, keyboard controls or narrow layouts lose access to scenes or evidence.

## End-to-end acceptance

Run the server and publisher through their public command-line interfaces with a temporary bundle root. Replay saved, explicitly labeled verification inputs through queued, partial and completed publications. Confirm that existing evidence survives an interrupted update and that a connected browser discovers valid new scenes and updated assets without a page reload. Confirm selection and inspection controls are retained. Stop and restart the temporary backend to check connection recovery.

Inspect the actual primary viewer in desktop and narrow viewports: map, comparison layouts, experiment list, reference-only or missing-prediction states, and original-team archive. Verify native scores remain identical to the saved artifacts and distinct from exploratory display scores. Save HTTP/CLI receipts and browser screenshots as repeatable evidence. No training, new satellite acquisition, or changed scientific outcomes are part of this work.
