# Recovery visual review and end-to-end checks

Written before the recovery review renderer.

Failures to prevent: showing a training crop as held-out success; stretching prediction scores separately per image; using a smoothed display to imply higher resolution; swapping old and revised outputs; comparing different scenes; silently hiding missed plumes or false alarms; changing a checkpoint during rendering; interpreting a no-plume label as proof that a facility never emits.

Use saved arrays and independently rebuild the native image and mask from original TIFFs. Pin checkpoint, source, manifest and prediction hashes. Display the same current image, reference, annotation and fixed 0.50 masks for all models. Show scores on a fixed zero-to-one scale and errors on the same valid support. Keep all selected confirmation scenes, including failures and exclusions. Training memorization examples must be labeled TRAIN ONLY. Choose training and development examples by stable scene ID, not prediction quality.

Before sealed confirmation rasters are opened, freeze the revised configuration, threshold and checkpoint using only training and development results. Run both original checkpoints and the chosen revision on the same confirmation scenes. Reproduce saved predictions in a separate process. Publish real grids and georeferenced overlays through the existing local viewer. Inspect the browser with a positive case, a negative case, and error and score displays. Save screenshots and a machine-readable verification receipt. Report any failed quality check plainly.
