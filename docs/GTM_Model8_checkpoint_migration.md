# Model8 checkpoint migration

Use `tools/GTM_Model8_checkpoint_migration.py` only after the Windows trainer has stopped at a completed epoch and both source checkpoint hashes have been recorded independently. Do not copy an in-progress `last_epoch.pt`. Transfer the original files without changing them. The bridge reads trusted PyTorch checkpoints with `weights_only=False`, so hashes are mandatory before deserialization.

On Linux, prepare the frozen code, source, data, plan, and runtime. Run the existing trainer in `--mode preflight` with batch 32, bf16, two workers, 65,536 draws, validation batch 8, epoch cap 170, and patience 30. Confirm `fit/preflight.json` has `status: passed`. The destination fit directory may contain preflight artifacts but must have no checkpoint, epoch report, run status, or claim. Do not invoke `GTM_Model8_linux_job.py run` for this migrated fit: its fresh-run controller refuses an occupied fit directory.

Create `target_settings.json` containing exactly:

```json
{"batch_size":32,"precision":"bf16","num_workers":2,"draws":65536,"epochs":170,"patience":30,"validation_batch_size":8}
```

Use the transferred Windows `controller_state.json` as `--source-settings`. Run from the frozen repository with the destination Python environment:

```sh
python tools/GTM_Model8_checkpoint_migration.py \
  --source-full /transfer/last_epoch.pt \
  --source-full-sha256 FULL_SHA256_FROM_WINDOWS \
  --source-best /transfer/best_epoch.pt \
  --source-best-sha256 BEST_SHA256_FROM_WINDOWS \
  --source-preflight /transfer/preflight.json \
  --source-settings /transfer/controller_state.json \
  --source-plan /transfer/plan.json \
  --target-preflight /linux/outputs/GTM_Model8_mars_baseline_20261030/fit/preflight.json \
  --target-settings /linux/target_settings.json \
  --target-plan /linux/code/tool-diagnostics/mars-baseline-20261030/plan.json \
  --plan-sha256 3b29c62bb250dd9155b1c2af90637b605f2709f7ec6c64b18662a0c9ed2a6345 \
  --target-fit /linux/outputs/GTM_Model8_mars_baseline_20261030/fit
```

The command writes derived `last_epoch.pt` and `best_epoch.pt` plus `migration_receipt.json`. It verifies checkpoint schema, finite floats, saved epoch and validation-best selection, and every training-state field after serialization. Only the fingerprint's runtime lock, PyTorch version, and NumPy version may differ. The receipt records both original and derived hashes and each permitted difference. Any scientific, trainer, upstream, data, or unknown fingerprint difference fails. Original checkpoint bytes are checked again before success.

Then run the frozen trainer directly in an owned tmux session with the same preflight command arguments, changing only `--mode preflight` to `--mode train`. It will load the derived full state and resume at `epoch + 1`. The existing controller's final review path is not used; after successful training completion, run the frozen reviewer separately on the final `best_epoch.pt` and inspect its publication receipts. Cross-platform continuation preserves the saved state, but numerical updates after resume are not promised to be bitwise identical across CUDA, Python, or hardware. A forward parity check on a fixed input before the first resumed update is useful when the server environment is ready.
