# Friday review package cleanup acceptance, 2026-10-02

Before changing the tools, these are the failure cases the package must reject or avoid:

- A required review asset is missing, a directory, or a symbolic link (including a linked parent directory).
- A data-declared asset path is absolute, contains `..`, uses a backslash or drive prefix, or collides with `SHA256SUMS.json` or another asset name.
- An unrelated or stale file under the review directory enters the ZIP or hash manifest.
- A previous extraction leaves extra files that appear to belong to a new verification run.
- The ZIP has an unexpected entry, a corrupt member, or an extracted file whose hash differs from the source.

Acceptance: an isolated invocation with `--review-dir` and `--output-dir` packages exactly the declared viewer, evidence, diagram, PDF, and panel assets plus `SHA256SUMS.json`; verifies the exact extracted entry set and hashes in a fresh output subdirectory; and writes a receipt without changing old generated outputs. The exporter retains the same data and metrics and no longer tries to unlink an obsolete assets-level verification file.
