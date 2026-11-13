# Hugging Face pilot acquisition plan

> Public source publication note: Referenced generated viewer pages, images, local outputs, and runtime receipts are retained as research pointers but are omitted from this Git publication. Those links require the original local artifacts.

Joshua asked for relevant Hugging Face data downloads and a plan before analysis, labeling, or consolidation. The following plan was communicated before opening the new samples.

1. Stage the complete CAFOSat California, Idaho, Louisiana and Washington archives (published total about 35 MB), unchanged, with pinned repository revision and checksum receipts.
2. Inspect published metadata for current EMIT/MARS and MethaneUnion offerings. Check existing acquisition reports before duplicating any data. Prefer small independently useful files; do not download whole repositories.
3. Perform only a read-only suitability review of staged samples: inspect their file schemas, image examples, dates, coordinates and label provenance. Keep facility labels distinct from methane/plume truth.
4. Report what is usable and propose the next experiment before relabeling, building combined training sets, or training. Keep the external shared research journal unchanged.

## Resource limits

The existing shared ledger is `outputs/cafo_pilot_2026-09-11/range_cache/budget.json`. It begins this pass at 42,467,328 charged bytes of 104,857,600. All new script-based metadata and dataset transfers use that ledger. Failed requests remain charged; the limit is not raised or reset. Browser/search page inspection is separate from dataset transfer accounting. Originals and staging receipts stay under ignored `outputs/`; no data is published.

## Failure conditions to address before acquisition code

- A file listing changes between inspection and download: resolve a commit first and fetch that exact revision.
- A large CSV/archive is selected accidentally: allow only the announced small archives and small documentation/metadata files, check published sizes, impose a per-file ceiling and the persistent total budget.
- A request fails or the process stops: reserve bytes before transfer, keep failed charges, never automatically retry large files, and never treat partial downloads as complete.
- Existing data would be overwritten: verify its size/hash for reuse or refuse it; write new transfers to an exclusive partial file then rename only after validation.
- The server returns a wrong size, truncated body or wrong content: enforce exact response length and expected LFS SHA-256 or Git blob ID. No full-body unbounded reads.
- An archive contains unsafe paths, links, extreme expansion or many members: inspect only bounded regular members; do not use blind extraction or execute included code.
- Facility labels are mistaken for methane labels, or a tiny geographic sample is mistaken for a balanced benchmark: retain original labels and explicitly restrict conclusions.
- Prior test data reappears in a new release: retain source identities and overlap warnings, and do not create new train/test splits in this pass.

This plan does not promise that the small archives contain useful methane supervision. Suitability is the question the pilot is intended to answer.
