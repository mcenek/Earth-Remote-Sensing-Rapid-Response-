# GTM_Model6 cached-pair readiness: failure cases and acceptance

Written before implementing the replay command. Scope: the already cached Volusia positive and Black Oak provider-declared control. This is a data-contract audit, not training, relabeling, raster conversion, or a new scientific result.

## Failure cases

1. A missing, replaced, truncated, or edited source file could silently change the conclusion. Pin existing metadata hashes, verify raster hashes against the pinned receipts, and stop on mismatch. Never repair or overwrite a source file.
2. Receipt paths could point outside the selected cache or follow a different machine's absolute path. Resolve only known relative paths within the explicit cache root.
3. The Sentinel item says its BOA offset was applied, while its five assets still declare offset -0.1. Do not pick either conversion by appearance, minimum pixel value, acquisition year, or a report about another scene. Report the contradiction and keep the conversion unapproved.
4. A mask value of zero could become a reviewed negative by accident. Preserve the observed mask codes and nodata=0. Neither zero pixels nor `isplume=False` establish valid negative supervision for this exact release.
5. Band descriptions such as `mf`, `wmf`, and `mag1c` could be mistaken for physical units. Report descriptions and units separately. Do not inherit units or QA definitions from NASA L2B products or an older MARS release.
6. Available target bands could be mistaken for a complete temporal pair or an independent cohort. Report the missing reference observation, missing Sentinel match for Black Oak, and development-only selection.
7. A readiness replay could accidentally import a training/acquisition module or consume the imagery allowance. Use only Python's standard library, stream hashes in small blocks, make no network requests, and verify the ledger is byte-identical before and after.
8. New work could erase old failed results or overwrite a previous report. Check the original R5 checkpoint/results hashes and use exclusive creation for each output file.

## Acceptance

- The public command reads only its pinned local inputs and produces a JSON report. Intact but scientifically blocked data is a successful audit, not permission to train.
- All ten native EMIT files and six Sentinel files match their saved hashes. The recorded raster metadata remains tied to those exact bytes. Pixel counts and geometry are explicitly attributed to the prior native audit; this low-memory pass does not decode rasters again.
- Report the observed Volusia count of 1,503 positive cells, Black Oak's zero positive cells, native 60 m annotation resolution, five available Volusia target bands, and Black Oak's absent Sentinel match. Preserve all unresolved semantic gates.
- The same inputs produce the same report bytes. Missing or changed inputs produce an integrity failure and no readiness approval. Existing output files are refused unchanged.
- A small E2E runner invokes the public command on the actual cache and on an isolated copy with a changed mask and a missing band. It saves command results and hashes. No unit tests or production-cache mutations.
- Keep the imagery ledger at 100 MiB with 3,768,186 bytes remaining. At most two CPU threads, no GPU, installs, inference, or training. Under the observed sub-1-GiB free RAM, use standard-library metadata/hash work only.

No deterministic radiometric correction is established at this point. Any future correction needs exact-asset provider provenance or an independently verified matching source-pixel comparison, plus a separate failure-first check against double correction.
