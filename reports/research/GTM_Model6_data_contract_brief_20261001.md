# GTM_Model6: data decisions for the October 2 meeting

> Public source publication note: Referenced generated viewer pages, images, local outputs, and runtime receipts are retained as research pointers but are omitted from this Git publication. Those links require the original local artifacts.

I have a complete five-band Sentinel target image for the Volusia example and a real EMIT plume annotation to inspect over it. I do not yet have a trustworthy new training set. This follow-up checked the cached evidence and the provider documentation. It did not establish a safe pixel correction or produce a new model result.

The [Friday visual demo](../../GTM_Viewer/friday_20261002/index.html) remains the best way to show what works and what fails. R5 still has 2.15% pooled plume overlap and 2.25% pixel precision across 65 reused development scenes. Its three positive scenes belong to one connected group. The attractive example does not overturn those results.

## Martin's idea and what I propose to keep

The goal stays the same: use Sentinel to find possible methane plumes across the United States, with suitable EMIT observations teaching and checking the model. Train the 16/32/64/128 context models separately, then train another model to combine their predictions. Split physical sites before making the eight rotated/mirrored views. Check actual predictions in the viewer before accepting a score. These choices follow the [meeting record and design](../../docs/GTM_Model6_multiscale_reconstruction_design.md).

The specific window interpretation, timing limits, loss functions, and rules below are our implementation assumptions, not provider guarantees. EMIT supplies training evidence; the eventual national scan must work without requiring an EMIT image at every location. A finer display grid is not evidence of recovered finer methane detail.

## What the evidence actually says

| Question | Verified evidence | Current decision |
|---|---|---|
| Is Volusia a useful paired example? | Sentinel is 12.74 minutes after EMIT. The earlier raster audit found 1,503 annotated cells on the native 60 m grid. RGB, B11, B12 and SCL are cached. | Ready for input/reference inspection. No temporal reference image or independent evaluation cohort is present. |
| Is Black Oak a reviewed negative? | The training CSV declares `isplume=False`; its mask is all zero and declares zero as nodata. The frozen one-hour Sentinel search returned no match. | Keep it as a provider-declared control candidate, not a paired, reviewed methane-free scene. |
| Are the retrievals quantitative methane targets? | `mf`, `wmf`, and `mag1c` are present as band descriptions, but their TIFF unit fields are empty. Both provider RGB files, including their fourth validity channel, contain only ones. | Preserve stored values. Those fields do not establish physical units, retrieval uncertainty, sensitivity, or usable methane QA. |
| Does the exact release document these meanings? | The pinned v2025 card only identifies an updated release; it provides no mask-background, units, or QA contract. The live card inspected October 1 is similarly sparse. | Ask for the exact release's definitions or processing code. Do not import another product's units or negative-label semantics. |

The local observations above come from [hash-verified cached evidence](../../output/GTM_Model6_data_contract_20261001/e2e/baseline.json), including the pinned training CSV and prior native raster audit. The provider release is [MARS-Hyperspectral-EMIT-v2025](https://huggingface.co/datasets/UNEP-IMEO/MARS-Hyperspectral-EMIT-v2025), revision `4e6c50999eb430fa33abc9488b71cd1a0dcd7991`.

The [older MARS-Hyperspectral card](https://huggingface.co/datasets/UNEP-IMEO/MARS-Hyperspectral) describes a main-event binary mask and warns that the source's `isplume` flag is not a substitute for inspecting the pixel label. That is useful context, but it does not settle whether all zero pixels in this newer release were reviewed or whether other plumes can be unlabelled. Its matched-filter names also do not establish this release's stored-value conversion.

## Sentinel correction: still unresolved for these exact files

The cached Volusia item is `S2B_17RMN_20250601_0_L2A`, processing baseline 05.11, cataloging software `sentinel2-to-stac` 2025.03.06. It says `earthsearch:boa_offset_applied=true`, while all five spectral assets declare scale 0.0001 and offset -0.1.

Earth Search's [current instructions](https://github.com/Element84/earth-search/blob/main/README.md#gainoffset-in-items-after-jan-25-2022) say to apply the asset offset after scaling. However, an [Element 84 maintainer's earlier description](https://github.com/Element84/earth-search/issues/23) says their COG conversion already applied the offset and clamped very low valid values to 1. That strengthens the concern about correcting twice; it does not prove which treatment these particular 2025 files received. A report about a different scene cannot resolve this.

The two possible interpretations differ by 0.1 reflectance: `DN * 0.0001` versus `DN * 0.0001 - 0.1`. Neither is approved here. [Copernicus documents the original product's conversion](https://sentiwiki.copernicus.eu/web/s2-products), but that alone cannot tell us what a downstream COG conversion already changed. We need exact-asset processing evidence or a verified comparison with matching original source pixels. No correction was applied.

## Decisions and help to ask Martin for

1. **A provider contact or exact processing documentation.** For this v2025 release, what do mask zero, nodata, and `isplume` mean? Are labels complete for every plume? What are each retrieval's units, scaling, normalization, and QA fields? One documented positive/control example would be more useful than another large download.
2. **Independent, dated examples and reviewed controls.** We need usable Sentinel target/reference pairs and EMIT observations from different physical sites. Facility coordinates can guide selection, but cannot supply methane-free labels or exact emission origins.
3. **Agreement on the first success criterion.** Start with credible plume outlines on independent native EMIT support, report misses and false detections, and inspect the fixed cases in the viewer. Quantitative methane reconstruction needs verified units and QA. Finer-resolution accuracy and Gaussian source localization need additional independent measurements and source/wind evidence.

## Replay and limits of this pass

From the research checkout, use an existing Python 3.11+ installation:

```text
python tools/GTM_Model6_cached_readiness.py --output output/GTM_Model6_readiness_replay.json
python tools/GTM_Model6_cached_readiness_e2e.py --output-dir output/GTM_Model6_readiness_e2e_replay
```

Use new output names on each run; existing outputs are preserved. Exit 0 means the frozen evidence is intact, not that training is approved. The [E2E evidence](../../output/GTM_Model6_data_contract_20261001/e2e/e2e.json) records deterministic replay, rejection of a changed mask and missing band, and refusal to overwrite a prior report.

All 37 pinned files, including 16 rasters, the original receipts, budget ledger, R5 checkpoint and saved results, matched their hashes. Counts and geometry were read from the hash-linked prior audit; this pass did not decode rasters again. With under 1 GiB of free RAM observed, it used only small standard-library metadata/hash operations. No imagery transfer, training, inference, external journal write, or publication occurred. The imagery allowance remains 3,768,186 bytes out of the original 100 MiB cap. No scientific readiness gate has been cleared.
