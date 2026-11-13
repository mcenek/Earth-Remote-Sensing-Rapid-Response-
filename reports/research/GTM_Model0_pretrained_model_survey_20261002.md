# GTM_Model0: pretrained satellite and methane model survey

Checked: October 2, 2026. Research only. No model weights or datasets were downloaded or executed.

### MAPL feasibility update, October 3

The author's [Kaggle model page](https://www.kaggle.com/models/vishalbatchu/emit-methane-plume-detection-and-quantification/) now confirms the TensorFlow 2 default variation, Version 4, with CC BY 4.0 terms. The file explorer lists 638.17 MB total, including a 396.97 MB `sample_input/sample.tfrecord`. These are listed file sizes; compressed transfer size has not been measured. This supersedes the unverified weight-license note in the table below.

The model expects 256 by 256 float32 tiles with 285 radiance bands and three observation-geometry inputs. A bounded inference check using the included sample is feasible in principle after TensorFlow setup; local runtime and peak memory remain unmeasured. The active Python environment has no TensorFlow and had 4.79 GiB free RAM when checked. We have not downloaded or run the model.

For a first visual comparison, check exact acquisition matches in Google's [published enhancement and probability collection](https://developers.google.com/earth-engine/datasets/catalog/projects_climate-and-sustainability_assets_ghg_emit_mapl_emit_enhancements_v1_0). This can avoid downloading raw radiance granules. The catalog exposes `ee_asset_id`, but exact matches to our scenes remain unconfirmed. The active cache contains derived EMIT products and L1B scene metadata, not the raw RAD/OBS NetCDF pair required by the official runner. Google outputs remain predictions; an EMIT-input versus Sentinel-input comparison does not isolate the effect of model architecture.

## Recommendation

There are useful pretrained models. We do not need to start every feature extractor from random weights. However, satellite pretraining teaches image structure; it does not automatically teach methane detection.

For the three-CNN proposal, compare a small model trained from scratch with **SSL4EO-S12 ResNet-18** as the first pretrained CNN. Test **DOFA** as a transformer that can accept varying band counts. Test **Prithvi-EO-2.0 tiny-TL** if its six-band HLS input can be supplied correctly. These are candidate initializations, not validated winners for this project.

For a separate measured-methane comparison, inspect **MARS-S2L/CH4Net** on compatible Sentinel temporal pairs and **MAPL-EMIT** on EMIT radiance. **STARCOP** is another useful methane baseline where its retrieval inputs are available. These systems have different inputs, so their results must be labeled and compared on common observations where possible.

## Satellite feature extractors

Availability means the authors or an official model provider expose weights or a documented download. It does not mean the checkpoint was loaded successfully here. Download sizes below come from the provider listings, not a local transfer.

| Model and primary source | Verified inputs and release | Use in this project |
|---|---|---|
| [SSL4EO-S12 ResNet-18](https://github.com/DLR-MF-DAS/SSL4EO-S12), [TorchGeo HF checkpoint](https://huggingface.co/torchgeo/resnet18_sentinel2_all_moco) | Small CNN. The all-band checkpoint was trained on 13-band Sentinel-2 L1C. RGB weights also exist. Authors list CC BY 4.0 for weights and Apache 2.0 for code. | First pretrained CNN comparison. Our five-band input is not an exact match. A band-mapped five-channel input layer must be an explicit adaptation, with the remaining weights transferred and the adapter tested. Verify product level and normalization. Do not silently pad missing physical bands. |
| [SatlasPretrain](https://github.com/allenai/satlaspretrain_models) | AI2 publishes ResNet and Swin checkpoints. Sentinel versions include RGB and nine-band multispectral models. Separate aerial models use high-resolution RGB. Weight license: ODC-BY. | Use an aerial checkpoint for the facility task and a Sentinel checkpoint for satellite features. Choose the correct sensor version. Multispectral weights require a documented band adapter or additional native bands. Preserve their spatial feature maps. |
| [DOFA](https://huggingface.co/earthflow/DOFA), [author code](https://github.com/zhu-xlab/DOFA), [files](https://huggingface.co/earthflow/DOFA/tree/main) | ViT with wavelength-conditioned input handling. Authors support varying band counts, including Sentinel-2 and NAIP examples. The listed base encoder file is 448 MB. Model card: CC BY 4.0. | Best input-flexibility lead for our five bands. Supply correct wavelengths, scale, and normalization. First try frozen spatial features with a small head, then limited fine-tuning. Variable-band support does not prove methane sensitivity. |
| [Prithvi-EO-2.0 tiny-TL](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL) | A published 5M-parameter transformer. Pretraining uses six HLS bands at 30 m: blue, green, red, narrow NIR, SWIR1, SWIR2. Time/location-aware variants and larger models are available. Apache 2.0. | Attractive small transformer comparison. Our five-band stack lacks narrow NIR, and native Sentinel processing differs from HLS. Supply a verified HLS contract or explicitly test adaptation. Do not manufacture B8A from another band. |
| [Clay v1.5](https://clay-foundation.github.io/model/release-notes/specification.html) | Wavelength-based dynamic input embedding with GSD, date, and location metadata. Encoder: 311M parameters, about 1.25 GB. Apache 2.0 code and weights. | Flexible but larger than the first experiment needs. Keep it as a later frozen-feature comparison if smaller models show a useful signal. Its pretraining is not a methane detector. |
| [AnySat](https://github.com/gastruc/AnySat) | Official pretrained base model. Its Sentinel interface uses ten optical bands, dates, and a time series. | Relevant when temporal imagery is available. It is not a direct replacement for a single five-band crop. Confirm the exact checkpoint license before redistribution. |
| [SatMAE](https://github.com/sustainlab-group/SatMAE) | Official pretrained multispectral ViT weights and training code. The multispectral workflow uses ten Sentinel bands after removing B1, B9, and B10. | A later comparison with a larger input requirement. Verify the chosen weight license and band normalization. |
| [CROMA](https://github.com/antofuller/CROMA) | Official pretrained base/large models. The optical pathway uses 12 Sentinel-2 bands. Optical features can be extracted without a Sentinel-1 image. | Useful optical/SAR representation research. Our five-band data require adaptation. Verify exact weight terms when selecting the file. |
| [TerraMind 1.0](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base), [published base weight file](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base/blob/main/TerraMind_v1_base.pt) | IBM/ESA multimodal model. The current public file page lists the base checkpoint at 1.52 GB and Apache 2.0. | Available for later comparison. An older README mentioned private access; the current public file listing resolves the availability concern. Loading, modality adapters, and actual runtime remain untested. |

Do not compare vendor benchmark percentages as if they were methane results. These papers use different tasks, labels, regions, and splits. Run a small common held-site comparison before selecting an encoder.

## Methane-specific models

| Model and primary source | Release and required input | Appropriate role and limit |
|---|---|---|
| [MAPL-EMIT release](https://research.google/blog/mapping-global-methane-emissions-from-space-with-deep-learning/), [inference code](https://github.com/google-research/mapl), [Kaggle weights](https://www.kaggle.com/models/vishalbatchu/emit-methane-plume-detection-and-quantification/) | Google's September 1, 2026 release links a trained model and public outputs. The ViT system takes EMIT L1B RAD and OBS granules. Code produces enhancement, plume geometry, and source estimates. Code is Apache 2.0; weight license was not independently confirmed from the Kaggle page. | Strong lead for an EMIT reference comparison. It cannot take our five-band Sentinel chip. Start with available public outputs or one compatible cached observation before requesting large radiance files. Outputs are predictions, not independent truth. Source estimates need their own validation. |
| [MARS-S2L and CH4Net](https://huggingface.co/datasets/UNEP-IMEO/MARS-S2L), [official inference workflow](https://unep-imeo-mars.github.io/marss2l/notebooks/examples/download_and_inference.html) | Trained models are included in the official release. The Sentinel workflow uses current/reference imagery, B2/B3/B4/B8/B11/B12, valid-pixel masks, and wind. The workflow also supports physical retrieval and quantification. Models/data use CC BY-NC-SA 4.0. | Most direct pretrained methane baseline for Sentinel, subject to input compatibility. Existing target/reference caches may help. Audit overlap with the model's released training/evaluation data before an independence claim. B8, timing, and reference processing cannot be silently omitted. |
| [STARCOP](https://github.com/spaceml-org/STARCOP), [HF weights](https://huggingface.co/isp-uv-es/starcop) | Published `hyperstarcop_mag1c_only` and `hyperstarcop_mag1c_rgb` models use methane retrieval features, optionally RGB. Research covers airborne hyperspectral data and EMIT transfer. HF weights: CC BY-NC 4.0; repository: GPL 3.0. | Good comparison if the cached retrieval product matches the checkpoint's units and normalization. A TIFF named `mag1c` is not sufficient evidence of compatibility. RGB alone or a facility footprint does not supply the missing methane feature. |
| [MethaneMapper](https://github.com/UCSB-VRL/MethaneMapper-Spectral-Absorption-aware-Hyperspectral-Transformer-for-Methane-Detection) | Authors publish a hyperspectral transformer pipeline and describe pretrained detection/mask weights. Airborne hyperspectral input is required. The repository has its own UCSB license terms. | An airborne plume baseline if the intended data are compatible imaging spectroscopy. It cannot consume sparse aircraft point samples or ordinary aerial RGB. Confirm the weight path and terms before execution. |
| [SpaceML plume-hunter](https://github.com/spaceml-org/plume-hunter), [UnorthoDOS](https://huggingface.co/datasets/SpaceML/UnorthoDOS) | Authors link EMIT hyperspectral segmentation checkpoints and orthorectified/unorthorectified data. Code is MIT. The public dataset omits plume-free examples available by request. | A later EMIT segmentation comparison. The public positive-only subset cannot establish a representative false-positive rate. Do not download the complete archive to obtain a checkpoint. |

None of these models turns an aerial picture of a farm into a measured methane plume. A model prediction used to supplement labels must be marked as a pseudo-label. Keep an independent measurement-based evaluation set outside that process.

## Embedding products and facility data

- **[TESSERA](https://github.com/ucam-eo/tessera)** has both precomputed embeddings and downloadable inference weights. It is not limited to a hosted feature lookup. Its temporal Sentinel processing is more involved than a single-image backbone. Software is MIT; embeddings/weights are CC0. Choose a product version and time interval explicitly. Its source-specific preprocessing is important, including a documented correction for double-applied Sentinel offsets. This is a useful later facility-context comparison, not dated methane truth.
- **[AlphaEarth Satellite Embedding](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)** supplies annual 64-dimensional embedding data under CC BY 4.0. The reviewed release is an embedding dataset, not a local model checkpoint. Annual information may help characterize sites but cannot establish a plume on a particular overpass. Avoid future-year context when evaluating earlier predictions.
- **[CAFOSat](https://huggingface.co/datasets/oishee3003/CAFOSat)** supplies high-resolution facility imagery and infrastructure labels under CC BY 4.0. The [author repository](https://github.com/oishee-hoque/CAFOSat) provides classification code. We did not verify a ready pretrained CAFO checkpoint. Use it for facility review or training after provenance checks, not methane supervision.
- **[MethaneSET](https://huggingface.co/datasets/tacofoundation/methaneset)** is a data collection, not a backbone. It includes methane labels and multiple sensors, but does not solve our pairing, overlap, and negative-control requirements automatically. Existing local audits must be checked before acquisition.

## First comparison order

1. Establish one data contract and independent evaluation groups.
2. Reproduce a simple physical or pretrained methane baseline where its input contract is already met.
3. Train the small CNN 2 model from scratch.
4. Compare the small satellite-pretrained CNN under the same data and time budget.
5. Compare one transformer, preferably DOFA for band flexibility or Prithvi tiny for low compute with compatible six-band input.
6. Add the airborne branch and learned fusion only after the base task works and airborne measurements are identified.

For a five-band CNN adaptation, record the original channel order, selected weight slices, any new layers, normalization, and product-level differences. Fine-tune and compare with random initialization. Do not claim the original all-band checkpoint's performance for the adapted model.

Use a fixed small development budget first. Compare quality, peak memory, latency, and transfer size. Retain actual arrays and complete viewer cases. General model size or a visually attractive selected example is not sufficient evidence to choose a winner.

The current machine has not executed these candidates. Available GPU memory and a one-batch measurement will determine the feasible batch size. Check exact file sizes, hashes, licenses, and loaders before future downloads. No large model or imagery download is authorized merely by this shortlist.
