> **HISTORICAL PROPOSAL / REPORT.** Active work has restarted at [GTM_Model6](../../research/GTM_Model6/README.md). Claims and recommendations below belong to earlier research; they are not current results or execution instructions.

# ERSRR Research Report: Sentinel-2 to EMIT Methane Super-Resolution

**Date:** April 2026
**Project:** Earth Remote Sensing Rapid Response (ERSRR)
**Authors:** Research report by AI assistant for the ERSRR team

---

## 1. Problem Statement

The ERSRR project aims to predict methane concentration maps at the spatial resolution of Sentinel-2 (~20m) using the accurate but coarser EMIT methane products (~60m) as ground truth labels. This is fundamentally a **guided super-resolution / image-to-image translation** problem:

- **Input:** Sentinel-2 multispectral imagery (5 bands at 20m: B2, B3, B4, B11, B12)
- **Target:** EMIT L2B CH4 plume concentration maps (originally 60m, bilinearly resampled to 256x256 grids matching S2 extent)
- **Output:** Predicted methane concentration maps + binary plume segmentation masks at 20m resolution

The core challenge: EMIT provides reliable methane column enhancement data but at 60m GSD. Sentinel-2 has finer 20m spatial resolution in SWIR bands (B11 at 1610nm, B12 at 2190nm) but lacks the spectral resolution for precise CH4 retrieval. The model must learn to exploit subtle methane absorption signals in S2's broad SWIR bands to produce high-resolution methane maps that generalize beyond what S2 can directly measure.

---

## 2. Current Repository Analysis

### 2.1 Architecture Overview

The current codebase implements a **Vision Transformer (ViT) encoder-decoder** with a multi-task head:

```
Input (256x256x5) -> Patch Extraction (8x8) -> Patch Encoding ->
  8x Transformer Blocks (with skip connections at layers 1, 4, 7) ->
  Reshape to 2D -> 3x Decoder Blocks with skip connections ->
    Head 1: Methane regression (sigmoid)
    Head 2: Plume segmentation mask (sigmoid)
```

### 2.2 Data Pipeline

1. **Data Collection** (`s2_image_exporter.py`): Uses Google Earth Engine to fetch Sentinel-2 images paired with EMIT plume geometries. Generates overlapping 5.12km x 5.12km grid tiles.

2. **Preprocessing** (`process_all.py`): Reprojects EMIT data to match S2 geospatial coordinates using Gaussian resampling, combines into 6-band GeoTIFFs (5 S2 bands + 1 EMIT band).

3. **Training** (`ERSRR_Model.py`): Loads 6-band TIFFs, splits first 5 bands as input (X) and band 6 as target (Y). Applies log-transform + percentile normalization. Trains with AdamW optimizer.

4. **Evaluation** (`Plume_Classifier.py`): Binary thresholding of predictions, connected component analysis, peak detection, matching to CAFO database via geodesic distance.

### 2.3 Dataset Statistics

- **Train/test samples:** ~99 paired images
- **Validation samples:** 3 images
- **Image size:** 256 x 256 pixels
- **Input bands:** 5 (S2 B2, B3, B4, B11, B12)
- **Target:** 1 (EMIT CH4 enhancement)

### 2.4 Identified Weaknesses in Current Approach

| Area | Current State | Issue |
|------|---------------|-------|
| **Dataset size** | ~99 training samples | Far too small for deep learning; orders of magnitude below what published methods use |
| **Architecture** | Custom ViT encoder-decoder | Patch-based approach loses fine spatial detail; decoder uses simple transposed convolutions with resize-based skip connections |
| **Input bands** | 5 bands only | Missing B5, B8, B8A, B9 which are used by state-of-the-art methods (CH4Net uses all 13 bands) |
| **Temporal info** | Single time step | Published methods (Rouet-Leduc 2024, MEQNet 2025) use bi-temporal input to separate transient plume signals from static background |
| **Loss functions** | Weighted MSE + BCE/Dice | No perceptual/structural loss; no uncertainty weighting between tasks |
| **Normalization** | Per-image percentile normalization | Statistics not portable across images; denormalization assumptions fragile |
| **Data augmentation** | None | Critical missing piece for small dataset |
| **Validation** | 3 samples, random split | Statistically meaningless; needs spatial/geographic split |
| **Evaluation metrics** | MSE, MAE, BCE on pixels | Lacks standard SR metrics (PSNR, SSIM) and plume-level metrics (IoU, F1) |

---

## 3. State of the Art: Deep Learning for Methane Detection

### 3.1 Key Published Systems

#### CH4Net (Vaughan et al., 2024 - Atmospheric Measurement Techniques)
- **Architecture:** U-Net with 4 encoder/decoder blocks
- **Input:** All 13 Sentinel-2 bands interpolated to 10m (single time step)
- **Dataset:** 10,046 images across 23 super-emitter sites in Turkmenistan, 925 with plumes
- **Task:** Binary plume segmentation (not regression)
- **Key result:** 84% recall vs 24% for traditional MBMP baseline
- **Relevance:** Demonstrates U-Net > ViT for this task with limited data; shows all bands help

#### Rouet-Leduc & Hulbert (2024 - Nature Communications)
- **Architecture:** ViT-base encoder + U-Net decoder
- **Input:** 10 S2 bands at TWO time steps (20 channels total)
- **Dataset:** ~1.65 million 2.5km scenes with synthetic Gaussian plumes embedded via Beer-Lambert law
- **Task:** Plume segmentation (binary)
- **Key result:** Order-of-magnitude improvement; detects plumes down to 200-300 kg/h
- **Relevance:** Proves ViT works with massive synthetic data + bi-temporal input; validates Gaussian plume augmentation

#### S2MetNet (Radman et al., 2023 - Remote Sensing of Environment)
- **Architecture:** EfficientNet-V2L (fine-tuned)
- **Input:** Sentinel-2 imagery
- **Dataset:** Large-scale synthetic plumes via Large Eddy Simulation (LES)
- **Task:** Emission rate regression
- **Key result:** RMSE 2101 kg/h, Pearson R 95.7%
- **Relevance:** Shows CNN-based approaches can quantify emission rates without wind data

#### MEQNet (Xu et al., 2025 - NeurIPS Climate Change Workshop)
- **Architecture:** End-to-end network
- **Input:** Bi-temporal S2 imagery + 10m wind vectors
- **Task:** Joint methane column enhancement + emission rate estimation
- **Key result:** Direct mapping from reflectance to emissions across complex surfaces
- **Relevance:** Most similar to ERSRR's dual-head approach; demonstrates value of auxiliary wind data

#### Tran et al. (2025 - arXiv:2508.16282)
- **Architecture:** U-Net + ResNet34 encoder
- **Input:** 3-channel Varon ratio / Sanchez ratio feature stack
- **Task:** Small plume segmentation
- **Key result:** Detects plumes down to single pixel (400 m2); F1 78.39%
- **Relevance:** Feature engineering (spectral indices) + pretrained encoder significantly boosts small-plume detection

#### AttMetNet (Ahsan et al., 2025 - ICLR submission)
- **Architecture:** Attention-enhanced U-Net with NDMI spectral index
- **Key takeaway:** Reviewers noted combining existing components without novel paradigms; domain-specific feature engineering (NDMI from B11/B12) helps

### 3.2 Survey of ML for Methane (Tiemann et al., 2024 - arXiv:2408.15122)

Key findings from the comprehensive survey:
- CNN-based U-Net architectures dominate for segmentation
- Transformer-based methods gaining ground for global context
- Synthetic data generation (Gaussian plumes, LES) is standard practice
- Transfer learning from ImageNet weights is standard
- Sentinel-2 detection threshold: ~1000-5000 kg/h (multispectral), EMIT: ~200-300 kg/h (hyperspectral)
- Future missions (CHIME 2028-30) will provide <10nm spectral resolution at 20-30m GSD

---

## 4. State of the Art: Remote Sensing Super-Resolution

### 4.1 Relevant SR Architectures

| Architecture | Year | Type | Key Feature | PSNR on Urban100 (x4) |
|---|---|---|---|---|
| SwinIR | 2021 | Transformer | Shifted window self-attention | 27.45 dB |
| HAT | 2023 | Hybrid | Channel attention + window self-attention + overlapping cross-attention | 28.60 dB |
| ESRGAN | 2018 | GAN | Residual-in-residual dense blocks, perceptual loss | High perceptual quality |
| SR3 / Diffusion | 2022 | Diffusion | Iterative denoising from LR conditioning | Excellent realism, slow |
| CDEHAT | 2026 | Diffusion+Transformer | Conditional diffusion prior + hybrid attention SR | SOTA for RS SR |

### 4.2 CDEHAT (Wu et al., 2026 - ISPRS Journal)

The most directly relevant recent work for remote sensing SR:
- **Dual-branch architecture:** CDEHAT-Diff (prior generation) + CDEHAT-SR (reconstruction)
- **LEHAB blocks:** Local Enhancement via channel-spatial fusion attention
- **GEHAB blocks:** Global Enhancement via shifted overlapping window attention + global contextual transposition attention
- **Dataset:** Sentinel-2 to NAIP cross-sensor SR
- **Result:** +0.20 to +0.51 dB PSNR improvement over prior SOTA

### 4.3 Diffusion Models for RS SR (Key Insight)

A critical finding from Wang et al. (2026, Int. J. Applied Earth Observation):
- Cross-platform SR (satellite -> aerial) faces severe **domain gap** challenges
- Models trained from scratch outperform fine-tuned models when dataset is large enough
- Standard CV metrics (PSNR, SSIM) are inadequate; environmental metrics (NDVI preservation) are better
- SR3 diffusion model achieved 27.28 dB on single-source data but only 16.85 dB cross-platform

**Implication for ERSRR:** The Sentinel-2 -> EMIT mapping is inherently cross-domain (multispectral -> hyperspectral-derived). Simple PSNR optimization may not capture the right methane signal characteristics.

---

## 5. Spectral Band Analysis for Methane Detection

### 5.1 Methane Absorption Physics

CH4 has absorption features in the SWIR at:
- **~1.65 um (1650 nm):** Weak absorption band - covered by Sentinel-2 B11 (1610 nm)
- **~2.3 um (2300 nm):** Strong absorption band - covered by Sentinel-2 B12 (2190 nm)

### 5.2 Sentinel-2 Band Utility for CH4

| Band | Wavelength | Resolution | CH4 Sensitivity | Role |
|------|-----------|------------|-----------------|------|
| B1 | 443 nm | 60m | None | Aerosol correction |
| B2 | 490 nm | 10m | None | Blue - surface reference |
| B3 | 560 nm | 10m | None | Green - surface reference |
| B4 | 665 nm | 10m | None | Red - surface reference |
| B5 | 705 nm | 20m | None | Red edge - vegetation/surface |
| B6 | 740 nm | 20m | None | Red edge |
| B7 | 783 nm | 20m | None | Red edge |
| B8 | 833 nm | 10m | None | NIR - albedo baseline |
| B8A | 865 nm | 20m | None | NIR - albedo baseline |
| B9 | 945 nm | 60m | None | Water vapor |
| B10 | 1375 nm | 60m | None | Cirrus detection |
| **B11** | **1610 nm** | **20m** | **Weak** | **SWIR-1: CH4 weak absorption + reference** |
| **B12** | **2190 nm** | **20m** | **Strong** | **SWIR-2: Primary CH4 absorption band** |

### 5.3 Feature Engineering Approaches

The literature consistently shows that spectral indices improve detection:

**Varon Ratio (multi-band single-pass):**
```
V(B12, B11) = (c * B12 - B11) / B11
where c = least-squares scaling factor
```

**Sanchez Ratio (regression-based):**
```
S(B12) = V(B12, B12_predicted)
where B12_predicted = multi-linear regression on non-absorbing bands
```

**MBMP (Multi-Band Multi-Pass) - best traditional method:**
```
MBMP = (c*R12 - R11)/R11 - (c'*R'12 - R'11)/R'11
where R = current image, R' = reference image
```

**Recommendation:** Include computed spectral indices as additional input channels alongside raw bands.

---

## 6. EMIT Instrument Specifications

| Parameter | Value |
|-----------|-------|
| Full name | Earth Surface Mineral Dust Source Investigation |
| Platform | International Space Station (ISS) |
| Orbit | ~51.6 deg inclination, ~420 km altitude |
| Instrument | VSWIR Dyson imaging spectrometer |
| Spectral range | 380-2500 nm |
| Spectral sampling | ~7 nm (285 contiguous bands) |
| Spatial resolution | **~60 m GSD** |
| Swath width | ~75 km |
| Latitudinal coverage | 52 deg N to 52 deg S |
| CH4 product | EMITL2BCH4PLM (plume complexes) and EMITL2BCH4ENH (enhancement maps) |
| Units | ppm m (parts per million meter) |
| NoData | -9999 |
| Manual review | All plume complexes are manually verified before release |
| Data access | NASA LP DAAC, Google Earth Engine |

**EMIT CH4 detection threshold:** ~200-300 kg/h (good conditions)

---

## 7. Dataset Requirements and Recommendations

### 7.1 Current Dataset Issues

The current ~99 sample dataset is the single biggest bottleneck. Published methods use:

- CH4Net: 10,046 images
- Rouet-Leduc 2024: ~1,650,000 scenes
- S2MetNet: Large-scale synthetic + real backgrounds

### 7.2 Recommended Data Strategy

1. **Expand real data:** Use the `s2_image_exporter.py` pipeline to collect more paired S2+EMIT samples from diverse global locations. Target 1000+ samples minimum.

2. **Synthetic plume augmentation:** Generate synthetic CH4 plumes using:
   - Gaussian plume models (fast, proven by Rouet-Leduc 2024)
   - Large Eddy Simulation if available (more realistic, used by S2MetNet)
   - Embed synthetic plumes into real S2 backgrounds using Beer-Lambert law
   - This is the single highest-impact improvement available

3. **Bi-temporal data:** Fetch S2 images from two time steps (current + reference) to enable transient signal detection.

4. **Geographic diversity:** Current data appears concentrated in specific regions. Add samples from diverse surfaces (arid, vegetated, urban, water-adjacent).

5. **Proper data splits:** Use geographic splits (train on one continent, test on another) rather than random splits to measure generalization.

---

## 8. Evaluation Metrics

### 8.1 Standard Metrics (Pixel-Level)

| Metric | Use | Current |
|--------|-----|---------|
| MSE/MAE | Regression accuracy | Yes (masked) |
| PSNR | Peak signal-to-noise ratio | No - add |
| SSIM | Structural similarity | No - add |
| R-squared | Variance explained | No - add |

### 8.2 Plume-Level Metrics

| Metric | Use | Current |
|--------|-----|---------|
| IoU | Plume segmentation overlap | No - add |
| F1 Score | Plume detection balance | Partial (via CAFO matching) |
| Precision/Recall | Detection accuracy | Partial |
| Emission rate RMSE | kg/h accuracy | No - add |

### 8.3 Physical Metrics

| Metric | Use | Current |
|--------|-----|---------|
| Source localization error | Distance to true source | Partial (CAFO matching) |
| Plume extent accuracy | Area overlap | No |
| Concentration profile error | Vertical/horizontal cross-sections | No |

---

## 9. Key References

### Methane Detection Papers

1. Vaughan et al. (2024) "CH4Net: a deep learning model for monitoring methane super-emitters with Sentinel-2 imagery" - Atmospheric Measurement Techniques, 17, 2583-2593
2. Rouet-Leduc & Hulbert (2024) "Automatic detection of methane emissions in multispectral satellite imagery using a vision transformer" - Nature Communications, 15, 4106
3. Radman et al. (2023) "S2MetNet: A novel dataset and deep learning benchmark for methane point source quantification using Sentinel-2 satellite imagery" - Remote Sensing of Environment, 294, 113708
4. Xu et al. (2025) "MEQNet: Deep Learning for Methane Point Source Emission Quantification from Sentinel-2 Observations" - NeurIPS Climate Change Workshop
5. Tran et al. (2025) "Robust Small Methane Plume Segmentation in Satellite Imagery" - arXiv:2508.16282
6. Tiemann et al. (2024) "Machine Learning for Methane Detection and Quantification from Space - A Survey" - arXiv:2408.15122

### Super-Resolution Papers

7. Liang et al. (2021) "SwinIR: Image Restoration Using Swin Transformer" - ICCV Workshops
8. Chen et al. (2023) "HAT: Hybrid Attention Transformer for Image Restoration" - CVPR / IEEE TPAMI
9. Wu et al. (2026) "CDEHAT: Conditional Diffusion-Assisted Enhanced Hybrid Attention Transformer for Remote Sensing Imagery Super-Resolution" - ISPRS Journal
10. Wang et al. (2026) "Cross-platform super-resolution: A diffusion model approach for enhancing satellite imagery with aerial data" - Int. J. Applied Earth Observation
11. Saharia et al. (2022) "Image Super-Resolution via Iterative Refinement" - IEEE TPAMI (SR3 diffusion)

### Methane Retrieval Physics

12. Varon et al. (2021) "Sentinel-2 satellite observations of methane point sources" - multi-band-pass retrieval method
13. Green et al. (2022) "NASA's Earth Surface Mineral Dust Source Investigation: The EMIT Instrument"
14. Thorpe et al. (2023) "EMIT methane point source quantification"
15. Cusworth et al. (2024) "EMIT methane plume catalog and analysis"

### Multi-Task Learning

16. Kendall et al. (2018) "Multi-Task Learning Using Uncertainty to Weigh Losses" - CVPR
17. Vandenhende et al. (2021) "Multi-Task Learning for Dense Prediction Tasks: A Survey" - IEEE TPAMI

### Satellite Data Resources

18. EMIT L2B CH4 Plume Complexes V001/V002 - NASA LP DAAC (DOI: 10.5067/EMIT/EMITL2BCH4PLM.001)
19. EMIT L2B Methane Enhancement Data V001/V002 - NASA LP DAAC
20. Google Earth Engine: NASA/EMIT/L2B/CH4PLM dataset
