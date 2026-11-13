> **HISTORICAL PROPOSAL / REPORT.** Active work has restarted at [GTM_Model6](../../research/GTM_Model6/README.md). Claims and recommendations below belong to earlier research; they are not current results or execution instructions.

# ERSRR Architecture Proposal: Recommended Stacks

**Date:** April 2026
**Companion to:** RESEARCH_REPORT.md
**Status:** Proposal for team evaluation

---

## Decision Framework

Before selecting an architecture, the team must decide on key constraints:

| Question | Options | Recommendation |
|----------|---------|----------------|
| **GPU budget?** | None (CPU only) / Single consumer GPU / Multi-GPU | Determines model size |
| **Time to next deliverable?** | Weeks / Months / Semester | Determines complexity |
| **Priority: accuracy or speed?** | Best possible / Good enough / Real-time | Determines architecture family |
| **Dataset expansion feasible?** | Yes (can run GEE pipeline) / Limited / No | Determines augmentation strategy |
| **Framework preference?** | Keras (current) / PyTorch / Either | Determines code migration |

---

## Architecture Stack 1: U-Net + Feature Engineering (Recommended First Step)

**Why:** Highest ROI for effort. U-Net is the proven architecture for this exact task (CH4Net, Rouet-Leduc 2024, Tran 2025). Feature engineering (spectral indices) is free performance. Pretrained encoders compensate for small datasets.

### Architecture

```
Input Pipeline:
  Raw S2 bands (B2, B3, B4, B11, B12)
  + Computed features:
    - Varon ratio: V(B12, B11) = (c*B12 - B11) / B11
    - NDVI: (B8 - B4) / (B8 + B4)   [if B8 available]
    - B12/B11 ratio
  => 7-8 channel input

Encoder: ResNet34 (ImageNet pretrained) or EfficientNet-B4
  - Freeze early layers, fine-tune deeper layers
  - Transfer learning compensates for ~99 sample dataset

Decoder: Standard U-Net upsampling path
  - Skip connections from encoder
  - Conv blocks: Conv2D -> BN -> ReLU -> Conv2D -> BN -> ReLU

Head 1 (Regression): 1x1 Conv -> Sigmoid (CH4 concentration 0-1)
Head 2 (Segmentation): 1x1 Conv -> Sigmoid (plume mask 0-1)
```

### Loss Functions

```python
# Regression head
L_reg = L1_loss(valid_mask) + 0.1 * GradientLoss(valid_mask)
  # L1 is more robust to outliers than MSE for sparse plumes
  # Gradient loss preserves plume edges

# Segmentation head
L_seg = 0.5 * BCE_loss + 0.5 * Dice_loss
  # Dice handles extreme class imbalance (plumes << 1% of pixels)
  # Alternative: Focal Loss (gamma=2) if Dice insufficient

# Multi-task weighting
L_total = w_reg * L_reg + w_seg * L_seg
  # Start with w_reg=1.0, w_seg=1.0
  # Optionally use Kendall uncertainty weighting (learned)
```

### Implementation

```python
# Using segmentation_models_pytorch (or equivalent Keras)
import segmentation_models as smp

model = smp.Unet(
    encoder_name="resnet34",        # pretrained on ImageNet
    encoder_weights="imagenet",
    in_channels=7,                   # 5 raw + 2 engineered features
    classes=2,                       # regression + segmentation
    activation=None,                 # apply sigmoid per-head
)
```

### Expected Performance

| Metric | Expected | Basis |
|--------|----------|-------|
| Training time | ~30 min (GPU) | Small model, small dataset |
| PSNR improvement | +2-4 dB over current | U-Net > ViT for segmentation with limited data |
| Plume IoU | 0.55-0.65 | CH4Net baseline with similar setup |
| Detection threshold | ~1000-2000 kg/h | Limited by S2 spectral resolution |

### Pros and Cons

| Pros | Cons |
|------|------|
| Fast to implement (1-2 weeks) | U-Net lacks global context for large plumes |
| Pretrained encoders work with small data | Fixed receptive field |
| Proven for this exact task in literature | May miss fine spatial details |
| Well-understood, easy to debug | Limited room for improvement |

---

## Architecture Stack 2: Hybrid Attention Transformer (Recommended for Maximum Accuracy)

**Why:** HAT (Chen 2023, CVPR) is the current SOTA for image restoration/super-resolution. CDEHAT (2026) extends it for remote sensing. This is the architecture most likely to produce state-of-the-art results for the S2->EMIT downscaling task.

### Architecture

```
Input Pipeline:
  10 S2 bands (B2,B3,B4,B5,B8,B8A,B9,B11,B12 + B1 or B6)
  resampled to 20m, cropped to 256x256
  + 2 engineered features (Varon ratio, B12/B11)
  => 12 channel input, 256x256

Shallow Feature Extraction:
  Conv2D(12, 180, kernel_size=3, padding=1)

Deep Feature Extraction (6x RHAG - Residual Hybrid Attention Groups):
  Each RHAG contains 6x HAB (Hybrid Attention Blocks):
    - Window-based Multi-Head Self-Attention (W-MSA), window_size=16
    - Shifted W-MSA (SW-MSA) for cross-window interaction
    - Channel Attention Block (CAB) for global information
    - Overlapping Cross-Attention Block (OCAB) for pixel activation

Skip connections at RHAG outputs (3 levels)

Multi-Task Decoder:
  Pixel-shuffle upsampling (2x per level, 3 levels = 8x if starting from 32x32)
  OR Conv2DTranspose with skip connections from encoder stages

  Head 1: Regression -> Conv2D(1, sigmoid)
  Head 2: Segmentation -> Conv2D(1, sigmoid)
```

### Key Design Decisions

**Why HAT over plain SwinIR:**
- HAT's channel attention activates more input pixels (proven via LAM analysis)
- +0.52 dB PSNR over SwinIR on standard benchmarks
- Overlapping cross-attention captures information between non-adjacent windows

**Window size = 16 (not 8):**
- Methane plumes span 10-100+ pixels at 20m resolution
- Larger windows capture plume-scale context
- +0.36 dB improvement over window_size=8

**Same-task pre-training:**
- Pre-train on generic remote sensing SR (e.g., Sentinel-2 -> higher resolution aerial)
- Fine-tune on S2+EMIT pairs
- Shown to outperform multi-task pre-training by +0.5 dB

### Loss Functions

```python
# Regression: combination of pixel, structural, and perceptual
L_reg = lambda1 * L1_loss(valid_mask) \
      + lambda2 * SSIM_loss(valid_mask) \
      + lambda3 * GradientLoss(valid_mask)

# Segmentation
L_seg = 0.5 * FocalLoss(gamma=2) + 0.5 * DiceLoss

# Total with learned task weights
L_total = exp(-s_reg) * L_reg + s_reg + exp(-s_seg) * L_seg + s_seg
# where s_reg, s_seg are learnable parameters (Kendall weighting)
```

### Implementation Notes

- Start from the official HAT codebase: github.com/XPixelGroup/HAT
- Adapt input channels from 3 (RGB) to 12 (multi-band + features)
- Add second output head for segmentation
- Same-task pre-train on remote sensing SR data before fine-tuning on EMIT pairs

### Expected Performance

| Metric | Expected | Basis |
|--------|----------|-------|
| Training time | 4-8 hours (GPU) | Larger model |
| PSNR improvement | +4-6 dB over current | HAT benchmarks |
| Plume IoU | 0.60-0.70 | Improved spatial detail |
| Detection threshold | ~500-1500 kg/h | Better feature extraction |

### Pros and Cons

| Pros | Cons |
|------|------|
| SOTA for image restoration | Complex implementation (4-8 weeks) |
| Best spatial detail recovery | Requires significant GPU memory |
| Pre-training strategy proven | More hyperparameters to tune |
| Largest room for improvement | Overkill if dataset remains ~99 samples |

---

## Architecture Stack 3: ViT-U-Net with Synthetic Data (Best Published Results)

**Why:** This replicates the approach of Rouet-Leduc & Hulbert (Nature Communications 2024), which achieved the best published results for methane detection from Sentinel-2. The key insight is that massive synthetic data + ViT global context + bi-temporal input is the winning combination.

### Architecture

```
Input Pipeline:
  10 S2 bands at TWO time steps (current + reference)
  => 20 channel input, 256x256

  OR (simpler version):
  5 S2 bands at current time step + 5 at reference
  + temporal difference features (B12_diff, B11_diff)
  => 12 channel input

Encoder: ViT-Base (patch_size=16)
  - Pretrained on ImageNet or SatMAE
  - 12 transformer layers, 12 heads, dim=768
  - Captures long-range dependencies (plume extent)

Decoder: U-Net convolutional decoder
  - Deconvolutional upsampling
  - Skip connections from ViT intermediate layers

Heads: Regression + Segmentation (same as Stack 2)
```

### Synthetic Data Generation Pipeline

This is the critical differentiator. Generate 10,000+ training samples:

```python
# 1. Collect plume-free S2 backgrounds (diverse surfaces)
background_scenes = collect_cloud_free_s2(regions="global", max_cloud=5)

# 2. Generate synthetic Gaussian plumes
for i in range(num_synthetic):
    # Random emission rate: 100 - 10000 kg/h
    emission_rate = random.uniform(100, 10000)

    # Random wind speed and direction
    wind_speed = random.uniform(1, 10)  # m/s
    wind_dir = random.uniform(0, 360)   # degrees

    # Generate 2D Gaussian plume
    plume = gaussian_plume_model(
        emission_rate, wind_speed, wind_dir,
        grid_size=256, resolution=20  # meters
    )

    # Add turbulence (colored noise)
    plume += colored_noise_2d(shape=(256,256), correlation_length=5)

    # Embed into S2 B12 using Beer-Lambert law
    # B12_absorbed = B12_original * exp(-alpha * CH4_column)
    modified_B12 = background_B12 * np.exp(-absorption_coeff * plume)

    # Create 6-band training sample
    input_bands = [B2, B3, B4, B11, modified_B12]
    target = plume  # normalized CH4 enhancement
    mask = (plume > threshold)  # binary plume mask
```

### Training Strategy

```
Phase 1: Train on synthetic data only (10,000+ samples)
  - Learn basic plume morphology and spectral signatures
  - 50 epochs, lr=1e-3

Phase 2: Fine-tune on real S2+EMIT pairs (~100-1000 samples)
  - Adapt to real instrument characteristics
  - 100 epochs, lr=1e-5 (10x lower)

Phase 3: Multi-task training
  - Add segmentation head
  - Joint optimization
  - 50 epochs, lr=1e-5
```

### Expected Performance

| Metric | Expected | Basis |
|--------|----------|-------|
| Training time | 8-16 hours (GPU) | Large synthetic dataset |
| Plume detection F1 | 0.70-0.84 | CH4Net/Rouet-Leduc baselines |
| Detection threshold | ~200-500 kg/h | With sufficient synthetic data |
| False positive rate | <3% pixel-wise | Rouet-Leduc result |

### Pros and Cons

| Pros | Cons |
|------|------|
| Best published results replicated | Synthetic data pipeline is substantial work |
| Massive dataset compensates for model size | Gaussian plumes may not match real physics |
| Bi-temporal input reduces false positives | Requires fetching reference S2 images |
| Proven in Nature Communications paper | Risk of sim-to-real gap |

---

## Architecture Comparison Matrix

| Criterion | Stack 1: U-Net | Stack 2: HAT | Stack 3: ViT+Synthetic |
|-----------|---------------|-------------|----------------------|
| **Implementation time** | 1-2 weeks | 4-8 weeks | 6-12 weeks |
| **GPU requirement** | 4-8 GB | 12-24 GB | 8-16 GB |
| **Data needed** | 100-500 real | 500+ real | 10,000+ synthetic + 100 real |
| **Expected accuracy** | Good | Best (pixel-level) | Best (plume-level) |
| **Risk level** | Low | Medium | Medium-High |
| **Framework** | Keras or PyTorch | PyTorch | PyTorch |
| **Pre-trained weights** | ImageNet encoder | ImageNet + RS SR | SatMAE or ImageNet |
| **Key innovation** | Spectral indices + transfer learning | Hybrid attention SR | Synthetic data at scale |
| **Best for** | Quick improvement | Maximum spatial detail | Maximum detection rate |
| **Publication potential** | Low (incremental) | High (novel application) | High (replicated SOTA) |

---

## Recommended Roadmap

### Phase 1: Quick Wins (Weeks 1-2)

Implement Stack 1 (U-Net + Feature Engineering) as an immediate improvement:

1. **Add feature engineering:** Compute Varon ratio and B12/B11 as extra input channels
2. **Switch to pretrained encoder:** ResNet34 with ImageNet weights, freeze first 2 stages
3. **Add data augmentation:** Random flips, rotations, brightness/contrast jitter, random crops
4. **Fix loss functions:** L1 (not MSE) for regression, Dice + BCE for segmentation
5. **Add proper metrics:** IoU, F1, PSNR, SSIM
6. **Expand validation set:** At least 20 samples, geographic split

Expected result: Significant improvement over current ViT with minimal effort.

### Phase 2: Data Expansion (Weeks 2-4)

1. **Run data collection pipeline** to gather 500-1000+ paired S2+EMIT samples
2. **Implement synthetic plume generation** (Gaussian model + Beer-Lambert embedding)
3. **Generate 5,000+ synthetic training samples** with diverse backgrounds and emission rates
4. **Add bi-temporal input** (current + reference S2 image)

### Phase 3: Advanced Architecture (Weeks 4-8)

Based on Phase 1-2 results, choose between Stack 2 (HAT) or Stack 3 (ViT+Synthetic):

- **If spatial detail is priority** (accurate concentration maps) -> Stack 2: HAT
- **If detection rate is priority** (finding plumes, matching CAFOs) -> Stack 3: ViT+Synthetic

### Phase 4: Integration (Weeks 8-10)

1. Integrate trained model into the website prediction pipeline
2. Real-time S2 tile fetching + prediction overlay
3. CAFO matching and source attribution
4. KML/export generation

---

## Technology Stack Recommendations

### Framework

| Component | Current | Recommended | Reason |
|-----------|---------|-------------|--------|
| ML Framework | Keras 3 | PyTorch + segmentation_models_pytorch | Better ecosystem for this task; all published code is PyTorch |
| Data Loading | Manual numpy | torchgeo / rasterio with DataLoader | Efficient streaming for large GeoTIFF datasets |
| Experiment Tracking | None | Weights & Biases or TensorBoard | Essential for hyperparameter tuning |
| Hyperparameter Tuning | Manual bash scripts | Optuna or Ray Tune | Systematic search |
| Geospatial | rasterio | rasterio + xarray + rioxarray | Better handling of CRS, transforms, nodata |
| Synthetic Data | N/A | Custom pipeline (see Phase 2) | Critical for scaling |

### Hardware Requirements

| Phase | Minimum GPU | Recommended GPU | VRAM |
|-------|------------|-----------------|------|
| Phase 1 (U-Net) | GTX 1060 6GB | RTX 3060 12GB | 6-12 GB |
| Phase 2 (Data gen) | CPU sufficient | Any GPU for validation | 4 GB |
| Phase 3 (HAT/ViT) | RTX 3080 10GB | RTX 4090 24GB | 12-24 GB |
| Phase 3 (Synthetic) | RTX 3070 8GB | RTX 4080 16GB | 8-16 GB |

If GPU access is limited, consider:
- Google Colab Pro (A100 40GB)
- Kaggle Notebooks (T4 16GB, free)
- University HPC clusters

---

## Quick-Start: Minimal Changes for Immediate Improvement

If the team has limited time, these 5 changes to the existing Keras codebase will yield the biggest improvement:

### 1. Add Spectral Features (ImagePreprocessing.py)

```python
# After loading bands, compute:
B11 = image_data[:, :, 3]  # SWIR-1
B12 = image_data[:, :, 4]  # SWIR-2

# Varon ratio
c = np.sum(B12 * B11) / (np.sum(B11 * B11) + 1e-6)
varon_ratio = (c * B12 - B11) / (B11 + 1e-6)

# Band ratio
band_ratio = B12 / (B11 + 1e-6)

# Stack as extra channels -> input_shape becomes (256, 256, 7)
X_split = np.stack([B2, B3, B4, B11, B12, varon_ratio, band_ratio], axis=-1)
```

### 2. Add Data Augmentation (ERSRR_Model.py)

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.5),  # 180 degrees
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])
# Apply to X only, not Y
```

### 3. Switch to Pretrained Encoder (ERSRR_Model.py)

Replace the custom ViT with a ResNet backbone:
```python
base_model = keras.applications.ResNet34(
    weights='imagenet',
    include_top=False,
    input_shape=(256, 256, 7)
)
# Freeze first 2 stages
for layer in base_model.layers[:50]:
    layer.trainable = False
```

### 4. Fix Loss Function (ERSRR_Model.py)

```python
# Replace weighted_masked_mse with L1:
def masked_l1(y_true, y_pred):
    valid_mask = ops.logical_not(ops.isnan(y_true))
    valid_mask = ops.cast(valid_mask, y_pred.dtype)
    y_true_safe = ops.where(ops.isnan(y_true), ops.zeros_like(y_true), y_true)
    abs_err = ops.abs(y_pred - y_true_safe) * valid_mask
    return ops.sum(abs_err) / (ops.sum(valid_mask) + 1e-6)
```

### 5. Add Evaluation Metrics (ERSRR_Model.py)

```python
def psnr_metric(y_true, y_pred):
    valid = ops.logical_not(ops.isnan(y_true))
    valid = ops.cast(valid, y_pred.dtype)
    y_true_safe = ops.where(ops.isnan(y_true), ops.zeros_like(y_true), y_true)
    mse = ops.sum(ops.square(y_pred - y_true_safe) * valid) / (ops.sum(valid) + 1e-6)
    return 10.0 * ops.log(1.0 / (mse + 1e-6)) / ops.log(10.0)
```

---

## Summary

| Priority | Action | Impact | Effort |
|----------|--------|--------|--------|
| 1 | Expand dataset (synthetic + real) | Very High | Medium |
| 2 | Add spectral feature engineering | High | Low |
| 3 | Switch to U-Net with pretrained encoder | High | Low |
| 4 | Add data augmentation | Medium-High | Low |
| 5 | Add bi-temporal input | High | Medium |
| 6 | Fix loss functions (L1 + Dice + Focal) | Medium | Low |
| 7 | Implement proper evaluation metrics | Medium | Low |
| 8 | Upgrade to HAT or ViT architecture | High | High |
| 9 | Migrate to PyTorch | Medium | Medium |
| 10 | Add wind data as auxiliary input | Medium | Medium |

The single highest-impact action is **synthetic data generation** (Priority 1). No architecture change will compensate for a 99-sample dataset. With 10,000+ synthetic samples and a simple U-Net, the project will outperform any architecture on 99 samples.
