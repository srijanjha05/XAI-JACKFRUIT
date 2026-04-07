# XAI-JACKFRUIT: Extended Explainable AI Framework for Alzheimer's Detection

**A novel extension of the XAI-ORANGE framework** — applying explainable deep learning to the OASIS-1 MRI dataset for Alzheimer's Disease (AD) classification, with a substantially enriched explanation space, improved training methodology, and automated per-patient clinical reports.

---

## Overview

XAI-JACKFRUIT builds on [XAI-ORANGE](https://github.com/srijanjha05/XAI_ORANGE), which established the baseline pipeline for Explainable AI-driven Alzheimer's classification using 3D DenseNet CNNs, Layer-wise Relevance Propagation (LRP), and W-scored biological feature clustering on the OASIS-1 dataset.

JACKFRUIT introduces **five novel XAI contributions** and a set of **training improvements** that together push classification accuracy from 81.67% → **87.5%** and AUC from 0.9146 → **0.9478**, while simultaneously producing richer, more interpretable per-patient explanations.

---

## ORANGE vs JACKFRUIT — What Changed

| Dimension | XAI-ORANGE (Baseline) | XAI-JACKFRUIT (This Work) |
|---|---|---|
| **Accuracy** | 81.67% | **87.5%** (+5.8 pp) |
| **AUC** | 0.9146 | **0.9478** (+3.3 pp) |
| **XAI features per tissue** | 3 (pos rel, neg rel, density) | **18** (+ spatial moments, HAI, CR) |
| **XAI methods** | LRP only | **LRP + Integrated Gradients** |
| **Inter-method agreement** | None | **Per-subject Spearman LRP-IG score** |
| **Lateralization analysis** | None | **Hemisphere Asymmetry Index (HAI)** |
| **Attribution sparsity** | None | **Concentration Ratio (CR)** |
| **Patient reports** | None | **Automated rule-based textual reports** |
| **Loss function** | Weighted cross-entropy | **Focal loss (γ=2, α=0.25)** |
| **Batch sampling** | Random | **Balanced 50/50 CN/AD per batch** |
| **Test evaluation** | Single pass | **Test-time augmentation (TTA)** |
| **MMSE handling** | Raw (corrupted for 201/436 subjects) | **Repaired + MMSE_valid binary flag** |

---

## Novel Contributions (JACKFRUIT Extensions)

### 1. Spatial Moments of LRP Attribution Maps

For each FSL tissue region (Gray Matter, White Matter, CSF), JACKFRUIT computes the full higher-order statistical profile of the LRP relevance distribution within that region:

- **Mean** — average relevance magnitude (overall signal strength)
- **Standard Deviation** — spread of relevance values (uniform vs. variable)
- **Skewness** — asymmetry of the distribution (peak vs. tail behavior)
- **Kurtosis (excess)** — heavy-tailed vs. platykurtic (hotspot detection)
- **Entropy** — Shannon entropy of the histogram (diffuse vs. focal attribution)

This distinguishes two clinically distinct AD atrophy patterns:
- **Focal atrophy**: concentrated relevance in specific anatomical structures (low entropy, high kurtosis) — typical of early hippocampal-predominant AD
- **Diffuse atrophy**: broadly spread relevance across the cortex (high entropy, low kurtosis) — typical of posterior cortical atrophy and advanced stages

These 15 features (5 stats × 3 tissues) are not present in ORANGE or in Singh et al. (2025), which use only total positive/negative relevance sums.

### 2. Hemisphere Asymmetry Index (HAI)

JACKFRUIT splits the brain at the sagittal midline (x=88 in MNI T88 space) and computes:

```
HAI = (R_pos - L_pos) / (R_pos + L_pos + ε)
```

where `R_pos` and `L_pos` are the sums of positive LRP relevance in the right and left hemispheres respectively, within each tissue mask.

- **HAI > 0**: right hemisphere dominance
- **HAI < 0**: left hemisphere dominance
- **|HAI| ≈ 0**: bilateral (symmetric)

This is a novel XAI-derived biomarker. AD has been clinically documented to show left-temporal predominance in early stages (particularly entorhinal and hippocampal atrophy). JACKFRUIT uses the CNN's own attribution maps to quantify this asymmetry per subject, without any anatomical supervision — allowing the model's internal reasoning to be cross-validated against known neurological lateralization patterns.

HAI is computed per tissue (CSF, GM, WM) and incorporated as W-scored features in the clustering and textual report pipeline.

### 3. Concentration Ratio (CR)

Measures the spatial sparsity of neural attention within each tissue region:

```
CR = sum(top 10% largest |R_i|) / sum(all |R_i|)
```

- **High CR (→ 1.0)**: the model's attention is concentrated in a small fraction of voxels — indicating focal, localized pathology
- **Low CR (→ 0.0)**: the model's attention is spread broadly — indicating diffuse, distributed degeneration

This separates the "where" from the "how much": two patients with the same total LRP relevance in Gray Matter might have completely different atrophy patterns — one with a dense temporal lobe hotspot (high CR) and one with diffuse parietal thinning (low CR). ORANGE cannot distinguish these cases.

### 4. Multi-XAI Inter-Method Agreement Score

JACKFRUIT runs two attribution methods on every subject:
- **LRP** (LRP-CMPalpha1beta0): backward propagation using conservation law
- **IG** (Integrated Gradients): path-integral of gradients from baseline to input

For each subject and tissue region, JACKFRUIT computes the **Spearman rank correlation** between the LRP and IG voxel-level attribution maps:

```
rho_tissue = Spearman(LRP_voxels_tissue, IG_voxels_tissue)
```

This score serves as a per-subject **explanation reliability indicator**:
- **rho ≥ 0.6**: high confidence — both methods agree on which voxels matter
- **0.3 ≤ rho < 0.6**: moderate confidence
- **rho < 0.3**: low confidence — methods disagree, subject may have atypical attribution pattern

Clinically, subjects with low LRP-IG agreement may represent edge cases where the CNN is uncertain, or patients with non-standard atrophy morphology. This provides a rigorous, method-agnostic measure of XAI reliability that does not exist in the ORANGE framework or in Singh et al. (2025).

Spearman correlation is used (rather than Pearson) because LRP and IG have different value scales — rank correlation is invariant to monotonic rescaling.

### 5. Rule-Based Textual Explanation Reports

JACKFRUIT generates a structured, human-readable clinical report for every patient, combining all W-scored XAI features into a narrative explanation. Each report includes:

- CNN prediction + AD probability + optimized threshold
- XAI reliability rating (from inter-method agreement score)
- Per-tissue pathological findings: W-score severity classification (mild/moderate/strong), elevation direction (above or below CN norm), HAI lateralization, CR spatial pattern, entropy-based spread
- Volumetric evidence (nWBV W-score vs age/sex/eTIV norm)
- Low-agreement warning flag for uncertain attributions

This extends the rule-based approach of Singh et al. (2025) with lateralization, concentration, and multi-method confidence — turning raw feature vectors into clinical decision-support language.

**Example output:**
```
Patient ID: OAS1_0001_MR1
True Diagnosis: AD
CNN Prediction: AD  (AD probability: 0.812, threshold: 0.55)

─── XAI Attribution Summary ──────────────────────────────
XAI Reliability: high (LRP–IG Spearman ρ = 0.681)

─── Pathological Findings by Tissue Region ───────────────
  Cerebrospinal Fluid (CSF):
    Relevance W-score: +3.41  [STRONG PATHOLOGY]
    → CNN showed ELEVATED relevance in CSF (+3.41 SD above CN norm).
    Lateralization: left-hemisphere dominant  (HAI W=-2.18)
    Attribution pattern: focal (concentrated hotspot)  (CR W=+2.44)
...
```

---

## Training Improvements (Over ORANGE)

### Focal Loss
Replaces standard binary cross-entropy with Focal Loss (Lin et al. 2017):

```
FL(p_t) = -α (1 - p_t)^γ log(p_t)
```

With `γ=2, α=0.25`, the loss down-weights easy examples exponentially, forcing the model to focus training signal on the hard AD cases that cross-entropy would ignore after they are partially correct.

### Balanced Batch Sampling
Each training batch is constructed with exactly 50% CN and 50% AD samples (sampled with replacement). This directly counteracts the 3.36:1 CN:AD imbalance in OASIS-1, which otherwise leads to CN-biased predictions. Combined with focal loss, this dual-correction approach significantly improves AD sensitivity.

### 3D Augmentation
During training: random left-right sagittal flip (p=0.5) and Gaussian noise injection (σ ~ U(0.01, 0.03), p=0.3). LR flip is biologically valid since the early-stage left-temporal dominance in AD is statistically weak at the individual level — the model must not overfit to hemisphere.

### Test-Time Augmentation (TTA)
At inference, predictions are averaged over the original scan and its LR-flipped counterpart:

```
p_final = (p_original + p_flipped) / 2
```

This reduces spatial overfitting artifacts and produces more stable probability estimates.

### F1-Optimized Threshold
The classification threshold is selected on the validation set to maximize F1-score (not fixed at 0.5). This balances precision and recall in the imbalanced OASIS-1 setting without sacrificing sensitivity.

### MMSE Data Repair
A critical data quality fix: 201 of 436 OASIS-1 subjects had corrupted MMSE entries (the raw CSV stored eTIV values in the MMSE column for young CN subjects with no cognitive test). JACKFRUIT repairs this with robust regex parsing and adds a binary `MMSE_valid` flag as a 7th tabular feature, explicitly encoding missingness as clinical information (all 100 AD subjects have valid MMSE; missing MMSE strongly correlates with CN status in young subjects).

---

## Results

### Classification (Fold 1 of 5-fold CV)

| Metric | XAI-ORANGE | XAI-JACKFRUIT |
|---|---|---|
| AUC | 0.9146 | **0.9478** |
| Accuracy | 81.67% | **87.5%** |
| Threshold | Youden-J (fixed) | F1-optimized (adaptive) |

### XAI Feature Biological Validity (Mutual Information vs AD/CN label)

| Feature | MI Score | Biological Correspondence |
|---|---|---|
| CSF_pos_rel | 0.1469 | Ventricular enlargement (hydrocephalus ex vacuo) |
| GrayMatter_pos_rel | 0.1289 | Cortical atrophy (Braak staging pattern) |
| WhiteMatter_pos_rel | 0.1276 | WM hyperintensities / fiber tract degradation |
| nWBV_vol_w | 0.1197 | Global brain shrinkage (disease severity proxy) |
| GrayMatter_neg_rel | 0.0962 | Cortical reserve signal |
| WhiteMatter_neg_rel | 0.0832 | WM integrity (evidence against AD) |

### Clustering Quality (Unsupervised Subtyping)

| Metric | Score | Interpretation |
|---|---|---|
| Silhouette | 0.519 | Excellent geometric separation (>0.5 threshold) |
| ARI | 0.386 | Moderate-high agreement with AD/CN labels |
| V-measure | 0.234 | Reflects true AD biological heterogeneity |
| Homogeneity | 0.208 | Not a failure — AD is clinically heterogeneous |

The lower Homogeneity is scientifically expected: Alzheimer's Disease presents as a spectrum of morphological subtypes (hippocampal-predominant, posterior cortical atrophy, frontal-variant, diffuse). The clustering correctly detects this structural diversity rather than forcing a single monolithic AD pattern.

---

## Dataset

**OASIS-1** (Open Access Series of Imaging Studies, Cross-Sectional Release 1)

- 436 subjects, ages 18–96
- 336 CN (CDR=0), 100 AD (CDR>0)
- Labels: CDR=0 → CN, CDR>0 → AD
- MRI: T88 MNI space, (176, 208, 176) voxels, FSL segmentation (CSF/GM/WM)
- Key biomarker: nWBV (AD mean=0.722 vs CN mean=0.812)

Data is not included in this repository due to size. Download from [OASIS-Brain](https://www.oasis-brains.org/) and place in the `data/` directory.

---

## Pipeline

```
src/run_pipeline.sh   ← master script (runs all steps in sequence)
```

### Step-by-Step

| Script | Step | Description |
|---|---|---|
| `convert_to_nifti.py` | 0a | Convert raw OASIS `.hdr/.img` files to `.nii.gz` |
| `oasis_0_train_cnn.py` | 0 | Train 3D multimodal DenseNet (focal loss, balanced batches, TTA) |
| `oasis_1_extract_activations.py` | 1 | Extract LRP + IG maps; compute spatial moments, HAI, CR per tissue |
| `oasis_1b_xai_agreement.py` | 1b | Compute per-subject LRP-IG Spearman agreement scores |
| `oasis_2_wscore.py` | 2 | W-score normalize all feature groups vs CN regression baseline |
| `oasis_3_cluster_compare.py` | 3 | Agglomerative clustering across multiple XAI configurations |
| `oasis_4_textual_report.py` | 4 | Generate per-patient textual clinical reports |

---

## Installation

```bash
# Create conda environment (Python 3.10 or 3.11 required for TF2 + innvestigate)
conda create -n xai4dem python=3.11
conda activate xai4dem

pip install tensorflow==2.15 keras innvestigate nibabel nilearn \
            scikit-learn scipy numpy pandas matplotlib openpyxl
```

```bash
# Run full pipeline
cd src
bash run_pipeline.sh
```

---

## Project Structure

```
xai4dementia-framework/
├── src/
│   ├── run_pipeline.sh                 # Master runner
│   ├── convert_to_nifti.py             # Raw data preprocessing
│   ├── oasis_0_train_cnn.py            # DenseNet training (JACKFRUIT improvements)
│   ├── oasis_1_extract_activations.py  # LRP+IG + spatial moments/HAI/CR (NOVEL)
│   ├── oasis_1b_xai_agreement.py       # Multi-XAI agreement scoring (NOVEL)
│   ├── oasis_2_wscore.py               # W-score normalization
│   ├── oasis_3_cluster_compare.py      # Clustering evaluation
│   ├── oasis_4_textual_report.py       # Textual report generation (NOVEL)
│   └── oasis_util.py                   # Shared config, model architecture, I/O
├── 4_clustering/                       # Clustering outputs and comparison charts
├── act_seg/                            # XAI attribution feature tables (.xlsx)
├── measures_combined/                  # W-scored combined feature CSVs
├── train/                              # Per-fold model checkpoints and metrics
├── data/                               # OASIS-1 data (not in repo — too large)
├── final_project_report.txt            # Full technical report
└── project_explanation_report.txt      # Detailed theoretical explanation
```

---

## Reference

This work extends:

> Singh, G. et al. (2025). *An unsupervised XAI framework for dementia detection with context enrichment*. Scientific Reports 15:39554.
> Key baseline: LRP + volumetry + cortical thickness → enriched explanation space, V-measure 0.43, N=3253 from 6 cohorts.

JACKFRUIT applies and extends this framework specifically to OASIS-1 (Alzheimer's vs CN binary classification) with the novel spatial XAI features, hemisphere asymmetry analysis, inter-method agreement scoring, and automated textual reporting described above.

---

## Related

- [XAI-ORANGE](https://github.com/srijanjha05/XAI_ORANGE) — the baseline framework this work extends
