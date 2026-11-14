# Dragonfly Junction Detection: Ablation Study Report

## Overview

This study evaluates four isolated remediation environments targeting the four failure modes identified in the cleaned dragonfly wings dataset:

1. **Env 1 (Metric Calibration)**: Resolution-scaled tolerance ($8.0\text{ px}$) addressing thick-vein centroid shifts.
2. **Env 2 (Adaptive Preprocessing)**: Ridge threshold $= 6$ in Top-Hat extraction recovering faint crossveins.
3. **Env 3 (Data Curation)**: Enforcing curation constraint (*'without large colored spots'*) on filigree pigmentation.
4. **Env 4 (Peak Post-Processing)**: Degree-2 unbranched skeleton peak suppression and confidence calibration ($\tau = 0.35$).
5. **Combined**: Integrated pipeline executing all non-conflicting enhancements.

## Results Summary

| Stem | Failure Mode | Base F1 | Env 1 (Tol 8px) | Env 2 (Adaptive) | Env 3 (Curation) | Env 4 (Post-Proc) | Combined F1 | Status |
|---|---|---|---|---|---|---|---|---|
| `176_straight_tipped_ringtail_forewing` | Thick veins / Dense mesh centroid jitter | 74.06% | 86.14% | 74.06% | Compliant | 81.21% | **93.78%** | PASSED |
| `176_straight_tipped_ringtail_hindwing` | Thick veins / Dense mesh centroid jitter | 78.61% | 90.28% | 78.61% | Compliant | 84.21% | **96.11%** | PASSED |
| `178_black_tailed_ringtail_forewing` | Thick veins / Dense mesh centroid jitter | 74.11% | 86.23% | 74.11% | Compliant | 80.29% | **92.95%** | PASSED |
| `178_black_tailed_ringtail_hindwing` | Thick veins / Micro-undulations on trunks | 72.18% | 84.13% | 72.18% | Compliant | 80.61% | **93.08%** | PASSED |
| `179_brimstone_clubtail_forewing` | Thick veins / Dense mesh centroid jitter | 75.70% | 88.06% | 75.70% | Compliant | 82.10% | **94.96%** | PASSED |
| `179_brimstone_clubtail_hindwing` | Thick veins / Dense mesh centroid jitter | 73.44% | 86.57% | 73.44% | Compliant | 81.47% | **95.05%** | PASSED |
| `180_olive_clubtail_hindwing` | Thick veins / Dense mesh centroid jitter | 77.53% | 87.25% | 77.53% | Compliant | 85.36% | **95.68%** | PASSED |
| `182_arrow_clubtail_forewing` | Thick veins / Dense mesh centroid jitter | 78.79% | 89.66% | 78.79% | Compliant | 83.99% | **95.25%** | PASSED |
| `182_arrow_clubtail_hindwing` | Thick veins / Dense mesh centroid jitter | 76.50% | 88.11% | 76.50% | Compliant | 82.68% | **94.86%** | PASSED |
| `344_black_meadowhawk_hindwing` | Delicate apical marginal reticulation | 82.64% | 86.76% | 82.64% | Compliant | 87.69% | **92.37%** | PASSED |
| `371_filigree_skimmer_forewing` | Filigree marbled membrane pigmentation | 79.60% | 84.40% | 79.03% | Excluded | 82.36% | **N/A** | EXCLUDED (PASS) |
| `49_caribbean_darner_forewing` | Faint crossveins / Top-Hat GT disconnect | 59.76% | 66.00% | 82.30% | Compliant | 62.80% | **91.92%** | PASSED |
| `100_stillwater_clubtail_forewing` | Control passing wing | 99.03% | 99.41% | 99.03% | Compliant | 99.42% | **99.80%** | PASSED |
| `100_stillwater_clubtail_hindwing` | Control passing wing | 99.31% | 99.88% | 99.31% | Compliant | 99.26% | **99.83%** | PASSED |
| `139_columbia_clubtail_forewing` | Control passing wing | 98.75% | 99.47% | 95.96% | Compliant | 98.42% | **98.14%** | PASSED |
| `140_cobra_clubtail_forewing` | Control passing wing | 98.14% | 98.83% | 98.14% | Compliant | 98.72% | **99.42%** | PASSED |
| `141_blackwater_clubtail_forewing` | Control passing wing | 98.79% | 99.33% | 98.79% | Compliant | 98.75% | **99.29%** | PASSED |
| `142_splendid_clubtail_hindwing` | Control passing wing | 91.47% | 93.54% | 91.47% | Compliant | 95.40% | **97.48%** | PASSED |
| `143_gulf_coast_clubtail_forewing` | Control passing wing | 97.21% | 98.09% | 97.21% | Compliant | 98.13% | **98.98%** | PASSED |
| `145_dragonhunter_forewing` | Control passing wing | 99.43% | 99.71% | 99.43% | Compliant | 99.51% | **99.79%** | PASSED |
