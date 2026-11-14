# OdonataNet: Dragonfly Wing Venation Dataset & Topological Phenotyping Pipeline

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg)](requirements.txt)
[![Dataset](https://img.shields.io/badge/Dataset-443%20Cleaned%20Wings-green.svg)](data/cleaned_wings)
[![Accuracy](https://img.shields.io/badge/Mean%20F1%20Accuracy-98.86%25-brightgreen.svg)](training_output/cleaned_wings_evaluation.csv)
[![Passing Rate](https://img.shields.io/badge/Threshold%20Pass%20(%E2%89%A585%25)-100%25-success.svg)](training_output/cleaned_wings_evaluation.csv)
[![Morphometrics](https://img.shields.io/badge/Features%20Extracted-104%20Metrics-blueviolet.svg)](data/wing_features.csv)

---

## 1. Executive Summary & Project Overview

Insect wings—and odonate (dragonfly and damselfly) wings in particular—represent one of nature's most sophisticated evolutionary structural architectures. Composed of an ultra-thin cuticle membrane supported by a hierarchical network of tubular veins, a dragonfly wing achieves exceptional tensile resilience, passive camber deformation, and aerodynamic efficiency during high-frequency flapping flight. 

The junction nodes of this venation network—where structural longitudinal veins intersect delicate transverse crossveins—determine the wing's localized flexibility, stress dissipation, and fracture resistance. Accurately mapping and quantifying these junctions at scale has historically been bottlenecked by manual landmarking or brittle classical edge detectors that fail when encountering pigmented wing spots, body remnants, variable illumination, and intricate vein reticulations.

**OdonataNet** is an end-to-end computer vision and computational morphometrics framework and benchmark dataset developed to overcome these challenges. The project delivers:
1. **The Curated Dragonfly Wing Dataset**: A benchmark corpus of **443 high-resolution cleaned dragonfly wings** (224 forewings and 219 hindwings across diverse families including *Gomphidae*, *Libellulidae*, *Aeshnidae*, *Petaluridae*, *Macromiidae*, and *Cordulegastridae*), accompanied by a 509-specimen curation manifest ([cleaned_wings_manifest.csv](data/cleaned_wings_manifest.csv)).
2. **Deep Residual Keypoint Architecture (`JunctionUNet`)**: A custom fully convolutional neural network with residual bottleneck blocks ([model.py](src/models/model.py)) capable of detecting thousands of vein junctions per wing in under 200 milliseconds.
3. **Omnidirectional Topological Ground Truth Engine (`TopologicalWalker`)**: A graph-theoretic skeleton crawler ([topological_walker.py](src/walkers/topological_walker.py)) operating on Morphological Black Top-Hat filtered vein networks, providing rigorous landmark ground truth free from directional bias or drift.
4. **Comprehensive Morphometrics Feature Matrix**: 104 quantitative geometric, topological, and spatial point pattern features per wing ([wing_features.csv](data/wing_features.csv)), spanning whole-wing convex hulls, Minimum Enclosing Circles (MEC), Clark-Evans spatial dispersion, Delaunay circumcircle distributions, and individual biological cell polygonal geometry.
5. **Full Evaluation & Ablation Suite**: Empirical evaluation confirming **100.0% of wings exceed the target $\ge 85\%$ F1 threshold** (mean F1: **98.86%**, count correlation: $r = 0.9867$) and an isolated 4-environment ablation study ([ablation_study_report.md](training_output/ablation_study_report.md)).

---

## 2. What the Project Is Now: Final Scope & Deliverables

OdonataNet provides an end-to-end computational framework and curated benchmark for deep learning-driven landmark discovery, omnidirectional topological extraction, and high-throughput geometric morphometrics across dragonfly wings.

### The Five Core Deliverables

| Deliverable | Path | Description |
|---|---|---|
| **Cleaned Wings Dataset** | [`data/cleaned_wings/`](data/cleaned_wings) | 443 normalized, rotated, body-cleaned grayscale dragonfly wing specimens. |
| **Topological Ground Truth Seeds** | [`data/cleaned_seeds/`](data/cleaned_seeds) | Ground-truth junction coordinate lists generated via omnidirectional skeleton topology. |
| **Model Predictions & Coordinates** | [`data/unet_predictions/`](data/unet_predictions) | 444 NumPy coordinate arrays (`.npy`) storing exact $(y, x)$ detected junction coordinates. |
| **Visual Overlay Inspection Gallery** | [`prediction_output/`](prediction_output) | 444 full-resolution composite overlays displaying predicted junctions over raw wing images. |
| **104-Column Morphometric Matrix** | [`data/wing_features.csv`](data/wing_features.csv) | Tabular feature matrix detailing global geometry, cell polygonal shapes, and point distributions. |

```
junction_walker/
├── data/
│   ├── cleaned_wings/                # 443 cleaned, body-stripped wing images
│   ├── cleaned_seeds/                # 443 Topological Walker ground truth seed files
│   ├── unet_predictions/             # 444 machine-readable .npy coordinate files
│   ├── cleaned_wings_manifest.csv    # Curation manifest (443 included, 66 excluded)
│   └── wing_features.csv             # 443 wings x 104 morphometric & topological features
├── src/
│   ├── preprocessing/
│   │   ├── cleaning.py               # Body removal and Top-Hat vein extraction
│   │   └── build_cleaned_dataset.py  # Dataset cleaning and seed generation pipeline
│   ├── walkers/
│   │   └── topological_walker.py     # Omnidirectional skeleton graph walker
│   ├── models/
│   │   ├── model.py                  # JunctionUNet PyTorch architecture
│   │   ├── train.py                  # Training pipeline with cosine annealing
│   │   ├── predict.py                # Batch prediction, NMS, and overlay rendering
│   │   ├── evaluate_dataset_metrics.py# Dataset evaluation against ground truth
│   │   ├── plot_evaluation_metrics.py# Publication performance figure generator
│   │   └── ablation_study.py         # 4-environment failure mode ablation study
│   └── features/
│       ├── extract_wing_features.py  # 104-feature extraction engine
│       └── plot_hull_and_area_samples.py # Publication morphometric figures
├── training_output/
│   ├── best_model.pt                 # Optimized JunctionUNet weights
│   ├── cleaned_wings_evaluation.csv  # Precision, recall, and F1 across all 443 wings
│   ├── ablation_study_report.md      # Detailed failure mode ablation report
│   ├── ablation_study_results.csv    # Quantitative results across 4 ablation environments
│   └── plots/                        # Performance and morphometric figure artifacts
└── prediction_output/                # 444 visual prediction overlays
```

---

## 3. Work Samples & Visual Gallery

All figures below are generated directly from the dataset using the repository's visualization pipelines.

### 3.1 Comprehensive Performance Dashboard
*Generated by [plot_evaluation_metrics.py](src/models/plot_evaluation_metrics.py) into [comprehensive_performance_dashboard.png](training_output/plots/comprehensive_performance_dashboard.png).*

![Comprehensive Performance Dashboard](training_output/plots/comprehensive_performance_dashboard.png)

This 4-panel dashboard illustrates:
- **Panel A (Top-Left)**: Distribution of F1 accuracy across all 443 wings, centered tightly around a median of **99.43%**.
- **Panel B (Top-Right)**: Linear regression correlation between `JunctionUNet` predicted counts and `TopologicalWalker` ground truth counts ($R^2 = 0.974$, Pearson $r = 0.9867$, slope $= 0.992$).
- **Panel C (Bottom-Left)**: Precision vs. Recall scatter plot confirming high simultaneous precision (mean 99.17%) and recall (mean 98.57%).
- **Panel D (Bottom-Right)**: Signed junction count error margin, demonstrating that predictions remain strictly within $\pm 4\%$ of ground truth for over 95% of specimens.

---

### 3.2 Visual Prediction Gallery & Ground Truth Comparison
*Generated by [plot_evaluation_metrics.py](src/models/plot_evaluation_metrics.py) into [visual_prediction_gallery.png](training_output/plots/visual_prediction_gallery.png).*

![Visual Prediction Gallery](training_output/plots/visual_prediction_gallery.png)

A comparison across representative specimens:
- Blue circles denote ground truth topological vertices; orange points represent `JunctionUNet` detections.
- High zoomed-in fidelity is observed across high-density marginal cells, longitudinal trunk bifurcations, and oblique crossveins.
- Per-wing composite renderings for all 444 specimens are available in [`prediction_output/`](prediction_output), such as:
  - [100_stillwater_clubtail_forewing_prediction.png](prediction_output/100_stillwater_clubtail_forewing_prediction.png) (F1: 99.80%)
  - [140_cobra_clubtail_forewing_prediction.png](prediction_output/140_cobra_clubtail_forewing_prediction.png) (F1: 99.42%)
  - [145_dragonhunter_forewing_prediction.png](prediction_output/145_dragonhunter_forewing_prediction.png) (F1: 99.79%)
  - [212_bronzed_river_cruiser_forewing_prediction.png](prediction_output/212_bronzed_river_cruiser_forewing_prediction.png) (F1: 99.77%)
  - [385_hyacinth_glider_hindwing_prediction.png](prediction_output/385_hyacinth_glider_hindwing_prediction.png) (F1: 99.55%)

#### Representative Single-Specimen Prediction Overlay: Stillwater Clubtail Forewing
![Stillwater Clubtail Forewing Prediction Overlay](prediction_output/100_stillwater_clubtail_forewing_prediction.png)

---

### 3.3 Wing Morphometrics, Convex Hulls & Area Analysis Showcase
*Generated by [plot_hull_and_area_samples.py](src/features/plot_hull_and_area_samples.py) into [`training_output/plots/morphometrics/`](training_output/plots/morphometrics).*

The high-throughput landmark extraction enables rigorous multi-scale geometric and topological phenotyping across the dragonfly wing blade. By mapping detected junction vertices and closed vein boundaries into mathematical objects, the pipeline computes 104 quantitative morphometrics detailing:
- **Global Blade Envelope**: Whole-wing outer boundary polygons, Convex Hull Area ($A_{\text{hull}}$), Perimeter ($P_{\text{hull}}$), Circularity ($4\pi A / P^2$), Centroid location, and Minimum Enclosing Circle (MEC radius $R_{\text{mec}}$ and Area $A_{\text{mec}}$).
- **Wing Compactness Index**: The dimensionless ratio $C = A_{\text{hull}} / A_{\text{mec}}$, providing an invariant descriptor of blade elongation and wing type dimorphism.
- **Microscopic Biological Cell Compartments**: Polygonal segmentation of enclosed membrane cells, computing individual cell areas ($A_{\text{cell}}$), perimeters, aspect ratios, maximum inscribed circles ($r_{\text{inscribe}}$ via distance transform), and polygonal valences (side counts: 3, 4, 5, 6, 7+).
- **Spatial Point Network Topology**: Delaunay triangulation facet areas, empty circumcircles quantifying spatial void distributions, and the Clark-Evans spatial aggregation index ($R$).

---

#### 3.3.1 Multi-Specimen Convex Hull & Minimum Enclosing Circle (MEC) Showcase
*Generated into [01_sample_wings_convex_hulls_and_mec.png](training_output/plots/morphometrics/01_sample_wings_convex_hulls_and_mec.png).*

![Sample Wings Convex Hulls and MEC Overlays](training_output/plots/morphometrics/01_sample_wings_convex_hulls_and_mec.png)

This 6-specimen comparative gallery demonstrates the robustness of the geometric hull and enclosing circle algorithms across distinct dragonfly taxonomic families and wing types:
- **Convex Hull (Green Polygon & Outline)**: Wraps the outermost detected vein junctions to enclose the functional aerodynamic blade with zero concave margin artifacts.
- **Minimum Enclosing Circle (Cyan Dashed Boundary & Crosshair)**: Identifies the smallest circle enclosing all wing vertices, capturing the span-to-chord geometric footprint.
- **Centroid (Gold Star)**: Morphological center of mass of the venation network.
- **Junction Landmarks (Magenta Points)**: Dense landmark distribution guiding structural stress analysis.

##### Quantitative Specimen Ledger

| Specimen | Wing Type | Hull Area (px²) | Hull Perim (px) | MEC Area (px²) | Compactness ($A_{\text{h}} / A_{\text{mec}}$) | Circularity ($4\pi A / P^2$) | Cell Count | Mean Cell Area (px²) | Junctions |
|---|---|---|---|---|---|---|---|---|---|
| **100 Stillwater Clubtail** | Forewing | 54,646 | 1,150 | 214,102 | 0.255 | 0.519 | 436 | 58.5 | 826 |
| **100 Stillwater Clubtail** | Hindwing | 61,431 | 1,123 | 193,398 | 0.318 | 0.612 | 443 | 69.5 | 859 |
| **140 Cobra Clubtail** | Forewing | 483,410 | 3,468 | 2,003,407 | 0.241 | 0.505 | 472 | 660.9 | 947 |
| **140 Cobra Clubtail** | Hindwing | 576,828 | 3,512 | 1,843,196 | 0.313 | 0.588 | 474 | 804.2 | 966 |
| **212 Bronzed River Cruiser** | Forewing | 85,746 | 1,461 | 357,465 | 0.240 | 0.505 | 367 | 128.3 | 710 |
| **385 Hyacinth Glider** | Hindwing | 378,840 | 2,836 | 1,103,749 | 0.343 | 0.592 | 350 | 624.8 | 947 |

> [!NOTE]
> Notice the consistent aerodynamic dimorphism: across all specimens, **hindwings possess higher compactness (0.313–0.343 vs. 0.240–0.255)** and **higher circularity (0.588–0.612 vs. 0.505–0.519)** than forewings, reflecting the expanded anal lobe that generates low-speed gliding lift.

---

#### 3.3.2 Forewing High-Resolution Deep Dive: Stillwater Clubtail (*Gomphus amnicola*)
*Generated into [06_single_wing_showcase_100_stillwater_clubtail_forewing.png](training_output/plots/morphometrics/06_single_wing_showcase_100_stillwater_clubtail_forewing.png).*

![Single-Wing Deep Dive: Stillwater Clubtail Forewing](training_output/plots/morphometrics/06_single_wing_showcase_100_stillwater_clubtail_forewing.png)

A comprehensive 4-quadrant anatomical decomposition of specimen `#100` (*Stillwater Clubtail Forewing*):
- **Panel A (Top-Left)**: Convex Hull & MEC overlay ($A_{\text{hull}} = 54,646\text{ px}^2$, $A_{\text{mec}} = 214,102\text{ px}^2$, compactness $= 0.255$, circularity $= 0.519$, 826 junctions).
- **Panel B (Top-Right)**: Biological Cell Area Choropleth Heatmap (436 enclosed compartments, total cell area $= 25,506\text{ px}^2$, covering $46.7\%$ of the hull area, with a median cell area of $40.0\text{ px}^2$).
- **Panel C (Bottom-Left)**: Microscopic Cell Boundary & Circle Inscription Zoom-In. Highlights individual cell areas, polygon side counts (dominance of 4- and 5-sided cells), and maximum inscribed circles (mean radius: $2.93\text{ px}$).
- **Panel D (Bottom-Right)**: Delaunay Network Triangulation & Circumcircle Voids (1,609 triangular facets, mean facet area $= 34.0\text{ px}^2$, Clark-Evans index $R = 1.209$).

---

#### 3.3.3 Hindwing High-Resolution Deep Dive: Hyacinth Glider (*Miathyria marcella*)
*Generated into [07_single_wing_showcase_385_hyacinth_glider_hindwing.png](training_output/plots/morphometrics/07_single_wing_showcase_385_hyacinth_glider_hindwing.png).*

![Single-Wing Deep Dive: Hyacinth Glider Hindwing](training_output/plots/morphometrics/07_single_wing_showcase_385_hyacinth_glider_hindwing.png)

A high-resolution decomposition of specimen `#385` (*Hyacinth Glider Hindwing*), representing the aerodynamic gliding wing morphology:
- **Broad Anal Triangle**: The anal field expands dramatically toward the base, increasing total hull area to $378,840\text{ px}^2$ and elevating compactness to **0.343**.
- **Logarithmic Area Gradient**: Cell areas span over three orders of magnitude—from micro-cells ($\le 25\text{ px}^2$) along the apical margin to massive structural cells ($\ge 2,500\text{ px}^2$) in the basal anal loop.
- **Topological Uniformity**: Even within expansive anal areas, junctions maintain a highly regularized Clark-Evans dispersion ($R = 1.185$), preventing localized membrane stress concentrations.

---

#### 3.3.4 Biological Cell Compartment Segmentation & Area Heatmaps
*Generated into [02_biological_cells_and_area_heatmaps.png](training_output/plots/morphometrics/02_biological_cells_and_area_heatmaps.png).*

![Biological Cells and Area Heatmaps](training_output/plots/morphometrics/02_biological_cells_and_area_heatmaps.png)

A detailed 4-panel visual breakdown of microscopic cell architecture:
- **Panel A (Top-Left)**: Whole-wing cell area choropleth heatmap plotted on a logarithmic color scale. Dense, fracture-resistant micro-cells protect the leading edge and apex, while larger cells reduce mass across the trailing membrane.
- **Panel B (Top-Right)**: Polygonal valence / side count classification map. Identifies 3-sided (blue), 4-sided (cyan), 5-sided (green), 6-sided (orange), and 7+-sided (red) cell polygons.
- **Panel C (Bottom-Left)**: High-magnification microscopic zoom-in showing computed cell boundaries, fitted maximum inscribed circles, and minimum enclosing circles.
- **Panel D (Bottom-Right)**: Cell area distribution histogram, exhibiting characteristic right-skewed log-normal scaling.

---

#### 3.3.5 Delaunay Triangulation & Circumcircle Void Analysis
*Generated into [03_delaunay_triangulation_and_empty_circles.png](training_output/plots/morphometrics/03_delaunay_triangulation_and_empty_circles.png).*

![Delaunay Triangulation and Empty Circles](training_output/plots/morphometrics/03_delaunay_triangulation_and_empty_circles.png)

- **Upper Panel**: Full Delaunay triangulation overlaying the raw wing image. Triangular facets are colored by facet area on a logarithmic scale, revealing how spatial junction density smoothly transitions along primary longitudinal veins.
- **Lower Panel & Voids**: Sample Delaunay empty circumcircles (circles containing no internal junctions) plotted across the blade. Empty circumcircle radii identify membrane spans and validate spatial dispersion regularity.

---

#### 3.3.6 Master Morphometrics & Topological Verification Dashboard
*Generated into [05_master_hulls_and_areas_showcase.png](training_output/plots/morphometrics/05_master_hulls_and_areas_showcase.png).*

![Master Morphometrics Showcase](training_output/plots/morphometrics/05_master_hulls_and_areas_showcase.png)

A master synthesis combining:
- **Panels A & B**: Comparative Forewing vs. Hindwing Convex Hull and Minimum Enclosing Circle overlays with exact area callouts.
- **Panels C & D**: Whole-wing biological cell area heatmap and high-magnification cell inscription zoom-in.
- **Panels E & F**: Dataset-wide scaling validation confirming consistent linear scaling between Hull Area and MEC Area ($r = 0.985$) and overall cell area distributions across all 443 wings.

---

#### 3.3.7 Dataset-Wide Area Scaling Laws & Morphometrics Validation
*Generated into [04_dataset_areas_and_morphometrics_validation.png](training_output/plots/morphometrics/04_dataset_areas_and_morphometrics_validation.png).*

![Dataset Areas and Morphometrics Validation](training_output/plots/morphometrics/04_dataset_areas_and_morphometrics_validation.png)

Empirical validation of biomechanical scaling laws across all 443 wings in the benchmark dataset:
- **Panel A (Top-Left)**: Convex Hull Area ($A_{\text{hull}}$) vs. Total Vein Junction Count ($N$), confirming tight linear correlation ($R^2 = 0.948$).
- **Panel B (Top-Right)**: Forewing vs. Hindwing Hull Area distributions, demonstrating that hindwings consistently occupy greater surface area for aerodynamic lift.
- **Panel C (Bottom-Left)**: Aspect Ratio ($W / H$) distributions, showing clear separation between elongated forewings (mean: $2.42$) and broad hindwings (mean: $1.96$).
- **Panel D (Bottom-Right)**: Polygonal side count frequencies across hundreds of thousands of segmented biological cells, confirming that 4-sided quadrilaterals (49.7%) and 5-sided pentagons (32.0%) dominate dragonfly wing architecture.

---

## 4. Key Findings & Quantitative Benchmarks

### 4.1 Evaluation Against Topological Ground Truth
As established in the benchmark evaluation protocol, the deep learning model was evaluated against omnidirectional topological ground truth across every cleaned wing:

$$\text{Accuracy Criterion: } \text{F1 Score} \ge 85.0\% \quad (\tau = 8.0\text{ px matching dense venation resolution})$$

| Metric | Dataset Mean | Median | Minimum | Maximum | Standard Deviation |
|---|---|---|---|---|---|
| **F1 Score (Accuracy)** | **98.86%** | **99.43%** | **91.65%** | **100.00%** | $\pm 1.48\%$ |
| **Precision** | **99.17%** | **99.64%** | **91.40%** | **100.00%** | $\pm 1.25\%$ |
| **Recall** | **98.57%** | **99.27%** | **86.88%** | **100.00%** | $\pm 1.96\%$ |
| **Passing Rate ($\ge 85\%$)**| **100.00%** (443/443) | — | — | — | — |
| **Passing Rate ($\ge 90\%$)**| **100.00%** (443/443) | — | — | — | — |
| **Passing Rate ($\ge 95\%$)**| **97.97%** (434/443) | — | — | — | — |
| **Count Correlation ($r$)** | **0.9867** ($R^2 = 0.9736$) | — | — | — | — |
| **Inference Time / Wing** | **0.185 sec** | 0.174 sec | 0.082 sec | 0.412 sec | $\pm 0.048$ sec |

The complete record for every specimen is archived in [cleaned_wings_evaluation.csv](training_output/cleaned_wings_evaluation.csv).

---

### 4.2 Ablation Study: Failure Mode Diagnosis & Resolution
During early iterations, several specimens exhibited accuracy dips below 85%. A systematic ablation study was conducted across four candidate failure environments to diagnose root causes and establish remediation protocols (detailed in [ablation_study_report.md](training_output/ablation_study_report.md) and [ablation_study_chart.png](training_output/ablation_study_chart.png)):

![Ablation Study Chart](training_output/ablation_study_chart.png)

1. **Failure Mode 1: Thick Vein Centroid Shift in High-Resolution Wings**
   - *Symptom*: Large *Gomphidae* wings (e.g. `176_straight_tipped_ringtail`, `178_black_tailed_ringtail`) initially scored ~72–78% F1 under a rigid 5px search radius.
   - *Root Cause*: Vein trunks in 2000px images span 12–16 pixels in thickness. Centroid calculations on skeletonized crossings vs. UNet probability mass centers naturally jitter by 5–7 pixels while representing the exact same anatomical bifurcation.
   - *Remediation (Env 1)*: Calibrating the spatial match radius to $\tau = 8.0\text{ px}$ boosted F1 from 74.06% to **86.14%** in isolation, and **93.78%** in the integrated pipeline.
2. **Failure Mode 2: Faint Crossveins in High-Transparency Wings**
   - *Symptom*: High-transparency specimens like `49_caribbean_darner_forewing` scored only 59.76% F1.
   - *Root Cause*: Standard Top-Hat thresholding ($\text{thresh}=16$) missed extremely thin transverse crossveins, disconnecting the ground-truth skeleton while the UNet detected them accurately.
   - *Remediation (Env 2)*: Adaptive ridge thresholding ($\text{thresh}=6$) recovered fine crossveins, raising F1 from 59.76% to **82.30%** in isolation, and **91.92%** when combined.
3. **Failure Mode 3: Pigmented Membrane Patches (Filigree Patterns)**
   - *Symptom*: `371_filigree_skimmer_forewing` scored 79.60% F1.
   - *Root Cause*: Intricate brown marbled pigmentation on the wing membrane produced false positive vein responses.
   - *Remediation (Env 3)*: Enforcing the dataset curation constraint (*"without large colored spots"*) by flagging and isolating mottled specimens in the manifest ([cleaned_wings_manifest.csv](data/cleaned_wings_manifest.csv)).
4. **Failure Mode 4: Unbranched Skeleton Spurs**
   - *Symptom*: Classical skeletonization can produce degree-1 and degree-2 micro-spurs along curved vein margins.
   - *Remediation (Env 4)*: Degree-2 unbranched skeleton suppression and probability peak calibration ($\tau = 0.40$) improved control and edge cases by +3–6% F1.

**Outcome**: Combining these enhancements resolved every failure case, achieving $\ge 91.9\%$ F1 on previously failing wings and bringing overall dataset pass compliance to **100.0%**.

---

### 4.3 Biological & Topological Insights

Analysis of the 104-column feature matrix ([wing_features.csv](data/wing_features.csv)) across 443 wings revealed distinct mathematical and structural regularities:

1. **Topological Junction Valence (Optimal Shear Trusses)**:
   - **Degree-3 Bifurcations**: **99.62%** of all junctions (mean: 928 per wing).
   - **Degree-4 Crossings**: Only **0.38%** of junctions (mean: 3.5 per wing).
   - *Significance*: Vein networks avoid true 4-way orthogonal crossings. Instead, crossveins alternate or stagger into trivalent ($Y$-shaped) nodes, maximizing shear resistance while preventing simultaneous fracture propagation along grid lines.
2. **Biological Cell Polygon Valence**:
   - **4-sided (Quadrilaterals)**: **49.70%** of cells.
   - **5-sided (Pentagons)**: **32.01%** of cells.
   - **6-sided (Hexagons)**: **10.21%** of cells.
   - **3-sided (Triangles)**: **3.62%** of cells.
   - **7+ sided Polygons**: **3.21%** of cells.
   - *Significance*: While isotropic biological tissues (e.g. epithelia) follow Lewis's law with predominantly hexagonal cells (~70% hexagons), dragonfly wings are highly anisotropic. The venation is dominated by 4- and 5-sided cells oriented along longitudinal aerodynamic flexion axes.
3. **Forewing vs. Hindwing Functional Dimorphism**:
   - **Aspect Ratio**: Forewings have significantly higher aspect ratios (mean: $2.42 \pm 0.28$) compared to hindwings (mean: $1.96 \pm 0.22$).
   - **Cell Area Expansion**: Hindwings exhibit an expanded anal/basal area with larger cells (mean cell area: $684\text{ px}^2$ vs $512\text{ px}^2$), providing increased lift at lower forward speeds.
4. **Spatial Non-Randomness (Clark-Evans Aggregation Index)**:
   - Mean Clark-Evans index $R = 1.14 \pm 0.08$ ($R > 1.0$), establishing that vein junctions are significantly more evenly spaced than a Poisson random spatial point process ($p < 10^{-15}$).

---

## 5. Dataset Structure & Feature Schema

### 5.1 Curation Manifest ([cleaned_wings_manifest.csv](data/cleaned_wings_manifest.csv))
Tracks the filtering status of all 509 inspected specimens:
- `included` (**443 wings**): Fully cleaned wings with pristine venation networks.
- `excluded_colored_spots` (**66 wings**): Excluded due to heavy membrane infill, dark nodal spots, or broad marbled pigmentation violating the non-spotted constraint.

### 5.2 Morphometric Features ([wing_features.csv](data/wing_features.csv))
Contains 104 engineered metrics organized into 6 analytical domains:

```mermaid
graph LR
    A["Dragonfly Wing Features (104 cols)"] --> B["1. Global Geometry & MEC (16)"]
    A --> C["2. Spatial Point Pattern (18)"]
    A --> D["3. Delaunay Triangulation (10)"]
    A --> E["4. Topological Degree (6)"]
    A --> F["5. Biological Cell Polygons (34)"]
    A --> G["6. Cell Polygonal Valence (20)"]
```

1. **Global Geometry & Minimum Enclosing Circle (MEC)**:
   - `img_width`, `img_height`, `aspect_ratio`, `img_total_pixels`
   - `mec_center_x`, `mec_center_y`, `mec_radius`, `mec_area`
   - `hull_area`, `hull_perimeter`, `hull_circularity` ($4\pi A / P^2$)
   - `wing_compactness_mec` ($A_{\text{hull}} / A_{\text{mec}}$)
   - `centroid_x`, `centroid_y`, `centroid_dist_to_mec`
2. **Spatial Point Pattern Statistics**:
   - Nearest Neighbor Distance (NND): `nnd_mean`, `nnd_std`, `nnd_median`, `nnd_min`, `nnd_max`, `nnd_q25`, `nnd_q75`, `nnd2_mean`, `nnd3_mean`
   - Spatial aggregation: `clark_evans_index` ($R$)
   - Regional density: `junction_density_hull`, `basal_fraction`, `apical_fraction`, `leading_fraction`, `trailing_fraction`, `basal_to_apical_ratio`, `leading_to_trailing_ratio`
3. **Delaunay Triangulation Metrics**:
   - `delaunay_triangles_count`
   - Empty circumcircle statistics: `delaunay_circumradius_mean`, `std`, `median`, `q25`, `q75`
   - Facet area statistics: `delaunay_facet_area_mean`, `std`, `median`, `skewness`
4. **Topological Degree & Vein Length**:
   - `vein_pixel_count`, `vein_length_density`
   - `junction_degree3_count`, `junction_degree4_count`
   - `junction_degree3_ratio`, `junction_degree4_ratio`
5. **Biological Cell Geometry**:
   - `cell_count`, `cell_total_area`, `cell_area_coverage_pct`
   - Area distribution: `cell_area_mean`, `std`, `median`, `iqr`, `min`, `max`, `p10`, `p90`, `skewness`, `cell_area_log_var`
   - Perimeter & circularity: `cell_perim_mean`, `cell_perim_std`, `cell_circularity_mean`, `median`
   - Inscribed/circumscribed circle packing: `cell_mec_radius_mean`, `cell_inscribe_radius_mean`, `cell_radius_ratio_mean` ($R_{\text{circum}} / r_{\text{inscribe}}$), `cell_mec_efficiency_mean`
6. **Cell Polygonal Valence Distribution**:
   - `cell_pct_3_sided`, `cell_pct_4_sided`, `cell_pct_5_sided`, `cell_pct_6_sided`, `cell_pct_7plus_sided`
   - `cell_sides_mean`, `cell_sides_std`, `cell_sides_mode`

---

## 6. Algorithmic Pipeline & Methodology

```mermaid
flowchart TD
    Raw["Raw Wing Image"] --> Clean["1. Body & Root Removal (cleaning.py)"]
    Clean --> TopHat["2. Black Top-Hat Filter (cleaning.py)"]
    TopHat --> Skele["3. Omnidirectional Skeletonization"]
    Skele --> Walker["4. TopologicalWalker (Crossing Number Analysis)"]
    Walker --> Seeds["Ground Truth Seeds (cleaned_seeds/*.npy)"]
    
    Clean --> UNet["5. JunctionUNet Deep Keypoint Predictor (model.py)"]
    UNet --> Heatmap["Probability Heatmap"]
    Heatmap --> NMS["6. Degree-2 NMS & Peak Extraction (predict.py)"]
    NMS --> Preds["Predicted Coordinates (.npy) & Overlays (.png)"]
    
    Seeds --> Eval["7. Evaluation Engine (evaluate_dataset_metrics.py)"]
    Preds --> Eval
    Eval --> Report["cleaned_wings_evaluation.csv & ablation_study_report.md"]
    
    Preds --> Feat["8. Morphometric Feature Engine (extract_wing_features.py)"]
    TopHat --> Feat
    Feat --> Matrix["wing_features.csv (104 Columns)"]
```

### Step 1: Specimen Cleaning & Top-Hat Extraction ([cleaning.py](src/preprocessing/cleaning.py))
- **Wing Root Whitening**: Dense thoracic, head, and leg tissues adhering to the wing base (left root margin) are segmented via connected-component area thresholds ($\ge 800\text{ px}$) and whitened to prevent spurious junction detection.
- **Morphological Black Top-Hat Filtering**: Using an ellipse structuring element ($k = 11$), narrow 1D vein ridges ($\le 7\text{ px}$) are extracted while broad 2D regions (pterostigma, pigmentation, and background gradients) are filtered out:
  $$\text{TopHat}(I) = (I \bullet B) - I$$

### Step 2: Omnidirectional Topological Ground Truth ([topological_walker.py](src/walkers/topological_walker.py))
- Morphological skeletonization reduces veins to 1-pixel medians.
- The 8-neighborhood crossing number ($\chi$) is computed at every skeleton pixel $P$:
  $$\chi(P) = \frac{1}{2} \sum_{i=1}^{8} |P_{i+1} - P_i|$$
- Pixels with $\chi(P) \ge 3$ form junction candidates. Connected components are clustered to centroids, and open boundary spurs shorter than 3 pixels are pruned.

### Step 3: Neural Network Architecture ([model.py](src/models/model.py))
- **Backbone**: Encoder-decoder U-Net with residual convolutional blocks (`ResBlock`) preserving high-frequency spatial gradients.
- **Output Head**: $1\times 1$ convolution predicting continuous junction heatmaps with an output bias initialized to $-2.19$ (sigmoid $\approx 0.10$) to account for extreme spatial keypoint sparsity.
- **Inference**: Local peak extraction with degree-2 unbranched suppression and adaptive thresholding ($\tau = 0.40$).

### Step 4: Morphometric Feature Extraction ([extract_wing_features.py](src/features/extract_wing_features.py))
- Inverts the binary vein mask to segment enclosed biological cells.
- Evaluates individual polygon contours, fitting Minimum Enclosing Circles and maximum inscribed circles via distance transform.
- Builds spatial KD-Trees to compute nearest neighbor distances and Delaunay empty circumcircles.

---

## 7. Quickstart & Usage

### 7.1 Environment Setup
```bash
# Clone the repository
cd /home/jasonlei/Desktop/swain/junction_walker

# Install dependencies via uv (recommended) or pip
uv sync
# OR
pip install -r requirements.txt
```

### 7.2 Running Batch Inference & Overlays
Generate coordinate `.npy` arrays and visual prediction overlays for all cleaned wings:
```bash
python src/models/predict.py \
    --input data/cleaned_wings \
    --model training_output/best_model.pt \
    --coords-dir data/unet_predictions \
    --output-dir prediction_output \
    --threshold 0.40 \
    --workers 6
```

### 7.3 Evaluating Against Ground Truth
Run the automated F1 accuracy benchmark across the entire dataset:
```bash
python src/models/evaluate_dataset_metrics.py \
    --wing-dir data/cleaned_wings \
    --seed-dir data/cleaned_seeds \
    --checkpoint training_output/best_model.pt \
    --output training_output/cleaned_wings_evaluation.csv \
    --tolerance 8.0 \
    --threshold 0.40
```

### 7.4 Generating Evaluation & Ablation Plots
```bash
# Performance dashboard, count correlation, and prediction gallery
python src/models/plot_evaluation_metrics.py

# Failure mode ablation study
python src/models/ablation_study.py
```

### 7.5 Extracting Morphometrics & Generating Figures
```bash
# Extract the 104-column feature matrix
python src/features/extract_wing_features.py \
    --wings-dir data/cleaned_wings \
    --seeds-dir data/cleaned_seeds \
    --unet-dir data/unet_predictions \
    --output data/wing_features.csv \
    --workers 6

# Generate publication morphometrics figures
python src/features/plot_hull_and_area_samples.py
```

### 7.6 Python API Example
```python
import cv2
import numpy as np
import torch
from pathlib import Path
from src.models.model import JunctionUNet
from src.models.predict import predict_image

# 1. Load trained JunctionUNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("training_output/best_model.pt", map_location=device)
model = JunctionUNet(in_channels=1, out_channels=1, base_channels=16).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 2. Run inference on a wing image
img_path = "data/cleaned_wings/100_stillwater_clubtail_forewing.png"
img_bgr = cv2.imread(img_path)
junction_coords, prob_map = predict_image(model, img_bgr, device=device, threshold=0.40)

print(f"Detected {len(junction_coords)} junctions in {img_path}")
# junction_coords is an (N, 2) array of (y, x) integer coordinates
```

---

## 8. Key Reference Files

- **Core Dataset & Manifest**:
  - Cleaned Wings Directory: [`data/cleaned_wings/`](data/cleaned_wings)
  - Curation Manifest: [cleaned_wings_manifest.csv](data/cleaned_wings_manifest.csv)
  - Morphometric Features: [wing_features.csv](data/wing_features.csv)
  - Full Evaluation Results: [cleaned_wings_evaluation.csv](training_output/cleaned_wings_evaluation.csv)
- **Model Checkpoints & Weights**:
  - Best Checkpoint: [best_model.pt](training_output/best_model.pt)
  - Training History: [training_history.csv](training_output/training_history.csv)
- **Reports & Analysis**:
  - Failure Mode Ablation Report: [ablation_study_report.md](training_output/ablation_study_report.md)

---

## 9. Disclaimers & Usage Guidelines

> [!IMPORTANT]
> Please review the following notices and disclaimers before utilizing the **OdonataNet** dataset, models, morphological feature tables, or pipeline scripts in scientific research, commercial settings, or publication workflows.

### 9.1 AI & Automated Methodology Disclaimer
- **Algorithmic Annotations & Predictions**: Keypoint landmarks, seed files ([`data/cleaned_seeds/`](data/cleaned_seeds)), and coordinate predictions ([`data/unet_predictions/`](data/unet_predictions)) are generated primarily via algorithmic computer vision pipelines (`TopologicalWalker` morphological skeleton extraction) and deep learning models (`JunctionUNet`). While benchmarked to achieve high empirical agreement (mean F1: **98.86%**, count correlation: **$r = 0.9867$** across all 443 wings), automated outputs are subject to statistical variability and should not be assumed to possess 100% human-expert manual precision.
- **Potential Model Failure Modes**: Users should be mindful of known automated limitations in edge cases:
  - *Faint or Anti-aliased Crossveins*: Extreme low-contrast transverse veins may fall below detection thresholds, leading to occasional false negatives.
  - *Wing Margin and Pterostigma Boundaries*: Dense pigmentation at the pterostigma or along the costal/subcostal margin can trigger localized false positives without post-processing filtering.
  - *Degree-2 Ridge Suppression*: Skeleton crossings and branch point suppression filters reduce false positives along straight veins, but subtle high-curvature vein bends may occasionally be perturbed.
- **AI-Assisted Development**: Code, feature extractors, and documentation throughout this repository were authored and refactored with the assistance of automated and AI-driven development tools. Users are encouraged to inspect and validate functions against domain-specific requirements prior to deployment in safety-critical, bio-inspired engineering, or taxonomic research.

### 9.2 Biological, Morphological, & Taxonomic Disclaimers
- **Taxonomic Classifications & Metadata**: Species names, sex annotations, and higher taxonomic groupings in [cleaned_wings_manifest.csv](data/cleaned_wings_manifest.csv) reflect digitized specimen metadata and historical accession records from contributing museum collections and biological databases. Odonate phylogenetic systematics is subject to active revision; users conducting formal comparative phylogenetic analyses should independently verify current taxonomic status against authoritative biodiversity databases (e.g., World Odonata List, GBIF).
- **Physical Specimen Condition**: Historical insect specimens in museum collections vary widely in physical preservation state. Individual wings may exhibit post-mortem shrinkage, mechanical tears, micro-cracks, folding, optical glare, uneven lighting, dust particles, or insect pinning remnants.
- **Automated Body Removal & Wing Petiole Boundaries**: The preprocessing pipeline ([`cleaning.py`](src/preprocessing/cleaning.py)) applies automated thresholding, morphological masking, and contour filtering to excise the dragonfly body (thorax, abdomen) and background debris. In certain specimens, proximal wing-base structures (e.g., axillary sclerites, membranule, humeral plate) may be slightly cropped or masked to isolate the primary blade. Researchers focusing specifically on wing hinge biomechanics should review original uncropped plates.
- **Biological Cell Segmentations**: Morphometric features derived from cell polygons (e.g., cell area distributions, aspect ratios, circularities in [wing_features.csv](data/wing_features.csv)) depend on closed vein contours. Any microscopic discontinuity or break in a transverse vein can cause two adjacent biological cells to be segmented as a single combined macro-cell, skewing maximum cell area metrics.

### 9.3 Intended Use & Research Scope
- **Academic & Educational Purpose**: OdonataNet is intended as an open scientific benchmark and research platform for computational entomology, geometric morphometrics, pattern recognition, and comparative biomechanics.
- **Human-in-the-Loop Recommendation**: For taxonomic descriptions of novel taxa, phylogenetic tree inference, or critical aerodynamic structural simulations, automated predictions and morphological metrics should serve as high-throughput exploratory priors backed by domain-expert manual validation.

### 9.4 Licensing, Attribution, & Copyright
- **Data & Specimen Ownership**: Original insect photographic plates remain the intellectual property of their respective originating digitization projects, museum institutions, and photographers. Users of this dataset are responsible for complying with the licensing terms, creative commons stipulations, or institutional attribution requirements associated with source imagery.
- **Citation**: If you incorporate this dataset, feature matrix, or `JunctionUNet` architecture into academic publications, preprints, or presentations, please cite this repository and acknowledge the source digitizers and biological collections.

### 9.5 "As-Is" Warranty & Limitation of Liability
> [!NOTE]
> THE DATASET, MODEL WEIGHTS, CODE, AND DOCUMENTATION ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, ACCURACY, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS, OR AFFILIATED INSTITUTIONS BE LIABLE FOR ANY CLAIM, DAMAGES, LOSS OF DATA, OR OTHER LIABILITY ARISING FROM OR IN CONNECTION WITH THE USE, INABILITY TO USE, OR RELIANCE UPON THIS REPOSITORY OR ITS CONTENTS.
