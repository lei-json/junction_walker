#!/usr/bin/env python3
"""
Physical Scale Calibration & Dimensionless Invariant Taxonomy.

Addresses Weakness 6 (Speculative Biology - Physical Scale):
1. Distinguishes Dimensionless Scale-Invariant Metrics (valid for uncalibrated comparison)
   from Scale-Dependent Metrics (pixel-dependent).
2. Derives estimated physical calibration factors (px/mm and μm/px) using reference
   biological forewing lengths from entomological monographs (Dennis Paulson,
   'Dragonflies and Damselflies of North America').
"""

import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
output_dir = repo_root / "training_output/empirical"
output_dir.mkdir(parents=True, exist_ok=True)

# Reference forewing lengths (mm) from Dennis Paulson (2009/2011) field guides
SPECIES_PHYSICAL_REFERENCE_MM = {
    "dragonhunter": {"fw_len_mm": 55.0, "family": "Gomphidae", "scientific": "Hagenius brevistylus"},
    "cobra_clubtail": {"fw_len_mm": 33.5, "family": "Gomphidae", "scientific": "Gomphurus vastus"},
    "stillwater_clubtail": {"fw_len_mm": 36.5, "family": "Gomphidae", "scientific": "Arigomphus lentulus"},
    "columbia_clubtail": {"fw_len_mm": 35.0, "family": "Gomphidae", "scientific": "Gomphurus lynnae"},
    "splendid_clubtail": {"fw_len_mm": 37.0, "family": "Gomphidae", "scientific": "Gomphurus externus"},
    "blackwater_clubtail": {"fw_len_mm": 38.0, "family": "Gomphidae", "scientific": "Gomphus dilatatus"},
    "bronzed_river_cruiser": {"fw_len_mm": 52.0, "family": "Macromiidae", "scientific": "Macromia taeniolata"},
    "common_sanddragon": {"fw_len_mm": 34.0, "family": "Gomphidae", "scientific": "Progomphus obscurus"},
    "green_striped_darner": {"fw_len_mm": 47.0, "family": "Aeshnidae", "scientific": "Aeshna verticalis"},
    "blue_eyed_darner": {"fw_len_mm": 48.0, "family": "Aeshnidae", "scientific": "Rhionaeschna multicolor"},
    "lake_darner": {"fw_len_mm": 51.0, "family": "Aeshnidae", "scientific": "Aeshna eremita"},
    "gray_petaltail": {"fw_len_mm": 52.0, "family": "Petaluridae", "scientific": "Tachopteryx thoreyi"},
    "blue_dasher": {"fw_len_mm": 28.0, "family": "Libellulidae", "scientific": "Pachydiplax longipennis"},
    "boreal_whiteface": {"fw_len_mm": 26.5, "family": "Libellulidae", "scientific": "Leucorrhinia borealis"},
    "evening_skimmer": {"fw_len_mm": 44.0, "family": "Libellulidae", "scientific": "Pseudoleon superbus"},
}


DIMENSIONLESS_INVARIANT_METRICS = [
    ("aspect_ratio", "Span-to-chord aspect ratio (W / H)"),
    ("hull_circularity", "Isoperimetric circularity quotient (4*pi*A / P^2)"),
    ("wing_compactness_mec", "Enclosure compactness ratio (A_hull / A_mec)"),
    ("clark_evans_index", "Clark-Evans spatial aggregation index (R)"),
    ("junction_degree3_ratio", "Trivalent Y-junction proportion (N_3 / N_tot)"),
    ("junction_degree4_ratio", "Quadrivalent X-junction proportion (N_4 / N_tot)"),
    ("cell_pct_3_sided", "Triangular cell frequency (% 3-sided)"),
    ("cell_pct_4_sided", "Quadrilateral cell frequency (% 4-sided)"),
    ("cell_pct_5_sided", "Pentagonal cell frequency (% 5-sided)"),
    ("cell_pct_6_sided", "Hexagonal cell frequency (% 6-sided)"),
    ("cell_pct_7plus_sided", "Heptagonal+ cell frequency (% 7+-sided)"),
    ("cell_area_log_var", "Log-transformed cell area variance (Var(ln A))"),
    ("cell_radius_ratio_mean", "Mean cell elongation quotient (R_mec / r_inscribe)"),
]

SCALE_DEPENDENT_PIXEL_METRICS = [
    ("hull_area", "px²", "Functional aerodynamic blade surface area"),
    ("hull_perimeter", "px", "Blade outer perimeter length"),
    ("mec_radius", "px", "Minimum Enclosing Circle radius"),
    ("mec_area", "px²", "Minimum Enclosing Circle area"),
    ("cell_area_mean", "px²", "Mean individual cell lumen area"),
    ("cell_perim_mean", "px", "Mean individual cell boundary perimeter"),
    ("vein_pixel_count", "px", "Total skeletonized vein network length"),
]


def run_physical_calibration(
    features_csv: str = "data/wing_features.csv",
    output_dir: str = "training_output/empirical",
):
    out_dir = repo_root / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(repo_root / features_csv)
    forewings = df[df["wing_type"] == "forewing"]

    calib_records = []

    for sp_key, ref in SPECIES_PHYSICAL_REFERENCE_MM.items():
        matched = forewings[forewings["species"].str.contains(sp_key, case=False, na=False)]
        if matched.empty:
            continue

        for _, row in matched.iterrows():
            span_px = float(row["junction_span_x"])
            ref_mm = ref["fw_len_mm"]
            px_per_mm = span_px / ref_mm
            um_per_px = (ref_mm / span_px) * 1000.0

            # Calibrate pixel areas to mm²
            calib_hull_area_mm2 = float(row["hull_area"]) / (px_per_mm ** 2)
            calib_cell_area_mean_mm2 = float(row["cell_area_mean"]) / (px_per_mm ** 2)

            calib_records.append({
                "stem": row["stem"],
                "species_common": sp_key,
                "scientific_name": ref["scientific"],
                "family": ref["family"],
                "ref_fw_length_mm": ref_mm,
                "span_px": span_px,
                "resolution_px_per_mm": round(px_per_mm, 2),
                "resolution_um_per_px": round(um_per_px, 2),
                "raw_hull_area_px2": int(row["hull_area"]),
                "calib_hull_area_mm2": round(calib_hull_area_mm2, 2),
                "calib_cell_area_mean_mm2": round(calib_cell_area_mean_mm2, 4),
            })

    calib_df = pd.DataFrame(calib_records)
    csv_path = out_dir / "physical_scale_calibration.csv"
    calib_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 95)
    print("           PHYSICAL SCALE CALIBRATION (ENTOMOLOGICAL GROUND TRUTH REFERENCE)")
    print("=" * 95)
    print(f"{'Species':22s} | {'Ref Length':10s} | {'Span px':8s} | {'Resolution':13s} | {'Scale':10s} | {'Calib Area':12s}")
    print("-" * 95)
    for _, r in calib_df.head(10).iterrows():
        print(f"{r['species_common']:22s} | {r['ref_fw_length_mm']:6.1f} mm   | {r['span_px']:6.0f} px | {r['resolution_px_per_mm']:5.1f} px/mm  | {r['resolution_um_per_px']:5.1f} μm/px | {r['calib_hull_area_mm2']:7.1f} mm²")

    print(f"\nResolution Variation across digitized plates:")
    print(f"  Min Resolution: {calib_df['resolution_px_per_mm'].min():.1f} px/mm ({calib_df['resolution_um_per_px'].max():.1f} μm/px)")
    print(f"  Max Resolution: {calib_df['resolution_px_per_mm'].max():.1f} px/mm ({calib_df['resolution_um_per_px'].min():.1f} μm/px)")
    print(f"  --> CONFIRMATION: Digitization resolution varies by {calib_df['resolution_px_per_mm'].max() / calib_df['resolution_px_per_mm'].min():.1f}x across specimens!")
    print(f"  --> Standardizing on Dimensionless Scale-Invariants resolves cross-specimen comparisons.")

    taxonomy_json = out_dir / "feature_scale_taxonomy.json"
    with open(taxonomy_json, "w") as f:
        json.dump({
            "dimensionless_invariants": DIMENSIONLESS_INVARIANT_METRICS,
            "scale_dependent_metrics": SCALE_DEPENDENT_PIXEL_METRICS,
            "summary": {
                "total_dimensionless_features": len(DIMENSIONLESS_INVARIANT_METRICS),
                "total_scale_dependent_features": len(SCALE_DEPENDENT_PIXEL_METRICS),
                "resolution_range_px_per_mm": [
                    float(calib_df["resolution_px_per_mm"].min()),
                    float(calib_df["resolution_px_per_mm"].max())
                ],
            }
        }, f, indent=2)

    print(f"Scale taxonomy archived to: {taxonomy_json}")


if __name__ == "__main__":
    run_physical_calibration()
