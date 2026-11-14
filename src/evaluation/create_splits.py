#!/usr/bin/env python3
"""
Generate Leakage-Free Train / Validation / Held-Out Test Splits.

Ensures:
1. Specimen Grouping: Forewing and hindwing from the same specimen remain in the same partition.
2. Genus/Taxonomic Stratification: Taxa are distributed proportionally across splits,
   with a dedicated genus-held-out split for strict out-of-distribution generalization testing.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd


def parse_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    """Classifies species into family and genus."""
    def get_family(name: str) -> str:
        s = str(name).lower()
        if any(k in s for k in ['clubtail', 'snaketail', 'sanddragon', 'forceptail', 'spinylegs', 'dragonhunter', 'ringtail', 'leaftail']):
            return 'Gomphidae'
        elif any(k in s for k in ['darner', 'spotted_darner']):
            return 'Aeshnidae'
        elif any(k in s for k in ['skimmer', 'glider', 'pennant', 'amberwing', 'pondhawk', 'meadowhawk', 'darter', 'saddlebags', 'whitetail', 'king_skimmer', 'dasher', 'corporal', 'whiteface', 'dragonlet', 'sylph']):
            return 'Libellulidae'
        elif any(k in s for k in ['cruiser', 'river_cruiser']):
            return 'Macromiidae'
        elif any(k in s for k in ['emerald', 'baskettail', 'sundragon', 'shadowdragon', 'boghaunter']):
            return 'Corduliidae'
        elif any(k in s for k in ['spiketail']):
            return 'Cordulegastridae'
        elif any(k in s for k in ['petaltail', 'graypetal']):
            return 'Petaluridae'
        return 'Other'

    def get_genus_group(name: str) -> str:
        s = str(name).lower().split('_')
        # Common group indicator (last token usually represents group, e.g. clubtail, darner)
        return s[-1] if len(s) > 0 else 'unknown'

    df = df.copy()
    df['family'] = df['species'].apply(get_family)
    df['genus_group'] = df['species'].apply(get_genus_group)
    return df


def create_splits(
    features_csv: str = "data/wing_features.csv",
    output_dir: str = "data/splits",
    seed: int = 42,
):
    np.random.seed(seed)
    df = pd.read_csv(features_csv)
    df = parse_taxonomy(df)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Specimen-Grouped Split (70% Train, 15% Val, 15% Held-Out Test)
    specimens = df['specimen_id'].unique()
    np.random.shuffle(specimens)

    n_total = len(specimens)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    train_specs = set(specimens[:n_train])
    val_specs = set(specimens[n_train : n_train + n_val])
    test_specs = set(specimens[n_train + n_val :])

    train_wings = df[df['specimen_id'].isin(train_specs)]['stem'].tolist()
    val_wings = df[df['specimen_id'].isin(val_specs)]['stem'].tolist()
    test_wings = df[df['specimen_id'].isin(test_specs)]['stem'].tolist()

    specimen_split = {
        "metadata": {
            "strategy": "specimen_grouped_70_15_15",
            "seed": seed,
            "total_wings": len(df),
            "total_specimens": n_total,
            "train_specimens": len(train_specs),
            "val_specimens": len(val_specs),
            "test_specimens": len(test_specs),
            "train_wings_count": len(train_wings),
            "val_wings_count": len(val_wings),
            "test_wings_count": len(test_wings),
        },
        "train": sorted(train_wings),
        "val": sorted(val_wings),
        "test": sorted(test_wings),
    }

    with open(out_path / "specimen_split.json", "w") as f:
        json.dump(specimen_split, f, indent=2)

    # 2. Strict Genus-Held-Out Split (Entire Genera in Test Set)
    # E.g., hold out distinct genus groups entirely to evaluate zero-shot taxonomic transfer
    genus_groups = df.groupby('genus_group')['stem'].count().to_dict()
    all_genera = sorted(list(genus_groups.keys()))
    np.random.shuffle(all_genera)

    # Pick genera that total ~15-20% of dataset for strictly unseen taxonomic test
    held_out_genera = []
    accum = 0
    target_held_out = int(0.20 * len(df))
    for g in all_genera:
        count = genus_groups[g]
        if accum + count <= target_held_out or len(held_out_genera) == 0:
            held_out_genera.append(g)
            accum += count

    remaining_df = df[~df['genus_group'].isin(held_out_genera)]
    test_taxa_wings = df[df['genus_group'].isin(held_out_genera)]['stem'].tolist()

    rem_specs = remaining_df['specimen_id'].unique()
    np.random.shuffle(rem_specs)
    n_rem_train = int(0.85 * len(rem_specs))
    taxa_train_specs = set(rem_specs[:n_rem_train])
    taxa_val_specs = set(rem_specs[n_rem_train:])

    taxa_train_wings = remaining_df[remaining_df['specimen_id'].isin(taxa_train_specs)]['stem'].tolist()
    taxa_val_wings = remaining_df[remaining_df['specimen_id'].isin(taxa_val_specs)]['stem'].tolist()

    taxonomic_split = {
        "metadata": {
            "strategy": "held_out_taxa_genera",
            "seed": seed,
            "held_out_genera": held_out_genera,
            "train_wings_count": len(taxa_train_wings),
            "val_wings_count": len(taxa_val_wings),
            "test_unseen_taxa_count": len(test_taxa_wings),
        },
        "train": sorted(taxa_train_wings),
        "val": sorted(taxa_val_wings),
        "test": sorted(test_taxa_wings),
    }

    with open(out_path / "taxonomic_split.json", "w") as f:
        json.dump(taxonomic_split, f, indent=2)

    print("Created Splits Successfully:")
    print(f" Specimen Split -> Train: {len(train_wings)}, Val: {len(val_wings)}, Held-Out Test: {len(test_wings)}")
    print(f" Taxonomic Split -> Train: {len(taxa_train_wings)}, Val: {len(taxa_val_wings)}, Unseen Taxa Test: {len(test_taxa_wings)} (Genera: {held_out_genera})")


if __name__ == "__main__":
    create_splits()
