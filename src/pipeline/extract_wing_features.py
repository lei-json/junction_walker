#!/usr/bin/env python3
"""
Wrapper forwarding to src.features.extract_wing_features.
"""
import sys
from pathlib import Path

# Add project root to sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root / "src/features") not in sys.path:
    sys.path.insert(0, str(repo_root / "src/features"))

from extract_wing_features import run_extraction_pipeline, parse_args

if __name__ == "__main__":
    args = parse_args()
    run_extraction_pipeline(
        wings_dir=args.wings_dir,
        seeds_dir=args.seeds_dir,
        unet_dir=args.unet_dir,
        manifest_path=args.manifest,
        output_csv=args.output,
        workers=args.workers,
        limit=args.limit,
    )
