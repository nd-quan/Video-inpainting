#!/usr/bin/env python
"""BG-only B/C diagnostic. ROI remains current z0; generated images unchanged."""
import os
import evaluate_v8_latent_warp_analysis as base

if __name__ == "__main__":
    base.REQUIRED_GPU = os.environ.get("LATENT_ANALYSIS_GPU_ID", "5")
    base.BG_ONLY = True
    base.main()
