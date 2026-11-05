# examples/brushnet layout

| Folder | What goes here |
|---|---|
| `inference/` | Current restoration pipeline (`generate.py`, `settings.json`, metrics). Start here. |
| `inference_experiments/` | Per-experiment inference scripts (`test_brushnet_*.py`) and their launchers (`run_*.py`, `run_*.sh`, postprocess). |
| `training/` | Training scripts, their training-only datasets/samplers, and training launchers. |
| `tests/` | Unit tests (`python -m unittest`). |
| `proposed/`, `original_res_exp/` | Research code for the proposed methods and original-resolution experiments. |
| `upstream/` | Original BrushNet scripts (demo app, evaluation, SD1.5/SDXL training). |
| `tools/` | One-off data/checkpoint utilities. |
| `legacy/` | Old scripts kept for reference; not maintained. |

Top-level modules are shared libraries imported by name from several folders
(`sfu_long_test_loader`, `sfu_temporal_clip_windows`, `vcmrs_codec_adapter`,
`shared_bg_noise_training`, `custom_dataset_v04`) — keep them here.

Scripts in a subfolder start with
`sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` so they import these
modules and the local `diffusers/` and `ip_adapter/` copies, not the pip-installed ones.
Add the same line to any new script that needs them.
