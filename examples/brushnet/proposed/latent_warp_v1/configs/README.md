# Latent warp training configs

`run1.yaml` defines the experiment. `run0.yaml` inherits it and changes only warp
conditioning dropout, output directory and tracker name. Relative `extends` paths
are resolved against the YAML file containing them. Use absolute model/data/output
paths. No inference configuration is included because this is the training entry point.

From the `latent_warp_v1` directory:

```bash
# Inspect required paths without launching training or querying GPUs.
CHECK_PATHS_ONLY=1 bash run_train_latent_warp_v1_run1.sh --config configs/run1.yaml

# Run after checking GPU availability.
CUDA_VISIBLE_DEVICES=0 bash run_train_latent_warp_v1_run1.sh --config configs/run1.yaml
CUDA_VISIBLE_DEVICES=0 bash run_train_latent_warp_v1_run1.sh --config configs/run0.yaml

# One-off overrides; use a separate output directory for every experiment.
CUDA_VISIBLE_DEVICES=0 bash run_train_latent_warp_v1_run1.sh \
  --config configs/run1.yaml --learning_rate 5e-6 --output_dir /absolute/path/to/new_run
```

Precedence: CLI arguments > existing launcher environment overrides > YAML >
legacy parser defaults. The `CONFIG` environment variable can select a YAML file;
`--config` takes precedence. `INITIAL_CHECKPOINT` overrides the three init paths
as a group. Runtime launcher controls remain `PRINT_ONLY`, `CHECK_PATHS_ONLY`,
`PYTHON_BIN`, `ACCELERATE_BIN`, `CUDA_VISIBLE_DEVICES`, and `ALLOW_SHARED_GPU`.
`PYTHON_BIN` must point to an environment that can import the trainer and PyYAML.
The YAML `runtime` section is consumed by the shell launcher; use the launcher
when selecting GPUs or checking that GPUs are free.

The shell launcher and trainer share one resolver. Unknown settings, invalid
choices and invalid value types fail early. CLI flags that only enable a boolean
cannot disable a YAML `true`; edit the YAML (or use an inherited config) to do so.

`diffusion.expected_*` asserts the schedule loaded from the pretrained model;
it does not alter betas or the prediction target. A mismatch stops training.
Only uniform timestep sampling is currently implemented. `lr_warmup_steps` is
inactive with `lr_scheduler: constant`. Gradient clipping is off to preserve the
previous behavior; enable `enable_gradient_clipping` to activate `max_grad_norm`.

Once initialization succeeds, `output_dir/resolved_config.yaml` records all
parsed arguments (excluding the hub token), the loaded diffusion schedule, actual
precision, process count and effective batch. If `scale_lr` is enabled, the saved
learning rate is the scaled value. This file is a run record, not an input YAML.

The configured init checkpoint is `experiments/latent_warp/init_ckpt`. Its
14-channel BrushNet, IP-Adapter and fusion weights passed CPU loading checks.
GPU access and a GPU training step have not been verified in this session.
