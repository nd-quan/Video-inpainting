# latent_warp_DDIM_inversion_v1

Fine-tuning starts from checkpoint 6750 with a fresh optimizer. A separate
frozen copy of checkpoint 6750 inverts the decoded input latent, using null
text, IP-Adapter, fusion, and zero warp channels. Each training clip samples
one position on the reverse DDIM grid. The teacher computes only the prefix
needed to reach that timestep. The student receives that latent and trains
against the epsilon target derived from the clean GT latent. The student
retains the original two-pass stable BG latent warp and GT flow conditioning.

This first implementation inverts online. It requires extra VRAM for the frozen
teacher and additional forward passes per training step. Measure VRAM and
epsilon-target magnitudes, particularly at low timesteps, before a full run.
The matching future inference path should reuse the frozen teacher and invert
to the final grid position, then use the fine-tuned student for denoising.

Run the input preflight with:

```bash
CHECK_PATHS_ONLY=1 bash run_train.sh
```

For a first GPU check, pass `--max_train_steps 1 --output_dir /path/to/smoke`
to the launcher in a scheduler job. `INVERSION_TEACHER` selects the frozen
teacher checkpoint and `INVERSION_STEPS` selects grid size (default 50).
The independent pipeline and scheduler modules live in the local Diffusers
tree under the versioned names used in `train_ddim_inversion.py`.
