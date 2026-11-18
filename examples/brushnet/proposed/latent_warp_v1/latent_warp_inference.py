"""Training-matched 14-input denoising with oracle warp and latent temporal DDIM.

No codec import: CGE is intentionally unsupported until a codec is installed.
All adjacent frames in a clip share the same diffusion timestep. GT is never
encoded as model conditioning; it is used only by the separate flow producer.
"""

import torch
import torch.nn.functional as F

from latent_warp_utils import build_latent_warp_condition, predict_x0
from diffusers.schedulers.scheduling_ddim_temporal import build_stable_bg_mask


@torch.no_grad()
def denoise_clip(
    predict_noise,
    scheduler,
    latents,
    base_condition,
    bg_masks,
    flow_backward,
    valid_backward,
    config,
):
    """predict_noise(z, t, condition) returns the final (optionally CFG) noise."""
    if base_condition.shape[1] != 5:
        raise ValueError("Expected decoded-input latent (4) + BG mask (1)")
    frames = latents.shape[0]
    if frames < 2:
        raise ValueError("Oracle inference requires at least two consecutive frames")
    scheduler.set_timesteps(config["num_inference_steps"], device=latents.device)
    temporal_enabled = (
        config.get("temporal_guidance_enabled", True)
        and config["temporal_guidance_scale"] > 0
    )
    stable = build_stable_bg_mask(bg_masks, flow_backward, visibility=valid_backward)
    scheduler.set_temporal_guidance(
        flow_backward=flow_backward,
        stable_bg=stable,
        bg_masks=bg_masks,
        guidance_scale=config["temporal_guidance_scale"],
        start_step=config["temporal_start"],
        end_step=config["temporal_end"],
        every_n_steps=config["temporal_every_n_steps"],
        guidance_space="latent",
        loss_type="l2",
        normalize_grad=True,
        detach_previous=False,
        enabled=temporal_enabled,
    )
    zero_warp = torch.zeros(
        (frames, 5, *latents.shape[-2:]),
        device=latents.device,
        dtype=base_condition.dtype,
    )
    reports = []
    for index, timestep in enumerate(scheduler.timesteps):
        model_input = scheduler.scale_model_input(latents, timestep)
        if config["latent_warp_enabled"]:
            # Same two-pass procedure as the training script; no ground-truth x0.
            base_noise = predict_noise(
                model_input, timestep, torch.cat([base_condition, zero_warp], 1)
            )
            x0 = predict_x0(latents, base_noise, timestep.expand(frames), scheduler)
            warped, fraction, stats = build_latent_warp_condition(
                x0,
                bg_masks,
                flow_backward.unsqueeze(0),
                valid_backward.unsqueeze(0),
                1,
                frames,
                x0_clamp=config["latent_warp_x0_clamp"] or None,
            )
            extra = torch.cat([warped, fraction], 1).to(base_condition.dtype)
        else:
            extra, stats = zero_warp, {}
        noise = predict_noise(
            model_input, timestep, torch.cat([base_condition, extra], 1)
        )
        # The existing scheduler differentiates predicted x0 only, not the models.
        latents = scheduler.step(noise, timestep, latents, eta=0.0).prev_sample
        if not torch.isfinite(latents).all():
            raise FloatingPointError(
                "Non-finite denoised latents at step {}".format(index)
            )
        requested = (
            temporal_enabled
            and config["temporal_start"] <= index < config["temporal_end"]
            and index % config["temporal_every_n_steps"] == 0
        )
        reports.append(
            dict(
                step_index=index,
                timestep=int(timestep),
                warp=stats,
                temporal_requested=requested,
                temporal_loss=scheduler.last_temporal_loss,
                temporal_update_norm=scheduler.last_temporal_update_norm,
                temporal_skipped_reason=scheduler.last_temporal_skipped_reason,
            )
        )
    return latents, dict(
        steps=reports,
        temporal_calls=scheduler.temporal_guidance_calls,
        temporal_applied=scheduler.temporal_guidance_applied_steps,
        temporal_skipped=scheduler.temporal_guidance_skipped_steps,
        stable_bg_fraction=float(stable.mean()),
        temporal_guidance_enabled=temporal_enabled,
        latent_warp_enabled=config["latent_warp_enabled"],
    )


class InferenceModels:
    """Load the three checkpoint components with strict adapter key matching."""

    def __init__(self, config, device, *, image_encoder=None):
        from pathlib import Path
        from diffusers import AutoencoderKL, BrushNetModel, UNet2DConditionModel
        from transformers import (
            AutoTokenizer,
            CLIPTextModel,
            CLIPVisionModelWithProjection,
            CLIPImageProcessor,
        )
        from safetensors.torch import load_file
        from ip_adapter.ip_adapter import ImageProjModel, FGBGFeatureFusion
        from ip_adapter.attention_processor import IPAttnProcessor2_0, AttnProcessor2_0

        self.config, self.device = config, torch.device(device)
        self.dtype = (
            torch.float16
            if config["mixed_precision"] == "fp16" and self.device.type == "cuda"
            else torch.float32
        )
        base, ckpt = config["pretrained_model"], Path(config["checkpoint_dir"])
        self.vae = AutoencoderKL.from_pretrained(
            base, subfolder="vae", torch_dtype=self.dtype
        )
        self.vae.enable_slicing()
        self.unet = UNet2DConditionModel.from_pretrained(
            base, subfolder="unet", torch_dtype=self.dtype
        )
        self.brushnet = BrushNetModel.from_pretrained(
            str(ckpt / "brushnet"), torch_dtype=self.dtype
        )
        if self.brushnet.config.conditioning_channels != 10:
            raise ValueError(
                "Checkpoint must have conditioning_channels=10 (14 total inputs)"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            base, subfolder="tokenizer", use_fast=False
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            base, subfolder="text_encoder", torch_dtype=self.dtype
        )
        self.image_encoder = image_encoder if image_encoder is not None else CLIPVisionModelWithProjection.from_pretrained(
            config["image_encoder"],
            cache_dir=config["cache_dir"],
            torch_dtype=self.dtype,
        )
        self.processor = CLIPImageProcessor()
        self.projection = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=4,
        )
        self.fusion = FGBGFeatureFusion(
            self.image_encoder.config.projection_dim, num_heads=8
        )
        processors = {}
        for name in self.unet.attn_processors:
            if name.endswith("attn1.processor"):
                processors[name] = AttnProcessor2_0()
                continue
            if name.startswith("mid_block"):
                hidden = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                hidden = list(reversed(self.unet.config.block_out_channels))[
                    int(name.split(".")[1])
                ]
            else:
                hidden = self.unet.config.block_out_channels[int(name.split(".")[1])]
            processors[name] = IPAttnProcessor2_0(
                hidden_size=hidden,
                cross_attention_dim=self.unet.config.cross_attention_dim,
            )
        self.unet.set_attn_processor(processors)
        state = load_file(str(ckpt / "ipadapter/model.safetensors"), device="cpu")
        consumed = set()
        prefix = "image_proj_model."
        self.projection.load_state_dict(
            {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)},
            strict=True,
        )
        consumed.update(k for k in state if k.startswith(prefix))
        for name, processor in self.unet.attn_processors.items():
            if name.endswith("attn2.processor"):
                prefix = "unet." + name + "."
                processor.load_state_dict(
                    {
                        k[len(prefix) :]: v
                        for k, v in state.items()
                        if k.startswith(prefix)
                    },
                    strict=True,
                )
                consumed.update(k for k in state if k.startswith(prefix))
        if consumed != set(state):
            raise ValueError(
                "Unrecognized adapter weights: " + str(sorted(set(state) - consumed))
            )
        self.fusion.load_state_dict(
            load_file(str(ckpt / "ipadapter/fusion_module.safetensors")), strict=True
        )
        for model in (
            self.vae,
            self.unet,
            self.brushnet,
            self.text_encoder,
            self.image_encoder,
            self.projection,
            self.fusion,
        ):
            model.requires_grad_(False).eval().to(device=self.device, dtype=self.dtype)
        self.batch_size = config["model_batch_size"]

    @torch.no_grad()
    def prepare(self, images, masks, input_pil, mask_pil, generator):
        import numpy as np
        from PIL import Image

        # Preserve the training loader's historical reversed FG/BG naming.
        historical_fg, historical_bg = [], []
        for image, mask in zip(input_pil, mask_pil):
            array = np.asarray(image)
            roi = (np.asarray(mask) >= 128)[..., None]
            historical_fg.append(
                Image.fromarray(np.where(roi, 0, array).astype("uint8"))
            )
            historical_bg.append(
                Image.fromarray(np.where(roi, array, 0).astype("uint8"))
            )

        def embed(pil):
            pixels = self.processor(images=pil, return_tensors="pt").pixel_values
            return torch.cat(
                [
                    self.image_encoder(chunk.to(self.device, self.dtype)).image_embeds
                    for chunk in pixels.split(self.batch_size)
                ]
            )

        embeds = embed(input_pil) + self.config["fusion_scale"] * self.fusion(
            embed(historical_fg), embed(historical_bg)
        )
        self.tokens = self.projection(embeds)
        self.uncond_tokens = self.projection(torch.zeros_like(embeds))
        ids = self.tokenizer(
            [""] * len(images),
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        self.text = self.text_encoder(ids)[0]
        encoded = torch.cat(
            [
                self.vae.encode(chunk).latent_dist.sample(generator=generator)
                for chunk in images.to(self.device, self.dtype).split(self.batch_size)
            ]
        )
        encoded *= self.vae.config.scaling_factor
        mask = F.interpolate(
            masks.to(self.device, self.dtype), size=encoded.shape[-2:], mode="nearest"
        )
        return torch.cat([encoded, mask], 1)

    @torch.no_grad()
    def predict_noise(self, z, timestep, condition):
        predictions = []
        for start in range(0, len(z), self.batch_size):
            end = start + self.batch_size
            sample, cond, text = (
                z[start:end],
                condition[start:end],
                self.text[start:end],
            )
            down, mid, up = self.brushnet(
                sample,
                timestep,
                encoder_hidden_states=text,
                brushnet_cond=cond,
                conditioning_scale=1.0,
                return_dict=False,
            )

            def predict(tokens):
                return self.unet(
                    sample,
                    timestep,
                    encoder_hidden_states=torch.cat([text, tokens[start:end]], 1),
                    down_block_add_samples=down,
                    mid_block_add_sample=mid,
                    up_block_add_samples=up,
                    return_dict=False,
                )[0]

            result = predict(self.tokens)
            if self.config["guidance_scale"] > 1:
                uncond = predict(self.uncond_tokens)
                result = uncond + self.config["guidance_scale"] * (result - uncond)
            predictions.append(result)
        return torch.cat(predictions)

    @torch.no_grad()
    def decode(self, z):
        return (
            torch.cat(
                [
                    self.vae.decode(
                        chunk / self.vae.config.scaling_factor
                    ).sample.float()
                    for chunk in z.split(self.batch_size)
                ]
            )
            .div(2)
            .add(0.5)
            .clamp(0, 1)
        )
