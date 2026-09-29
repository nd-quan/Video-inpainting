"""New DDIM pipeline with explicit, per-call predicted-clean latent memory.

This is a synchronous DGAF-inspired adaptation to SD1.5 BrushNet, not a
reproduction of the paper's complete SR model. Existing pipelines are intact.
"""
import math
import torch
from diffusers import DDIMScheduler
from diffusers.pipelines.brushnet.pipeline_sharedNoiseBG_org import StableDiffusionBrushNetPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from STC_encoder_v2_rgb.frozen_v8 import frozen_v8_predict


class DGAFBrushNetPipeline(StableDiffusionBrushNetPipeline):
    @torch.no_grad()
    def __call__(self, *, latents, base_condition, warper, prompt_embeds,
                 negative_prompt_embeds, brushnet_prompt_embeds,
                 negative_brushnet_prompt_embeds, num_inference_steps=50,
                 guidance_scale=7.5, temporal_guidance_scale=1.0, output_type="pil"):
        if not isinstance(self.scheduler, DDIMScheduler):
            raise ValueError("DGAF experiments require DDIM")
        if self.scheduler.config.clip_sample or self.scheduler.config.thresholding:
            raise ValueError("Use DDIM clip_sample=False, thresholding=False for latent guidance")
        if base_condition.shape != (latents.shape[0], 5, *latents.shape[-2:]):
            raise ValueError("base_condition must be [B*T,5,h,w]")
        if any(not math.isfinite(value) or value < 0 for value in (temporal_guidance_scale, guidance_scale)):
            raise ValueError("Guidance scales must be finite and nonnegative")
        b, t = warper.bg.shape[:2]
        if b * t != latents.shape[0]:
            raise ValueError("Flow clip layout and latent batch do not match")
        self.scheduler.set_timesteps(num_inference_steps, device=latents.device)
        # All frames read a snapshot from the previous step; never share state
        # between clips, CFG halves, or separate calls to this pipeline.
        cached_clean = None
        use_cfg = guidance_scale > 1.0
        context = torch.cat((negative_prompt_embeds, prompt_embeds)) if use_cfg else prompt_embeds
        brush_context = torch.cat((negative_brushnet_prompt_embeds, brushnet_prompt_embeds)) if use_cfg else brushnet_prompt_embeds
        dtype = self.unet.dtype
        for index, timestep in enumerate(self.progress_bar(self.scheduler.timesteps)):
            guidance = warper.guidance(cached_clean, index).flatten(0, 1) * temporal_guidance_scale
            condition = torch.cat((base_condition.float(), guidance), dim=1).to(dtype)
            model_input = self.scheduler.scale_model_input(latents, timestep).to(dtype)
            if use_cfg:
                model_input = torch.cat((model_input, model_input))
                condition = torch.cat((condition, condition))
            prediction = frozen_v8_predict(self.brushnet, self.unet, model_input, timestep,
                                           condition, brush_context, context).float()
            if use_cfg:
                unconditional, conditional = prediction.chunk(2)
                prediction = unconditional + guidance_scale * (conditional - unconditional)
            result = self.scheduler.step(prediction, timestep, latents.float(), eta=0.0, return_dict=True)
            cached_clean = result.pred_original_sample.detach().reshape(b, t, 4, *latents.shape[-2:])
            latents = result.prev_sample
            if not torch.isfinite(latents).all() or not torch.isfinite(cached_clean).all():
                raise FloatingPointError(f"Nonfinite diffusion state at step {index}")
        if output_type == "latent":
            return StableDiffusionPipelineOutput(images=latents, nsfw_content_detected=None)
        decoded = self.vae.decode(latents.to(self.vae.dtype) / self.vae.config.scaling_factor, return_dict=False)[0]
        images = self.image_processor.postprocess(decoded, output_type=output_type)
        return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=None)
