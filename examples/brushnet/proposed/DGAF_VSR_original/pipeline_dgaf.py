"""Alternating, sequential, same-timestep DGAF sampling for BrushNet."""
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
        if warper.config.direction == "bidirectional":
            raise ValueError("Simultaneous bidirectional guidance is not a sequential DGAF sweep")
        if not hasattr(self, "baseline_brushnet"):
            raise ValueError("A frozen baseline BrushNet is required for sweep boundaries")
        use_cfg = guidance_scale > 1.0
        dtype = self.unet.dtype
        latents = latents.float().clone()
        def select(context, frame):
            return context.reshape(b, t, *context.shape[1:])[:, frame]
        for index, timestep in enumerate(self.progress_bar(self.scheduler.timesteps)):
            reverse = (index % 2 == 1) if warper.config.direction == "alternating" else warper.config.direction == "next"
            order = range(t-1, -1, -1) if reverse else range(t)
            clean = None
            source = None
            for frame in order:
                positions = torch.arange(b, device=latents.device) * t + frame
                state = latents[positions]
                guidance = warper.frame_guidance(clean, frame, source) * temporal_guidance_scale
                baseline = clean is None or temporal_guidance_scale == 0
                condition = base_condition[positions].float()
                if not baseline:
                    condition = torch.cat((condition, guidance), dim=1)
                condition = condition.to(dtype)
                model_input = self.scheduler.scale_model_input(state, timestep).to(dtype)
                context = select(prompt_embeds, frame)
                brush_context = select(brushnet_prompt_embeds, frame)
                if use_cfg:
                    model_input = torch.cat((model_input, model_input))
                    condition = torch.cat((condition, condition))
                    context = torch.cat((select(negative_prompt_embeds, frame), context))
                    brush_context = torch.cat((select(negative_brushnet_prompt_embeds, frame), brush_context))
                prediction = frozen_v8_predict(
                    self.baseline_brushnet if baseline else self.brushnet, self.unet,
                    model_input, timestep, condition, brush_context, context).float()
                if use_cfg:
                    unconditional, conditional = prediction.chunk(2)
                    prediction = unconditional + guidance_scale * (conditional - unconditional)
                result = self.scheduler.step(prediction, timestep, state, eta=0.0, return_dict=True)
                clean = result.pred_original_sample.detach()
                latents[positions] = result.prev_sample
                source = frame
                if not torch.isfinite(result.prev_sample).all() or not torch.isfinite(clean).all():
                    raise FloatingPointError(f"Nonfinite state at step {index}, frame {frame}")
        if output_type == "latent":
            return StableDiffusionPipelineOutput(images=latents, nsfw_content_detected=None)
        decoded = self.vae.decode(latents.to(self.vae.dtype) / self.vae.config.scaling_factor, return_dict=False)[0]
        images = self.image_processor.postprocess(decoded, output_type=output_type)
        return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=None)
