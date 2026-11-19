"""Fine-tune with online, frozen-teacher DDIM inversion latents."""

import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "latent_warp_v1"))

from train_brushnet_latent_warp_v1 import main, parse_args

import torch
from latent_warp_inference import InferenceModels
from diffusers.pipelines.brushnet.pipeline_brushnet_latent_warp_ddim_inversion_v1 import invert_decoded_clip
from diffusers.schedulers.scheduling_ddim_inversion_latent_warp_v1 import (
    LatentWarpDDIMInversionV1,
    epsilon_target,
)


class InversionTrajectory:
    def __init__(self, args, device, training_scheduler, image_encoder):
        self.training_scheduler = training_scheduler
        self.steps = args.inversion_steps
        self.full_grid_smoke = os.environ.get("INVERSION_FULL_GRID_SMOKE") == "1"
        self.scheduler = LatentWarpDDIMInversionV1.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        torch.testing.assert_close(self.scheduler.alphas_cumprod, training_scheduler.alphas_cumprod)
        self.teacher = InferenceModels(
            dict(
                pretrained_model=args.pretrained_model_name_or_path,
                checkpoint_dir=args.inversion_teacher,
                image_encoder=args.image_encoder_name_or_path,
                cache_dir=args.cache_dir,
                mixed_precision=args.mixed_precision,
                model_batch_size=1,
                fusion_scale=args.fusion_scale,
                guidance_scale=1.0,
            ),
            device,
            image_encoder=image_encoder,
        )
        ids = self.teacher.tokenizer(
            [""], padding="max_length", truncation=True,
            max_length=self.teacher.tokenizer.model_max_length, return_tensors="pt",
        ).input_ids.to(device)
        with torch.no_grad():
            self.text = self.teacher.text_encoder(ids)[0]
        # The student already encodes the decoded video.
        del self.teacher.vae, self.teacher.text_encoder

    @torch.no_grad()
    def __call__(self, batch, condition, ground_truth, clips, frames):
        teacher = self.teacher

        def embed(key):
            images = batch[key].to(teacher.device, teacher.dtype)
            return torch.cat([teacher.image_encoder(frame).image_embeds for frame in images.split(1)])

        embeddings = embed("clip_images") + teacher.config["fusion_scale"] * teacher.fusion(
            embed("fg_clip_images"), embed("bg_clip_images")
        )
        tokens = teacher.projection(embeddings)
        states, timesteps = [], []
        indices = (
            [self.steps - 1] * clips
            if self.full_grid_smoke else torch.randint(self.steps, (clips,)).tolist()
        )
        for index, stop_index in enumerate(indices):
            start = index * frames
            end = start + frames
            teacher.tokens = tokens[start:end]
            teacher.text = self.text.expand(frames, -1, -1)
            state, timestep = invert_decoded_clip(
                self.scheduler, teacher.predict_noise,
                condition[start:end].to(teacher.dtype), self.steps, stop_index,
            )
            states.append(state.to(ground_truth.dtype))
            timesteps.append(timestep.expand(frames))
        noisy, times = torch.cat(states), torch.cat(timesteps)
        target = epsilon_target(noisy, ground_truth, times, self.training_scheduler)
        if not torch.isfinite(target).all():
            raise FloatingPointError("Non-finite inversion training target")
        return noisy, times, target


if __name__ == "__main__":
    os.environ.setdefault("CONFIG", str(HERE / "configs/train.yaml"))
    args = parse_args()
    args.inversion_teacher = os.environ.get(
        "INVERSION_TEACHER", str(Path(args.brushnet_model_name_or_path).parent)
    )
    args.inversion_steps = int(os.environ.get("INVERSION_STEPS", "50"))
    if not 1 <= args.inversion_steps <= args.expected_num_train_timesteps:
        raise ValueError("Invalid INVERSION_STEPS")
    main(args, trajectory_factory=InversionTrajectory)
