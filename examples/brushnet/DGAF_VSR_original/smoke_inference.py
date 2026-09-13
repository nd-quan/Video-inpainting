"""Real-model short inference smoke; saves a preview, never trains or checkpoints."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from DGAF_VSR_original.train_dgaf import parse_args
from DGAF_VSR_original.common import load_stack, make_dataset, guidance_from_args, json_dump
from DGAF_VSR_original.warping import SequenceWarper
from STC_encoder_v8_raft_deformable.raft_flow_provider import FrozenV7RAFTFlowProvider
from STC_encoder_v2_rgb.evaluate_rgb_stc_shared_noise import build_v8_prompt_embeddings

def main():
    args = parse_args()
    device = torch.device("cuda", 0)
    torch.manual_seed(args.seed)
    dataset = make_dataset(args)
    pipe, encoder, projection, fusion, _ = load_stack(args, device, inference=True)
    pipe.enable_vae_slicing()
    provider = FrozenV7RAFTFlowProvider(args.raft_student_path, device=device)
    sample = dataset[0]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with torch.inference_mode():
        rgb = sample["conditioning_pixel_values"].to(device)
        base = pipe.vae.encode(rgb.to(pipe.vae.dtype)).latent_dist.mode().float() * pipe.vae.config.scaling_factor
        bg = F.interpolate(sample["masks"].to(device), size=base.shape[-2:], mode="nearest")
        flow = provider.predict_sequence(rgb[None])
        warper = SequenceWarper(flow.forward, flow.backward, bg[None], guidance_from_args(args))
        prompt, negative, text, negative_text = build_v8_prompt_embeddings(
            pipe, encoder, projection, fusion, sample, device, args.fusion_scale)
        result = pipe(latents=torch.randn_like(base), base_condition=torch.cat((base,bg),1),
            warper=warper, prompt_embeds=prompt, negative_prompt_embeds=negative,
            brushnet_prompt_embeds=text, negative_brushnet_prompt_embeds=negative_text,
            num_inference_steps=3, guidance_scale=7.5, output_type="latent").images
        if not torch.isfinite(result).all():
            raise AssertionError("Nonfinite inference output")
        decoded = pipe.vae.decode(result.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor, return_dict=False)[0]
        for index, image in enumerate(pipe.image_processor.postprocess(decoded, output_type="pil")):
            image.save(args.output_dir / f"frame_{index}.png")
    json_dump(args.output_dir / "result.json", dict(status="ok", shape=list(result.shape),
        warp_mode=args.warp_mode, confidence=args.use_confidence_mask, diffusion_steps=3))
    print("Real sequential inference passed", flush=True)

if __name__ == "__main__":
    main()
