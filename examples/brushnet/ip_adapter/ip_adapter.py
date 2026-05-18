import os
from typing import List

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .utils import is_torch2_available, get_generator

if is_torch2_available():
    print("torch2실행되서 gate 실행되어요~")
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    print("torch1실행되서 gate 실행안되어요~")
    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from .resampler import Resampler


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )
        
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class IPAdapter:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor( # 여기서 내부에서 gate가 적용됨
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    # def load_ip_adapter(self):
    #     if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
    #         state_dict = {"image_proj": {}, "ip_adapter": {}}
    #         with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
    #             for key in f.keys():
    #                 if key.startswith("image_proj."):
    #                     state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
    #                 elif key.startswith("ip_adapter."):
    #                     state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
    #     else:
    #         state_dict = torch.load(self.ip_ckpt, map_location="cpu")
    #     self.image_proj_model.load_state_dict(state_dict["image_proj"])
    #     ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
    #     ip_layers.load_state_dict(state_dict["ip_adapter"])

    # 나은 train하니까 저장다르게되서 수정함
    def load_ip_adapter(self):
            if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
                state_dict = {"image_proj": {}, "ip_adapter": {}}

                # UNet의 모든 어텐션 프로세서 키 리스트 (순서가 매우 중요함)
                attn_names = list(self.pipe.unet.attn_processors.keys())

                with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                    f_keys = f.keys()
                    
                    # 1) image proj 로드 (image_proj_model. prefix 제거)
                    for key in f_keys:
                        if key.startswith("image_proj_model."):
                            new_key = key.replace("image_proj_model.", "")
                            state_dict["image_proj"][new_key] = f.get_tensor(key)

                    # 2) ip_adapter 로드 현황 파악을 위한 변수
                    load_count = 0
                    total_attn2_count = 0

                    for idx, name in enumerate(attn_names):
                        # IP-Adapter는 attn2.processor에만 존재함
                        if not name.endswith("attn2.processor"):
                            continue
                        
                        total_attn2_count += 1
                        
                        # 훈련 시 저장된 규격: unet.{name}.{weight_name}
                        # 예: unet.down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight
                        k_key = f"unet.{name}.to_k_ip.weight"
                        v_key = f"unet.{name}.to_v_ip.weight"
                        # 2. Gate 레이어 키 정의 (훈련 시 저장 규격에 맞춤)
                        
                        g0w_key = f"unet.{name}.gate.0.weight"
                        g0b_key = f"unet.{name}.gate.0.bias"
                        g2w_key = f"unet.{name}.gate.2.weight"
                        g2b_key = f"unet.{name}.gate.2.bias"
                        
                        # 3. K, V 로드 (필수)
                        if k_key in f_keys and v_key in f_keys:
                            state_dict["ip_adapter"][f"{idx}.to_k_ip.weight"] = f.get_tensor(k_key)
                            state_dict["ip_adapter"][f"{idx}.to_v_ip.weight"] = f.get_tensor(v_key)
                            
                            # 4. Gate 로드 (선택적: 파일에 있을 때만)
                            if g0w_key in f_keys:
                                state_dict["ip_adapter"][f"{idx}.gate.0.weight"] = f.get_tensor(g0w_key)
                                state_dict["ip_adapter"][f"{idx}.gate.0.bias"] = f.get_tensor(g0b_key)
                                state_dict["ip_adapter"][f"{idx}.gate.2.weight"] = f.get_tensor(g2w_key)
                                state_dict["ip_adapter"][f"{idx}.gate.2.bias"] = f.get_tensor(g2b_key)
                                # print(f"  - Gate layers loaded for: {name}") # 필요 시 주석 해제
                                
                            load_count += 1
                        else:
                            print(f"⚠️ [Skip] Base IP-Adapter keys not found: {name}")      

                # print(f"✅ IP-Adapter 로드 완료: {load_count}/{total_attn2_count} 레이어 매핑 성공")
                # --- 🔍 꼬장꼬장한 전수 조사 시작 ---
                all_f_ip_keys = [k for k in f_keys if "to_k_ip" in k or "to_v_ip" in k or "gate" in k]
                loaded_keys_in_file = []
                
                # 실제로 state_dict에 담긴 원본 파일 키들을 추적 (위 로직에서 수집하도록 수정 필요)
                # (위의 for idx, name 루프 안에서 found_keys 리스트를 만들어 담았다고 가정)
                
                print("\n" + "="*50)
                print("🕵️ 전수 조사 리포트")
                print(f"1. 파일 내 총 IP 관련 텐서: {len(all_f_ip_keys)}개")
                # 텐서당 K, V 2개 혹은 K, V, G0w, G0b, G2w, G2b 6개이므로 
                # (성공 레이어 수 * 각 레이어별 텐서 수)와 비교합니다.
                
                loaded_tensor_count = len(state_dict["ip_adapter"])
                print(f"2. state_dict에 담긴 총 텐서: {loaded_tensor_count}개")
                
                if len(all_f_ip_keys) == loaded_tensor_count:
                    print("✨ 결과: [완벽 일치] 파일의 모든 텐서가 누락 없이 매핑되었습니다.")
                else:
                    print(f"⚠️ 결과: [불일치] 파일에는 {len(all_f_ip_keys)}개가 있지만, {loaded_tensor_count}개만 로드됨.")
                    # 어떤 키가 빠졌는지 출력
                    loaded_set = set()
                    # 위 루프에서 사용된 k_key, v_key 등을 저장해서 비교하는 것이 가장 정확합니다.
                print("="*50 + "\n")
            else:
                state_dict = torch.load(self.ip_ckpt, map_location="cpu")

            # --- 실제 모델에 가중치 적용 ---
            
            # 1. Image Projection 적용
            if hasattr(self, "image_proj_model"):
                self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=False)

            # 2. IP-Adapter (UNet 프로세서들) 적용
            # attn_processors.values()의 순서가 attn_names의 인덱스와 일치함
            ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
            ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)
    
    
     
    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        # ---- 여기부터가 핵심 수정 ----
        # 1) kwargs에서 brushnet 필수 입력을 꺼내서 명시적으로 전달
        image = kwargs.pop("image", None)

        # pipeline 구현에 따라 mask 키가 mask_image일 수도, mask일 수도 있어서 둘 다 처리
        mask_image = kwargs.pop("mask_image", None)
        mask = kwargs.pop("mask", None)
        if mask_image is None and mask is not None:
            mask_image = mask

        # ip-adapter 이미지도 명시적으로 주는 게 안전 (kwargs에 있으면 제거해서 중복 방지)
        ip_adapter_image = kwargs.pop("ip_adapter_image", None)
        if ip_adapter_image is None:
            ip_adapter_image = pil_image

        # 2) generator 중복 방지: seed로 만든 generator를 쓰되, kwargs로 들어오면 그걸 우선
        generator = kwargs.pop("generator", None)
        if generator is None:
            generator = get_generator(seed, self.device)

        # 3) brushnet_conditioning_scale도 kwargs에 있으면 꺼내서 명시적으로 전달 (중복 방지)
        brushnet_conditioning_scale = kwargs.pop("brushnet_conditioning_scale", None)
        # ---- 수정 끝 ----

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            # ⭐ BrushNet 필수
            image=image,
            mask_image=mask_image,
            mask=mask_image,
            **kwargs,
        ).images


        return images


# 나은 추가 (feature fusion 부분 )
class FGBGFeatureFusion(torch.nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.fg_to_bg_attn = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.bg_to_fg_attn = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.out_proj = torch.nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, fg_feat, bg_feat):
        fg_feat = fg_feat.unsqueeze(1)   # [B, 1, D]
        bg_feat = bg_feat.unsqueeze(1)   # [B, 1, D]

        fg2bg, _ = self.fg_to_bg_attn(
            query=fg_feat,
            key=bg_feat,
            value=bg_feat,
        )
        bg2fg, _ = self.bg_to_fg_attn(
            query=bg_feat,
            key=fg_feat,
            value=fg_feat,
        )

        fused = torch.cat([fg2bg, bg2fg], dim=-1)   # [B, 1, 2D]
        fused = self.out_proj(fused)                # [B, 1, D]

        return fused.squeeze(1)                     # [B, D]

class FusionIPAdapter(IPAdapter):
    def __init__(
        self,
        sd_pipe,
        image_encoder_path,
        ip_ckpt,
        fusion_ckpt,
        device,
        num_tokens=4,
    ):
        super().__init__(sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens)

        self.fusion_module = FGBGFeatureFusion(
            embed_dim=self.image_encoder.config.projection_dim,
            num_heads=8,
        ).to(self.device, dtype=torch.float16)

        if os.path.splitext(fusion_ckpt)[-1] == ".safetensors":
            fusion_state = {}
            with safe_open(fusion_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    fusion_state[key] = f.get_tensor(key)
        else:
            fusion_state = torch.load(fusion_ckpt, map_location="cpu")

        self.fusion_module.load_state_dict(fusion_state, strict=True)
        self.fusion_module.eval()

    @torch.inference_mode()
    def get_fgbg_image_embeds(self, fg_pil_image=None, bg_pil_image=None):
        if isinstance(fg_pil_image, Image.Image):
            fg_pil_image = [fg_pil_image]
        if isinstance(bg_pil_image, Image.Image):
            bg_pil_image = [bg_pil_image]

        fg_clip = self.clip_image_processor(images=fg_pil_image, return_tensors="pt").pixel_values
        bg_clip = self.clip_image_processor(images=bg_pil_image, return_tensors="pt").pixel_values

        fg_clip = fg_clip.to(self.device, dtype=torch.float16)
        bg_clip = bg_clip.to(self.device, dtype=torch.float16)

        fg_embeds = self.image_encoder(fg_clip).image_embeds   # [B, D]
        bg_embeds = self.image_encoder(bg_clip).image_embeds   # [B, D]

        fused_embeds = self.fusion_module(fg_embeds, bg_embeds)   # [B, D]

        image_prompt_embeds = self.image_proj_model(fused_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(fused_embeds))

        return image_prompt_embeds, uncond_image_prompt_embeds

    def generate_fgbg(
        self,
        fg_pil_image,
        bg_pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(fg_pil_image, Image.Image) else len(fg_pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_fgbg_image_embeds(
            fg_pil_image=fg_pil_image,
            bg_pil_image=bg_pil_image,
        )

        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        image = kwargs.pop("image", None)
        mask_image = kwargs.pop("mask_image", None)
        mask = kwargs.pop("mask", None)
        if mask_image is None and mask is not None:
            mask_image = mask

        generator = kwargs.pop("generator", None)
        if generator is None:
            generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            image=image,
            mask_image=mask_image,
            mask=mask_image,
            **kwargs,
        ).images

        return images

class IPAdapterXL(IPAdapter):
    """SDXL"""

    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)
        
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            **kwargs,
        ).images

        return images


class IPAdapterPlus(IPAdapter):
    """IP-Adapter with fine-grained features"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


class IPAdapterFull(IPAdapterPlus):
    """IP-Adapter with full features"""

    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model


class IPAdapterPlusXL(IPAdapter):
    """SDXL"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images
