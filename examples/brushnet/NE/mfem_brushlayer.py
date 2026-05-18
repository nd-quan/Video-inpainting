import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from einops import rearrange
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from ldm.modules.diffusionmodules.util import conv_nd, linear, zero_module, timestep_embedding
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential, ResBlock, Downsample, Upsample

# -------------------------
# helpers
# -------------------------
def to_3d(x):   # (B,C,H,W) -> (B,HW,C)
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):  # (B,HW,C) -> (B,C,H,W)
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

# -------------------------
# LayerNorm (Restormer style)
# -------------------------
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        # x: (B,HW,C)
        var = x.var(-1, unbiased=False, keepdim=True)
        return x / torch.sqrt(var + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias   = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        # x: (B,HW,C)
        mean = x.mean(-1, keepdim=True)
        var  = x.var(-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + 1e-5)
        return x * self.weight + self.bias

class LayerNorm2d(nn.Module):
    """Apply token-wise LN by flattening spatial, like Restormer."""
    def __init__(self, dim, ln_type='WithBias'):
        super().__init__()
        self.body = WithBias_LayerNorm(dim) if ln_type=='WithBias' else BiasFree_LayerNorm(dim)
    def forward(self, x):
        b, c, h, w = x.shape
        return to_4d(self.body(to_3d(x)), h, w)

# -------------------------
# FeedForward (Gated-Dconv FFN, simplified)
# -------------------------
class GDFN(nn.Module):
    def __init__(self, dim, expansion=2.66, bias=False):
        super().__init__()
        hidden = int(dim * expansion)
        self.proj1 = nn.Conv2d(dim, hidden*2, 1, bias=bias)
        self.dw    = nn.Conv2d(hidden*2, hidden*2, 3, padding=1, groups=hidden*2, bias=bias)
        self.act   = nn.GELU()
        self.proj2 = nn.Conv2d(hidden, dim, 1, bias=bias)
    def forward(self, x):
        y = self.proj1(x)
        y = self.dw(y)
        y1, y2 = y.chunk(2, dim=1)
        y = self.act(y1) * y2
        y = self.proj2(y)
        return y

# -------------------------
# MDTA (simplified: conv-projected MSA on tokens)
# -------------------------
# class MDTA(nn.Module):
#     def __init__(self, dim, num_heads=8, bias=False):
#         super().__init__()
#         self.num_heads = num_heads
#         self.scale = (dim // num_heads) ** -0.5
#         self.qkv = nn.Conv2d(dim, dim*3, 1, bias=bias)
#         self.dw  = nn.Conv2d(dim*3, dim*3, 3, padding=1, groups=dim*3, bias=bias)  # depthwise spice
#         self.proj = nn.Conv2d(dim, dim, 1, bias=bias)

#     def forward(self, x):
#         b, c, h, w = x.shape
#         qkv = self.qkv(x)
#         qkv = self.dw(qkv)
#         q, k, v = qkv.chunk(3, dim=1)  # (B,C,H,W)

#         # -> tokens
#         q = rearrange(q, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
#         k = rearrange(k, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
#         v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
#                # 안정성: matmul은 float32로, 출력은 원 dtype로+       
#         q_dtype = q.dtype
#         q = q.float(); k = k.float(); v = v.float()
#         attn = (q * self.scale) @ k.transpose(-2, -1)  # (B,heads,N,N)
#         attn = attn.softmax(dim=-1)
#         # out  = attn @ v                                # (B,heads,N,d)
#         # out  = rearrange(out, 'b h n d -> b (h d) n')
#         # out  = to_4d(out[:, :, None], 1, out.shape[-1])  # (B,C,1,N)
#         # out  = rearrange(out, 'b c 1 n -> b c n 1')
#         # out  = F.interpolate(out, size=(h, w), mode='nearest')  # naive reshape back
#         # 더 안정적인 복원을 위해 직접 reshape:
#         # out = rearrange(out, 'b c h w -> b c h w')  # no-op, for clarity

#         # # 위의 token→image 복원은 간단화. 더 정확히 하려면:
#         # out = rearrange(out, 'b c h w -> b (h w) c')
#         # out = rearrange(out, 'b n c -> b c n 1')
#         # out = out.view(b, c, h, w)

#         # return self.proj(out)
#                # 바로 (B,C,H,W)로 복원
#         out  = rearrange(out, 'b h n d -> b (h d) n')        # (B,C,N)
#         out  = rearrange(out, 'b c (hw) -> b c h w', h=h, w=w)
#         out = out.to(q_dtype)
#         return self.proj(out)

class MDTA(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        assert dim % num_heads == 0, f"MDTA: dim({dim}) must be divisible by num_heads({num_heads})"
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.dw   = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w

        qkv = self.dw(self.qkv(x))                    # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)                 # each (B, C, H, W)

        # (B, C, H, W) -> (B, heads, N, head_dim)
        q = q.reshape(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2).contiguous()
        k = k.reshape(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2).contiguous()
        v = v.reshape(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2).contiguous()
        # now q/k/v: (B, heads, N, head_dim)

        # 안정성: matmul은 float32로
        orig_dtype = q.dtype
        q = q.float(); k = k.float(); v = v.float()

        attn = torch.matmul(q * self.scale, k.transpose(-2, -1))   # (B, heads, N, N)
        attn = attn.softmax(dim=-1)

        ctx  = torch.matmul(attn, v)                                # (B, heads, N, head_dim)
        ctx  = ctx.permute(0, 1, 3, 2).contiguous()                 # (B, heads, head_dim, N)
        ctx  = ctx.view(b, c, n)                                    # (B, C, N)
        out  = ctx.view(b, c, h, w)                                 # (B, C, H, W)
        out  = out.to(orig_dtype)

        return self.proj(out)                                       # (B, C, H, W)

# -------------------------
# TransformerBlock with adaLN (shift, scale from time embed)
# -------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, time_embed_dim, ffn_expansion=2.66, bias=False, ln_type='WithBias'):
        super().__init__()
        self.norm1 = LayerNorm2d(dim, ln_type)
        self.attn  = MDTA(dim, num_heads=num_heads, bias=bias)
        self.norm2 = LayerNorm2d(dim, ln_type)
        self.ffn   = GDFN(dim, expansion=ffn_expansion, bias=bias)

        # time embedding to shift/scale (adaLN)
        self.to_ff = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, dim*2))
        self.to_attn = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, dim*2))

    def forward(self, x, t_emb):
        # attn
        sh, sc = self.to_attn(t_emb).chunk(2, dim=1)   # (B,dim) x 2
        sh = sh[:, :, None, None]; sc = sc[:, :, None, None]
        y = modulate(self.norm1(x), sh, sc)
        y = self.attn(y)
        x = x + y

        # ffn
        sh, sc = self.to_ff(t_emb).chunk(2, dim=1)
        sh = sh[:, :, None, None]; sc = sc[:, :, None, None]
        y = modulate(self.norm2(x), sh, sc)
        y = self.ffn(y)
        x = x + y
        return x

# -------------------------
# MFEM Brush Layer (drop-in for BrushNetModel)
# -------------------------
class MFEMBrushLayer(nn.Module):
    
    @property
    def dtype(self):
        # 모듈 파라미터 중 첫 번째 텐서의 dtype을 따름. 파라미터가 없으면 float32.
        for p in self.parameters():
            return p.dtype
        return torch.float32
    """
    BrushNetModel 대체: forward 시그니처와 반환형을 동일하게 유지
      - inputs: (sample, timestep, encoder_hidden_states, brushnet_cond, return_dict)
      - outputs: (down_feats(list of 3), mid_feat, up_feats(list of 3))
    """
    def __init__(
        self,
        unet_config,                # diffusers UNet config (채널, cross_attn dim 참조용)
        in_channels: int = 4,       # noisy_latents 채널 수
        hint_channels: int = 5,     # conditioning_latents 4 + mask 1
        control_channels: int = 320,
        time_embed_dim: int = 1280,
        heads=(1, 2, 4, 8),
        ffn_expansion: float = 2.66,
        conv_resample: bool = True,
        dims: int = 2,
        ln_type: str = 'WithBias',
        bias: bool = False,
    ):
        super().__init__()
        self.C = control_channels
        self.dims = dims

        # t-embedding (diffusers 스타일)
        # self.time_proj  = Timesteps(control_channels, downscale_freq_shift=0)
        self.time_proj  = Timesteps(
        control_channels,
        flip_sin_to_cos=True,      # 혹은 False도 가능 (SD v1.x는 True가 흔함)
        downscale_freq_shift=0,
        )
        self.time_embed = TimestepEmbedding(control_channels, time_embed_dim)

        # 입력 프로젝션들
        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1), nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1), nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2), nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1), nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2), nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1), nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2), nn.SiLU(),
            zero_module(conv_nd(dims, 256, control_channels, 3, padding=1))
        )
        self.input_blocks = TimestepEmbedSequential(
            conv_nd(dims, in_channels, control_channels, 3, padding=1)
        )

        # 첫 ResBlock (원래 BrushNet도 종종 하나 넣음)
        self.resblock0 = ResBlock(control_channels, time_embed_dim, dropout=0, out_channels=control_channels)

        # Encoder (64→32→16), Downsample는 LDM의 Downsample 재사용
        self.enc1 = TransformerBlock(self.C,          heads[0], time_embed_dim, ffn_expansion, bias, ln_type)
        self.down1= Downsample(self.C, conv_resample, dims, self.C*2)

        self.enc2 = TransformerBlock(self.C*2,        heads[1], time_embed_dim, ffn_expansion, bias, ln_type)
        self.down2= Downsample(self.C*2, conv_resample, dims, self.C*4)

        self.enc3 = TransformerBlock(self.C*4,        heads[2], time_embed_dim, ffn_expansion, bias, ln_type)
        self.down3= Downsample(self.C*4, conv_resample, dims, self.C*4)  # 16→8 (채널 유지 1280)

        # Mid (8x8)
        self.mid1 = TransformerBlock(self.C*4, heads[3], time_embed_dim, ffn_expansion, bias, ln_type)
        self.mid2 = TransformerBlock(self.C*4, heads[3], time_embed_dim, ffn_expansion, bias, ln_type)
        self.mid3 = TransformerBlock(self.C*4, heads[3], time_embed_dim, ffn_expansion, bias, ln_type)

        # Decoder (8→16→32→64)
        self.up3  = Upsample(self.C*4, conv_resample, dims, self.C*4)   # 8→16
        self.red3 = nn.Conv2d(self.C*8, self.C*4, 1, bias=bias)         # cat(enc3) 후 채널 축소
        self.dec3 = TransformerBlock(self.C*4, heads[2], time_embed_dim, ffn_expansion, bias, ln_type)

        self.up2  = Upsample(self.C*4, conv_resample, dims, self.C*2)   # 16→32
        self.red2 = nn.Conv2d(self.C*4, self.C*2, 1, bias=bias)
        self.dec2 = TransformerBlock(self.C*2, heads[1], time_embed_dim, ffn_expansion, bias, ln_type)

        self.up1  = Upsample(self.C*2, conv_resample, dims, self.C)     # 32→64
        self.red1 = nn.Conv2d(self.C*2, self.C, 1, bias=bias)
        self.dec1 = TransformerBlock(self.C,   heads[0], time_embed_dim, ffn_expansion, bias, ln_type)

        # UNet에 줄 residual들을 1x1로 정렬 (TimestepEmbedSequential로 감싸 동일 호출 인터페이스)
        self.zero_convs = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, self.C,   self.C,   1)),
            TimestepEmbedSequential(conv_nd(dims, self.C*2, self.C*2, 1)),
            TimestepEmbedSequential(conv_nd(dims, self.C*4, self.C*4, 1)),
            TimestepEmbedSequential(conv_nd(dims, self.C*4, self.C*4, 1)),  # mid
            # 여유 슬롯(필요시 확장)
            TimestepEmbedSequential(conv_nd(dims, self.C*4, self.C*4, 1)),
            TimestepEmbedSequential(conv_nd(dims, self.C*4, self.C*4, 1)),
            TimestepEmbedSequential(conv_nd(dims, self.C*4, self.C*4, 1)),  # up @16
            TimestepEmbedSequential(conv_nd(dims, self.C*2, self.C*2, 1)),  # up @32
            TimestepEmbedSequential(conv_nd(dims, self.C,   self.C,   1)),  # up @64
        ])

    def _time_embed(self, timesteps: torch.Tensor):
        t = self.time_proj(timesteps)       # (B, C)
        t = self.time_embed(t)              # (B, time_embed_dim)
        return t

    def forward(
        self,
        sample: torch.FloatTensor,                # noisy_latents (unused for features, but kept for signature parity)
        timestep: torch.Tensor,                   # (B,)
        encoder_hidden_states: Optional[torch.Tensor] = None,   # (B,77,768) - 여기서는 직접 쓰진 않음(필요시 확장)
        brushnet_cond: Optional[torch.FloatTensor] = None,      # (B, 4+1, H, W)
        return_dict: bool = False,
        **kwargs,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:

        assert brushnet_cond is not None, "MFEMBrushLayer: brushnet_cond (latent+mask) is required."

        # time embedding
        t_emb = self._time_embed(timestep)    # (B, time_embed_dim)

        # 입력 처리: cond(hint)와 sample(latent)를 각각 proj
        h_hint = self.input_hint_block(brushnet_cond, t_emb, encoder_hidden_states)  # (B, C, H/8, W/8) after strides
        h_x    = self.input_blocks(sample, t_emb, encoder_hidden_states)             # (B, C, H, W)

        # 해상도를 맞춰 더함 (h_hint는 3번 stride=2 했으니 H/8)
        if h_hint.shape[-2:] != h_x.shape[-2:]:
            h_hint = F.interpolate(h_hint, size=h_x.shape[-2:], mode='bilinear', align_corners=False)
        h = h_x + h_hint

        # 초기 ResBlock
        # h = self.resblock0(h, t_emb, encoder_hidden_states)
        h = self.resblock0(h, t_emb) # 체크 필요
        
        # Encoder stage 1 (64x64, C)
        e1 = self.enc1(h, t_emb)
        r_d1 = self.zero_convs[0](e1, t_emb, encoder_hidden_states)     # down[0]
        # h  = self.down1(e1, t_emb, encoder_hidden_states)               # (B,2C,32,32)
        h=self.down1(e1)

        # Encoder stage 2 (32x32, 2C)
        e2 = self.enc2(h, t_emb)
        r_d2 = self.zero_convs[1](e2, t_emb, encoder_hidden_states)     # down[1]
        # h  = self.down2(e2, t_emb, encoder_hidden_states)               # (B,4C,16,16)
        h=self.down2(e2)
        
        # Encoder stage 3 (16x16, 4C)
        e3 = self.enc3(h, t_emb)
        r_d3 = self.zero_convs[2](e3, t_emb, encoder_hidden_states)     # down[2]
        # h  = self.down3(e3, t_emb, encoder_hidden_states)               # (B,4C,8,8)
        h=self.down3(e3)
        
        # Mid (8x8, 4C)
        m = self.mid1(h, t_emb)
        m = self.mid2(m, t_emb)
        m = self.mid3(m, t_emb)
        r_mid = self.zero_convs[3](m, t_emb, encoder_hidden_states)     # mid

        # Up stage 3: 8→16
        # u3 = self.up3(m, t_emb, encoder_hidden_states)                  # (B,4C,16,16)
        u3 = self.up3(m)
        cat3 = torch.cat([u3, e3], dim=1)                               # (B,8C,16,16)
        u3  = self.red3(cat3)
        u3  = self.dec3(u3, t_emb)
        r_u3 = self.zero_convs[6](u3, t_emb, encoder_hidden_states)     # up[0] @16x16

        # Up stage 2: 16→32
        # u2 = self.up2(u3, t_emb, encoder_hidden_states)                 # (B,2C,32,32)
        u2 = self.up2(u3)
        # enc2와 해상도 동일이므로 concat
        cat2 = torch.cat([u2, e2], dim=1)                               # (B,4C,32,32)
        u2  = self.red2(cat2)
        u2  = self.dec2(u2, t_emb)
        r_u2 = self.zero_convs[7](u2, t_emb, encoder_hidden_states)     # up[1] @32x32

        # Up stage 1: 32→64
        # u1 = self.up1(u2, t_emb, encoder_hidden_states)                 # (B,C,64,64)
        u1 = self.up1(u2)
        # enc1과 concat
        cat1 = torch.cat([u1, e1], dim=1)                               # (B,2C,64,64)
        u1  = self.red1(cat1)
        u1  = self.dec1(u1, t_emb)
        r_u1 = self.zero_convs[8](u1, t_emb, encoder_hidden_states)     # up[2] @64x64

        down_list = [r_d1, r_d2, r_d3]
        up_list   = [r_u3, r_u2, r_u1]

        return (down_list, r_mid, up_list)
