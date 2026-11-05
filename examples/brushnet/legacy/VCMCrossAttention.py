# VCMCrossAttention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ContrastiveEmphasisVCM", "BiDirCrossAttentionVCM"]


# # ---------- 공통 유틸 ----------
# def _posenc_2d(H, W, D, device):
#     y, x = torch.meshgrid(
#         torch.arange(H, device=device),
#         torch.arange(W, device=device),
#         indexing="ij",
#     )
#     dim = max(1, D // 4)
#     omega = torch.arange(dim, device=device) / dim
#     omega = 1.0 / (10000 ** omega)
#     y = y.reshape(-1, 1) * omega
#     x = x.reshape(-1, 1) * omega
#     pe = torch.cat([torch.sin(y), torch.cos(y), torch.sin(x), torch.cos(x)], dim=1)
#     if pe.shape[1] < D:
#         pe = F.pad(pe, (0, D - pe.shape[1]))
#     return pe  # (H*W, D)


# class _Windower:
#     @staticmethod
#     def tile(feat, win):
#         # (B,D,H,W) -> list of (B,D,win,win)
#         B, D, H, W = feat.shape
#         patches = []
#         for i in range(0, H, win):
#             for j in range(0, W, win):
#                 patches.append(feat[:, :, i:i+win, j:j+win])
#         return patches

#     @staticmethod
#     def merge(patches, H, W, win):
#         B, D = patches[0].shape[:2]
#         out = torch.zeros(B, D, H, W, device=patches[0].device, dtype=patches[0].dtype)
#         idx = 0
#         for i in range(0, H, win):
#             for j in range(0, W, win):
#                 out[:, :, i:i+win, j:j+win] = patches[idx]
#                 idx += 1
#         return out


# ==========================================================
# 1) ContrastiveEmphasisVCM
#    - ROI(reference): relevant 강조 (ROI <- BG)
#    - BG(target): ROI로 예측 가능한 성분 제거 + 잔차(irrelevant) 강조
# ==========================================================
# NE V1.1
# class ContrastiveEmphasisVCM(nn.Module):
#     """
#     입력/출력: BG, ROI 각각 (B, in_ch, H, W) -> (B, in_ch, H, W)
#     내부에 d_model로 올려서 연산 후 다시 in_ch로 투영.
#     - ROI branch:
#         ROI가 Q, BG가 K/V인 cross-attn으로 BG의 '중요한' 정보만 선택적으로 참조.
#         유사도 기반 gate_rel로 강도 제어.
#     - BG branch:
#         BG의 ROI-예측 성분을 제거(BG_proj_on_ROI)하여 잔차(residual)를 강조 (irrelevant preservation).
#         gate_irrel = 1 - gate_rel 로 잔차 강조.
#     """
#     def __init__(self, in_ch=4, d_model=64, nhead=4, attn_window=None,
#                  alpha_roi=1.0, beta_bg=1.0, use_layernorm=True):
#         super().__init__()
#         self.in_ch = in_ch
#         self.d_model = d_model
#         self.nhead = nhead
#         self.attn_window = attn_window  # e.g., 8 → 8x8 로컬 cross-attn
#         self.alpha_roi = alpha_roi      # ROI로 들어오는 cross-attn 정보의 스케일
#         self.beta_bg = beta_bg          # BG 잔차(irrelevant) 강화 스케일

#         # 1x1 proj
#         self.bg_in  = nn.Conv2d(in_ch, d_model, 1)
#         self.roi_in = nn.Conv2d(in_ch, d_model, 1)
#         self.bg_out  = nn.Conv2d(d_model, in_ch, 1)
#         self.roi_out = nn.Conv2d(d_model, in_ch, 1)

#         # ROI <- BG : Q=ROI, K/V=BG
#         self.mha_roi_q__bg_kv = nn.MultiheadAttention(d_model, nhead, batch_first=True)

#         # 선택적 정규화
#         self.ln_roi = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
#         self.ln_bg  = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

#         # BG 잔차 보정용 얕은 head
#         self.bg_residual_mix = nn.Conv2d(d_model, d_model, 1)

#     @staticmethod
#     def _cosine_gate(bg, roi, eps=1e-6):
#         """
#         bg, roi: (B, D, H, W) 정규화 후 점별 cosine 유사도 → gate_rel in [0,1]
#         """
#         B, D, H, W = bg.shape
#         bg_n  = F.normalize(bg,  dim=1, eps=eps)
#         roi_n = F.normalize(roi, dim=1, eps=eps)
#         # 점곱: (B,1,H,W)
#         sim = (bg_n * roi_n).sum(dim=1, keepdim=True)
#         # [-1,1] → [0,1]
#         gate_rel = (sim + 1.0) * 0.5
#         return gate_rel.clamp(0, 1)

#     def _roi_from_bg(self, roi, bg):
#         """
#         ROI <- BG cross-attn (전역 or 윈도우)
#         입력: roi,bg (B,D,H,W), 출력: roi_enhanced (B,D,H,W)
#         """
#         B, D, H, W = roi.shape
#         device = roi.device

#         if self.attn_window is None:
#             N = H * W
#             pe = _posenc_2d(H, W, D, device).unsqueeze(0)  # (1,N,D)

#             def to_tok(x):  # (B,D,H,W) -> (B,N,D)
#                 return x.flatten(2).transpose(1, 2) + pe

#             q = self.ln_roi(to_tok(roi))
#             kv = self.ln_bg(to_tok(bg))
#             out, _ = self.mha_roi_q__bg_kv(q, kv, kv, need_weights=False)
#             out = q + out  # residual
#             return out.transpose(1, 2).reshape(B, D, H, W)
#         else:
#             win = self.attn_window
#             assert H % win == 0 and W % win == 0
#             pe = _posenc_2d(win, win, D, device).unsqueeze(0)

#             roi_p = _Windower.tile(roi, win)
#             bg_p  = _Windower.tile(bg,  win)
#             out_p = []
#             for r_w, b_w in zip(roi_p, bg_p):
#                 def to_tok(x):
#                     return x.flatten(2).transpose(1, 2) + pe
#                 q  = self.ln_roi(to_tok(r_w))
#                 kv = self.ln_bg(to_tok(b_w))
#                 o, _ = self.mha_roi_q__bg_kv(q, kv, kv, need_weights=False)
#                 o = q + o
#                 out_p.append(o.transpose(1, 2).reshape(B, D, win, win))
#             return _Windower.merge(out_p, H, W, win)

#     @staticmethod
#     def _proj(bg, roi, eps=1e-6):
#         """
#         bg를 roi 방향으로 점별(projection) 투영: proj_bg_on_roi
#         bg, roi: (B,D,H,W), 동일 D 가정
#         proj = <bg, roi_hat> * roi_hat  (채널 방향)
#         """
#         roi_norm = F.normalize(roi, dim=1, eps=eps)
#         coeff = (bg * roi_norm).sum(dim=1, keepdim=True)  # (B,1,H,W)
#         proj = coeff * roi_norm
#         return proj

#     def forward(self, BG, ROI):
#         """
#         BG, ROI: (B, in_ch, H, W)
#         return: BG_out, ROI_out  (각 in_ch 유지)
#         """
#         B, _, H, W = BG.shape
#         bg  = self.bg_in(BG)     # (B,D,H,W)
#         roi = self.roi_in(ROI)   # (B,D,H,W)

#         # 1) relevance gate (detached로 안정화)
#         with torch.no_grad():
#             gate_rel = self._cosine_gate(bg, roi)           # (B,1,H,W)
#         gate_irrel = 1.0 - gate_rel

#         # 2) ROI ← BG (relevant 강조) : cross-attn 후 gate로 스케일
#         roi_from_bg = self._roi_from_bg(roi, bg)            # (B,D,H,W)
#         roi_enh = roi + self.alpha_roi * (roi_from_bg * gate_rel)

#         # 3) BG (irrelevant 보존): BG의 ROI-예측 성분 제거 → 잔차 강조
#         proj_bg_on_roi = self._proj(bg, roi.detach())       # 안정화를 위해 roi는 detach
#         bg_residual = bg - proj_bg_on_roi                   # ROI로 설명되는 성분 제거
#         bg_residual = self.bg_residual_mix(bg_residual)     # 얕은 보정
#         bg_enh = bg + self.beta_bg * (bg_residual * gate_irrel)

#         # 4) 원 채널로 투영
#         BG_out  = self.bg_out(bg_enh)
#         ROI_out = self.roi_out(roi_enh)
#         return BG_out, ROI_out

# # NE V1.11
# class ContrastiveEmphasisVCM(nn.Module):
#     """
#     입력/출력: BG, ROI 각각 (B, in_ch, H, W) -> (B, in_ch, H, W)

#     설계 개요
#       - ROI branch (reference, relevant 강조):
#           ROI가 Q, BG가 K/V인 cross-attention.
#           *PE는 Q/K에만 약하게(pe_alpha) 적용*, residual은 *PE 없는 q_raw*로 더해 PE 누출 방지.
#           선택적 ROI 마스크(roi_mask)로 *쿼리 행*을 제한(원하는 위치에만 어텐션).
#           restrict_keys=True면 BG의 K/V도 ROI 내부로만 보게 제한.
#       - BG branch (target, irrelevant 보존):
#           BG의 ROI-예측 성분을 제거(투영)해 잔차를 강조, gate_irrel로 스케일.

#     Args:
#       in_ch:   입력 채널(예: VAE latent 4)
#       d_model: 내부 채널
#       nhead:   MH-Attention head 수
#       attn_window: 로컬 윈도우 크기(예: 8). None이면 전역 어텐션
#       alpha_roi: ROI cross-attn 주입량 스케일
#       beta_bg:  BG 잔차 강화 스케일
#       use_layernorm: LayerNorm 사용 여부
#       pe_alpha: PE 강도(0.05~0.1 권장)
#       restrict_keys: True면 BG 키/밸류도 ROI 내부로 제한
#     """
#     def __init__(
#         self,
#         in_ch=4,
#         d_model=64,
#         nhead=4,
#         attn_window=None,
#         alpha_roi=1.0,
#         beta_bg=1.0,
#         use_layernorm=True,
#         pe_alpha=0.1,
#         restrict_keys=False,
#     ):
#         super().__init__()
#         self.in_ch = in_ch
#         self.d_model = d_model
#         self.nhead = nhead
#         self.attn_window = attn_window
#         self.alpha_roi = alpha_roi
#         self.beta_bg = beta_bg
#         self.pe_alpha = pe_alpha
#         self.restrict_keys = restrict_keys

#         # 1x1 proj
#         self.bg_in  = nn.Conv2d(in_ch, d_model, 1)
#         self.roi_in = nn.Conv2d(in_ch, d_model, 1)
#         self.bg_out  = nn.Conv2d(d_model, in_ch, 1)
#         self.roi_out = nn.Conv2d(d_model, in_ch, 1)

#         # ROI <- BG : Q=ROI, K/V=BG
#         self.mha_roi_q__bg_kv = nn.MultiheadAttention(d_model, nhead, batch_first=True)

#         # 선택적 정규화
#         self.ln_roi = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
#         self.ln_bg  = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

#         # BG 잔차 보정용 얕은 head
#         self.bg_residual_mix = nn.Conv2d(d_model, d_model, 1)

#     @staticmethod
#     def _cosine_gate(bg, roi, eps=1e-6):
#         """
#         bg, roi: (B, D, H, W) 정규화 후 점별 cosine 유사도 → gate_rel in [0,1]
#         """
#         bg_n  = F.normalize(bg,  dim=1, eps=eps)
#         roi_n = F.normalize(roi, dim=1, eps=eps)
#         sim = (bg_n * roi_n).sum(dim=1, keepdim=True)  # (B,1,H,W)
#         gate_rel = (sim + 1.0) * 0.5
#         return gate_rel.clamp(0, 1)

#     @staticmethod
#     def _proj(bg, roi, eps=1e-6):
#         """
#         bg를 roi 방향으로 점별(projection) 투영: proj_bg_on_roi
#         bg, roi: (B,D,H,W)
#         proj = <bg, roi_hat> * roi_hat  (채널 방향)
#         """
#         roi_norm = F.normalize(roi, dim=1, eps=eps)
#         coeff = (bg * roi_norm).sum(dim=1, keepdim=True)  # (B,1,H,W)
#         proj = coeff * roi_norm
#         return proj

#     def _tok_with_pe(self, fmap, H, W, ln, pe):
#         """
#         fmap: (B,D,H,W) → tokens (B,N,D)
#         반환:
#           t_raw: LN(fmap tokens)             # residual 기준 (PE 없음)
#           t_pe : LN(fmap tokens + α*PE)      # attention 입력
#         """
#         tokens = fmap.flatten(2).transpose(1, 2)    # (B,N,D)
#         t_raw = ln(tokens)
#         t_pe  = ln(tokens + self.pe_alpha * pe)
#         return t_raw, t_pe

#     def _roi_from_bg(self, roi, bg, roi_mask=None):
#         """
#         ROI <- BG cross-attn
#           - residual은 q_raw 기준 (PE 누출 방지)
#           - roi_mask가 있으면 쿼리 행(ROI 바깥 위치)의 어텐션 출력을 0으로 차단
#           - restrict_keys=True면 BG 키/밸류도 ROI 내부로 제한(key_padding_mask)
#         """
#         B, D, H, W = roi.shape
#         device = roi.device

#         # ROI 마스크 준비 (latent 해상도와 정합)
#         qm, k_mask = None, None
#         if roi_mask is not None:
#             m = F.interpolate(roi_mask, size=(H, W), mode='nearest')  # (B,1,H,W)
#             m_bin = (m > 0.5).to(roi.dtype)
#             # 쿼리 마스크: (B,N,1) — 나중에 attn 출력에 곱해 query row를 제한
#             qm = m_bin.flatten(2).transpose(1, 2)  # (B,N,1)
#             if self.restrict_keys:
#                 # 키/밸류 마스크: (B,N) — True가 mask (MultiheadAttention 규약)
#                 k_mask = (m_bin.flatten(2).squeeze(1) < 0.5)  # ROI=1 → False(사용), ROI=0 → True(mask)

#         if self.attn_window is None:
#             N = H * W
#             pe = _posenc_2d(H, W, D, device).unsqueeze(0)  # (1,N,D)

#             q_raw, q_pe = self._tok_with_pe(roi, H, W, self.ln_roi, pe)
#             _,     k_pe = self._tok_with_pe(bg,  H, W, self.ln_bg,  pe)

#             out, _ = self.mha_roi_q__bg_kv(
#                 q_pe, k_pe, k_pe,
#                 need_weights=False,
#                 key_padding_mask=k_mask  # (B,N_k) True=mask
#             )
#             if qm is not None:
#                 out = out * qm  # 쿼리(ROI 바깥) 위치의 어텐션 효과 제거

#             out = q_raw + out   # residual은 PE 없는 q_raw 기준
#             return out.transpose(1, 2).reshape(B, D, H, W)

#         else:
#             win = self.attn_window
#             assert H % win == 0 and W % win == 0, f"attn_window={win}는 (H,W)=({H},{W})를 정확히 나눠야 합니다."
#             pe_w = _posenc_2d(win, win, D, device).unsqueeze(0)

#             roi_p = _Windower.tile(roi, win)
#             bg_p  = _Windower.tile(bg,  win)
#             if roi_mask is not None:
#                 m_bin = (F.interpolate(roi_mask, size=(H, W), mode='nearest') > 0.5).to(roi.dtype)
#                 m_p   = _Windower.tile(m_bin, win)  # list of (B,1,win,win)

#             outs = []
#             for i, (r_w, b_w) in enumerate(zip(roi_p, bg_p)):
#                 q_raw, q_pe = self._tok_with_pe(r_w, win, win, self.ln_roi, pe_w)
#                 _,     k_pe = self._tok_with_pe(b_w, win, win, self.ln_bg,  pe_w)

#                 if roi_mask is not None:
#                     m_w  = m_p[i]                                        # (B,1,win,win)
#                     qm_w = m_w.flatten(2).transpose(1, 2)                # (B,win*win,1)
#                     if self.restrict_keys:
#                         k_mask_w = (m_w.flatten(2).squeeze(1) < 0.5)     # (B,win*win) True=mask
#                     else:
#                         k_mask_w = None
#                 else:
#                     qm_w, k_mask_w = None, None

#                 o, _ = self.mha_roi_q__bg_kv(
#                     q_pe, k_pe, k_pe,
#                     need_weights=False,
#                     key_padding_mask=k_mask_w
#                 )
#                 if qm_w is not None:
#                     o = o * qm_w

#                 o = q_raw + o
#                 outs.append(o.transpose(1, 2).reshape(B, D, win, win))
#             return _Windower.merge(outs, H, W, win)

#     def forward(self, BG, ROI, roi_mask=None):
#         """
#         BG, ROI: (B, in_ch, H, W)
#         roi_mask: (B,1,H,W) 0/1  — 이미지 도메인 마스크를 latent 해상도로 nearest 리사이즈해서 전달
#         """
#         B, _, H, W = BG.shape
#         bg  = self.bg_in(BG)     # (B,D,H,W)
#         roi = self.roi_in(ROI)   # (B,D,H,W)

#         # 1) relevance gate
#         with torch.no_grad():
#             gate_rel = self._cosine_gate(bg, roi)   # (B,1,H,W)
#         gate_irrel = 1.0 - gate_rel

#         # 2) ROI ← BG (relevant 강조) : cross-attn + (옵션)마스크
#         roi_from_bg = self._roi_from_bg(roi, bg, roi_mask=roi_mask)  # (B,D,H,W)
#         roi_enh = roi + self.alpha_roi * (roi_from_bg * gate_rel)

#         # 3) BG (irrelevant 보존): ROI-예측 성분 제거 → 잔차 강조
#         proj_bg_on_roi = self._proj(bg, roi.detach())       # 안정화를 위해 roi.detach()
#         bg_residual = bg - proj_bg_on_roi
#         bg_residual = self.bg_residual_mix(bg_residual)
#         bg_enh = bg + self.beta_bg * (bg_residual * gate_irrel)

#         # 4) 원 채널로 투영
#         BG_out  = self.bg_out(bg_enh)
#         ROI_out = self.roi_out(roi_enh)
#         return BG_out, ROI_out
# v1.12 ( no mask/ window/PE 삭제)
class ContrastiveEmphasisVCM(nn.Module):
    # """
    # - 전역 cross-attn (Q=ROI, K/V=BG), 윈도우/PE 없음
    # - ROI: gate_rel ↑ 강조, BG: (1-gate_rel) ↑ 강조
    # """
    def __init__(self, in_ch=4, d_model=64, nhead=4,
                 alpha_roi=1.0, beta_bg=1.0, use_layernorm=True):
        super().__init__()
        self.in_ch = in_ch
        self.d_model = d_model
        self.nhead = nhead
        self.alpha_roi = alpha_roi
        self.beta_bg  = beta_bg

        self.bg_in  = nn.Conv2d(in_ch, d_model, 1)
        self.roi_in = nn.Conv2d(in_ch, d_model, 1)
        self.bg_out  = nn.Conv2d(d_model, in_ch, 1)
        self.roi_out = nn.Conv2d(d_model, in_ch, 1)

        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln_q = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.ln_k = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

        self.bg_residual_mix = nn.Conv2d(d_model, d_model, 1)

    @staticmethod
    def _cosine_gate(bg, roi, eps=1e-6):
        bg_n  = F.normalize(bg,  dim=1, eps=eps)
        roi_n = F.normalize(roi, dim=1, eps=eps)
        sim = (bg_n * roi_n).sum(dim=1, keepdim=True)      # (B,1,H,W) in [-1,1]
        return ((sim + 1.0) * 0.5).clamp(0, 1)             # (B,1,H,W) in [0,1]

    @staticmethod
    def _proj(bg, roi, eps=1e-6):
        roi_n = F.normalize(roi, dim=1, eps=eps)
        coeff = (bg * roi_n).sum(dim=1, keepdim=True)      # (B,1,H,W)
        return coeff * roi_n

    def forward(self, BG, ROI):
        """
        BG, ROI: (B,in_ch,H,W)
        """
        B, _, H, W = BG.shape
        bg  = self.bg_in(BG)     # (B,D,H,W)
        roi = self.roi_in(ROI)   # (B,D,H,W)

        with torch.no_grad():
            gate_rel = self._cosine_gate(bg, roi)          # relevant
        gate_irrel = 1.0 - gate_rel                         # irrelevant

        # tokens
        to_tok = lambda x: x.flatten(2).transpose(1, 2)     # (B,D,H,W)->(B,N,D)
        q = self.ln_q(to_tok(roi))
        k = self.ln_k(to_tok(bg))
        v = k

        attn_out, _ = self.mha(q, k, v, need_weights=False) # (B,N,D)
        attn_map = attn_out.transpose(1, 2).reshape(B, self.d_model, H, W)

        # gains (no mask)
        roi_gain = gate_rel
        bg_gain  = gate_irrel

        # ROI relevant 강조
        roi_enh = roi + self.alpha_roi * (attn_map * roi_gain)

        # BG irrelevant 잔차 강조
        proj = self._proj(bg, roi.detach())
        bg_res = self.bg_residual_mix(bg - proj)
        bg_enh = bg + self.beta_bg * (bg_res * bg_gain)

        # back to in_ch
        BG_out  = self.bg_out(bg_enh)
        ROI_out = self.roi_out(roi_enh)
        return BG_out, ROI_out
    
# v1.13 ( no mask/ window/PE 삭제)
# class ContrastiveEmphasisVCM(nn.Module):
#     """
#     슬라이드 수식대로 구현:
#       F_tilde_ROI = softmax(Q_ROI K_BG^T) V_BG
#       F_tilde_BG  = softmax(-τ Q_BG K_ROI^T) V_ROI
#     + A방식 residual: 원본 BG/ROI에 Δ만 더함.
#     """
#     def __init__(self, in_ch=4, d_model=64, nhead=4,
#                  tau=1.0, alpha_roi_init=0.0, beta_bg_init=0.0,
#                  use_layernorm=True):
#         super().__init__()
#         self.in_ch = in_ch
#         self.d_model = d_model
#         self.nhead = nhead
#         self.tau = tau

#         self.bg_in  = nn.Conv2d(in_ch, d_model, 1)
#         self.roi_in = nn.Conv2d(in_ch, d_model, 1)
#         self.bg_delta_out  = nn.Conv2d(d_model, in_ch, 1)
#         self.roi_delta_out = nn.Conv2d(d_model, in_ch, 1)

#         # ROI <- BG (positve temp)
#         self.mha_roi = nn.MultiheadAttention(d_model, nhead, batch_first=True)
#         # BG <- ROI (negative temp 구현용: Q에 -tau 곱)
#         self.mha_bg  = nn.MultiheadAttention(d_model, nhead, batch_first=True)

#         self.ln_roi_q = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
#         self.ln_bg_q  = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
#         self.ln_bg_k  = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
#         self.ln_roi_k = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

#         self.alpha_roi = nn.Parameter(torch.tensor(alpha_roi_init))
#         self.beta_bg   = nn.Parameter(torch.tensor(beta_bg_init))

#     @staticmethod
#     def _tokens(x):  # (B,D,H,W) -> (B,N,D)
#         return x.flatten(2).transpose(1, 2)

#     def forward(self, BG, ROI):
#         B, _, H, W = BG.shape

#         bg_feat  = self.bg_in(BG)   # (B,D,H,W)
#         roi_feat = self.roi_in(ROI) # (B,D,H,W)

#         # ---------- (1) ROI branch: Q=ROI, K/V=BG ----------
#         q_roi = self.ln_roi_q(self._tokens(roi_feat))
#         k_bg  = self.ln_bg_k(self._tokens(bg_feat))
#         v_bg  = k_bg
#         F_roi_tokens, _ = self.mha_roi(q_roi, k_bg, v_bg, need_weights=False)
#         F_roi = F_roi_tokens.transpose(1, 2).reshape(B, self.d_model, H, W)  # ~F_ROI

#         # ---------- (2) BG branch: Q=BG, K/V=ROI, with -tau ----------
#         q_bg = self.ln_bg_q(self._tokens(bg_feat))
#         k_roi = self.ln_roi_k(self._tokens(roi_feat))
#         v_roi = k_roi

#         # softmax(-τ Q K^T)를 만들기 위해 Q에 -tau 곱
#         q_bg_neg = -self.tau * q_bg
#         F_bg_tokens, _ = self.mha_bg(q_bg_neg, k_roi, v_roi, need_weights=False)
#         F_bg = F_bg_tokens.transpose(1, 2).reshape(B, self.d_model, H, W)  # ~F_BG

#         # ---------- (3) residual로 원본 latent 보정 ----------
#         roi_delta = self.roi_delta_out(F_roi)   # (B,in_ch,H,W)
#         bg_delta  = self.bg_delta_out(F_bg)     # (B,in_ch,H,W)

#         ROI_out = ROI + self.alpha_roi * roi_delta
#         BG_out  = BG  + self.beta_bg  * bg_delta
#         return BG_out, ROI_out
    
# # 나중 버전 (mask/ window/PE 삭제)
# class ContrastiveEmphasisVCM(nn.Module):
#     """
#     - 전역 cross-attn (Q=ROI, K/V=BG), 윈도우/PE 없음
#     - roi_mask로 쿼리/키 범위를 제어
#       ROI: (mask * relevance) 강조
#       BG : ((1-mask) * (1-relevance)) 강조
#     """
#     def __init__(self, in_ch=4, d_model=64, nhead=4,
#                  alpha_roi=1.0, beta_bg=1.0, use_layernorm=True,
#                  restrict_keys=True):
#         super().__init__()
#         self.in_ch = in_ch
#         self.d_model = d_model
#         self.nhead = nhead
#         self.alpha_roi = alpha_roi
#         self.beta_bg  = beta_bg
#         self.restrict_keys = restrict_keys

#         self.bg_in  = nn.Conv2d(in_ch, d_model, 1)
#         self.roi_in = nn.Conv2d(in_ch, d_model, 1)
#         self.bg_out  = nn.Conv2d(d_model, in_ch, 1)
#         self.roi_out = nn.Conv2d(d_model, in_ch, 1)

#         self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
#         self.ln_q = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
#         self.ln_k = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

#         self.bg_residual_mix = nn.Conv2d(d_model, d_model, 1)

#     @staticmethod
#     def _cosine_gate(bg, roi, eps=1e-6):
#         bg_n  = F.normalize(bg,  dim=1, eps=eps)
#         roi_n = F.normalize(roi, dim=1, eps=eps)
#         sim = (bg_n * roi_n).sum(dim=1, keepdim=True)
#         return ((sim + 1.0) * 0.5).clamp(0, 1)

#     @staticmethod
#     def _proj(bg, roi, eps=1e-6):
#         roi_n = F.normalize(roi, dim=1, eps=eps)
#         coeff = (bg * roi_n).sum(dim=1, keepdim=True)
#         return coeff * roi_n

#     def forward(self, BG, ROI, roi_mask):
#         """
#         BG, ROI: (B,in_ch,H,W)
#         roi_mask: (B,1,H,W) in [0,1]  (latent 해상도로 resize하여 전달)
#         """
#         B, _, H, W = BG.shape
#         bg  = self.bg_in(BG)      # (B,D,H,W)
#         roi = self.roi_in(ROI)    # (B,D,H,W)

#         # soft mask (권장: bilinear + 살짝 blur는 호출부에서)
#         qm = roi_mask.to(BG.dtype).clamp(0,1)

#         with torch.no_grad():
#             gate_rel = self._cosine_gate(bg, roi)   # (B,1,H,W)
#         gate_irrel = 1.0 - gate_rel

#         # tokens
#         to_tok = lambda x: x.flatten(2).transpose(1, 2)    # (B,N,D)
#         q = self.ln_q(to_tok(roi))
#         k = self.ln_k(to_tok(bg))
#         v = k

#         # 키 마스킹(ROI 내부만 참조)
#         if self.restrict_keys:
#             k_mask = (qm.flatten(2).squeeze(1) < 0.5)      # (B,N) True=mask
#         else:
#             k_mask = None

#         attn_out, _ = self.mha(q, k, v, need_weights=False, key_padding_mask=k_mask)
#         attn_map = attn_out.transpose(1, 2).reshape(B, self.d_model, H, W)

#         # 쿼리 행 출력 마스킹(ROI 바깥 쿼리 제거)
#         q_mask = qm
#         attn_map = attn_map * q_mask

#         # semantic × spatial 게인
#         roi_gain = qm * gate_rel
#         bg_gain  = (1.0 - qm) * (1.0 - gate_rel)

#         # ROI relevant 강조
#         roi_enh = roi + self.alpha_roi * (attn_map * roi_gain)

#         # BG irrelevant 잔차 강조
#         proj = self._proj(bg, roi.detach())
#         bg_res = self.bg_residual_mix(bg - proj)
#         bg_enh = bg + self.beta_bg * (bg_res * bg_gain)

#         BG_out  = self.bg_out(bg_enh)
#         ROI_out = self.roi_out(roi_enh)
#         return BG_out, ROI_out    
# ==========================================================
# 2) BiDirCrossAttentionVCM
#    - ROI: Q/BG: K,V  AND BG: Q/ROI: K,V  (양방향 cross-attn)
# ==========================================================
# v1.20
# class BiDirCrossAttentionVCM(nn.Module):
#     """
#     입력/출력: BG, ROI 각각 (B, in_ch, H, W) -> (B, in_ch, H, W)
#     내부: d_model로 올려서 양방향 cross-attn, 다시 in_ch로 투영.
#     """
#     def __init__(self, in_ch=4, d_model=64, nhead=4, attn_window=None, use_layernorm=True):
#         super().__init__()
#         self.in_ch = in_ch
#         self.d_model = d_model
#         self.nhead = nhead
#         self.attn_window = attn_window

#         self.bg_in  = nn.Conv2d(in_ch, d_model, 1)
#         self.roi_in = nn.Conv2d(in_ch, d_model, 1)
#         self.bg_out  = nn.Conv2d(d_model, in_ch, 1)
#         self.roi_out = nn.Conv2d(d_model, in_ch, 1)

#         # BG <- ROI : Q=BG, K/V=ROI
#         self.mha_bg_q__roi_kv  = nn.MultiheadAttention(d_model, nhead, batch_first=True)
#         # ROI <- BG : Q=ROI, K/V=BG
#         self.mha_roi_q__bg_kv  = nn.MultiheadAttention(d_model, nhead, batch_first=True)

#         self.ln_bg  = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
#         self.ln_roi = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

#     def _once(self, Qmap, Kmap, ln_q, ln_k, mha, H, W):
#         B, D, _, _ = Qmap.shape
#         pe = _posenc_2d(H, W, D, Qmap.device).unsqueeze(0)  # (1,N,D)
#         def to_tok(x):
#             return x.flatten(2).transpose(1, 2) + pe
#         Q = ln_q(to_tok(Qmap))
#         K = ln_k(to_tok(Kmap))
#         out, _ = mha(Q, K, K, need_weights=False)
#         out = Q + out
#         return out.transpose(1, 2).reshape(B, D, H, W)

#     def _once_window(self, Qmap, Kmap, ln_q, ln_k, mha, H, W, win):
#         B, D = Qmap.shape[:2]
#         pe = _posenc_2d(win, win, D, Qmap.device).unsqueeze(0)
#         Qp = _Windower.tile(Qmap, win)
#         Kp = _Windower.tile(Kmap, win)
#         outs = []
#         for q, k in zip(Qp, Kp):
#             def to_tok(x):
#                 return x.flatten(2).transpose(1, 2) + pe
#             Q = ln_q(to_tok(q))
#             K = ln_k(to_tok(k))
#             o, _ = mha(Q, K, K, need_weights=False)
#             o = Q + o
#             outs.append(o.transpose(1, 2).reshape(B, D, win, win))
#         return _Windower.merge(outs, H, W, win)

#     def forward(self, BG, ROI):
#         B, _, H, W = BG.shape
#         bg  = self.bg_in(BG)
#         roi = self.roi_in(ROI)

#         if self.attn_window is None:
#             # BG <- ROI
#             bg_f  = self._once(bg,  roi, self.ln_bg, self.ln_roi, self.mha_bg_q__roi_kv, H, W)
#             # ROI <- BG
#             roi_f = self._once(roi, bg,  self.ln_roi, self.ln_bg,  self.mha_roi_q__bg_kv, H, W)
#         else:
#             win = self.attn_window
#             assert H % win == 0 and W % win == 0
#             # BG <- ROI
#             bg_f  = self._once_window(bg,  roi, self.ln_bg, self.ln_roi, self.mha_bg_q__roi_kv, H, W, win)
#             # ROI <- BG
#             roi_f = self._once_window(roi, bg,  self.ln_roi, self.ln_bg,  self.mha_roi_q__bg_kv, H, W, win)

#         BG_out  = self.bg_out(bg_f)
#         ROI_out = self.roi_out(roi_f)
#         return BG_out, ROI_out
###############################################################################
# # v1.21
# -------------------------
# 공통 베이스 (윈도우/PE 없음)
# -------------------------
class _BiDirBase(nn.Module):
    def __init__(self, in_ch=4, d_model=64, nhead=4, use_layernorm=True):
        super().__init__()
        self.in_ch = in_ch
        self.d_model = d_model
        self.nhead = nhead

        # 1x1 proj
        self.bg_in  = nn.Conv2d(in_ch, d_model, 1)
        self.roi_in = nn.Conv2d(in_ch, d_model, 1)
        self.bg_out  = nn.Conv2d(d_model, in_ch, 1)
        self.roi_out = nn.Conv2d(d_model, in_ch, 1)

        # BG <- ROI : Q=BG, K/V=ROI
        self.mha_bg_q__roi_kv  = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        # ROI <- BG : Q=ROI, K/V=BG
        self.mha_roi_q__bg_kv  = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.ln_bg_q  = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.ln_roi_k = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.ln_roi_q = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.ln_bg_k  = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

    @staticmethod
    def _tokens(x):  # (B,D,H,W) -> (B,N,D)
        return x.flatten(2).transpose(1, 2)

# -------------------------------------------------
# 1) 마스크 없이: 전역 양방향 cross-attention (PE/윈도우 없음)
# -------------------------------------------------
# v1.21
class BiDirCrossAttentionVCM(_BiDirBase):
    """
    출력:
      BG_out = BG + CrossAttn(BG<-ROI)
      ROI_out = ROI + CrossAttn(ROI<-BG)
    """
    def forward(self, BG, ROI):
        B, _, H, W = BG.shape

        bg  = self.bg_in(BG)     # (B,D,H,W)
        roi = self.roi_in(ROI)

        # BG <- ROI
        q_bg = self.ln_bg_q(self._tokens(bg))
        k_r  = self.ln_roi_k(self._tokens(roi))
        v_r  = k_r
        bg_attn, _ = self.mha_bg_q__roi_kv(q_bg, k_r, v_r, need_weights=False)
        bg_f = (q_bg + bg_attn).transpose(1, 2).reshape(B, self.d_model, H, W)

        # ROI <- BG
        q_r  = self.ln_roi_q(self._tokens(roi))
        k_bg = self.ln_bg_k(self._tokens(bg))
        v_bg = k_bg
        roi_attn, _ = self.mha_roi_q__bg_kv(q_r, k_bg, v_bg, need_weights=False)
        roi_f = (q_r + roi_attn).transpose(1, 2).reshape(B, self.d_model, H, W)

        BG_out  = self.bg_out(bg_f)
        ROI_out = self.roi_out(roi_f)
        return BG_out, ROI_out

##############################################################################
# # v1.22
# class _BiDirBase(nn.Module):
#     def __init__(self, in_ch=4, d_model=64, nhead=4, use_layernorm=True):
#         super().__init__()
#         self.in_ch = in_ch
#         self.d_model = d_model
#         self.nhead = nhead

#         # 1x1 proj
#         self.bg_in  = nn.Conv2d(in_ch, d_model, 1)
#         self.roi_in = nn.Conv2d(in_ch, d_model, 1)
#         self.bg_out  = nn.Conv2d(d_model, in_ch, 1)
#         self.roi_out = nn.Conv2d(d_model, in_ch, 1)

#         # BG <- ROI : Q=BG, K/V=ROI
#         self.mha_bg_q__roi_kv  = nn.MultiheadAttention(d_model, nhead, batch_first=True)
#         # ROI <- BG : Q=ROI, K/V=BG
#         self.mha_roi_q__bg_kv  = nn.MultiheadAttention(d_model, nhead, batch_first=True)

#         self.ln_bg_q  = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
#         self.ln_roi_k = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
#         self.ln_roi_q = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
#         self.ln_bg_k  = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

#     @staticmethod
#     def _tokens(x):  # (B,D,H,W) -> (B,N,D)
#         return x.flatten(2).transpose(1, 2)


# class BiDirCrossAttentionVCM(_BiDirBase):
#     """
#     출력 (A방식, residual):
#       BG_out  = BG  + alpha_bg  * Δ_BG       where Δ_BG  = CrossAttn(BG <- ROI)
#       ROI_out = ROI + alpha_roi * Δ_ROI      where Δ_ROI = CrossAttn(ROI <- BG)
#     """
#     def __init__(self, in_ch=4, d_model=64, nhead=4,
#                  alpha_bg_init=0.0, alpha_roi_init=0.0, use_layernorm=True):
#         super().__init__(in_ch=in_ch, d_model=d_model, nhead=nhead,
#                          use_layernorm=use_layernorm)
#         # cross-attention 영향 스케일 (처음엔 0 근처로 시작 → 안정)
#         self.alpha_bg  = nn.Parameter(torch.tensor(alpha_bg_init))
#         self.alpha_roi = nn.Parameter(torch.tensor(alpha_roi_init))

#     def forward(self, BG, ROI):
#         """
#         BG, ROI: (B, in_ch, H, W)
#         """
#         B, _, H, W = BG.shape

#         # d_model 공간으로 올리기
#         bg_feat  = self.bg_in(BG)     # (B,D,H,W)
#         roi_feat = self.roi_in(ROI)   # (B,D,H,W)

#         # (B,D,H,W) -> (B,N,D)
#         bg_tok  = self._tokens(bg_feat)
#         roi_tok = self._tokens(roi_feat)

#         # -------------------------------------------------
#         # BG branch: BG <- ROI  (Q=BG, K/V=ROI)
#         # -------------------------------------------------
#         q_bg = self.ln_bg_q(bg_tok)
#         k_r  = self.ln_roi_k(roi_tok)
#         v_r  = k_r
#         # MultiheadAttention 출력 = CrossAttn(Q=BG, K/V=ROI)
#         bg_attn_tok, _ = self.mha_bg_q__roi_kv(q_bg, k_r, v_r, need_weights=False)
#         # d_model feature로 reshape
#         bg_attn_d = bg_attn_tok.transpose(1, 2).reshape(B, self.d_model, H, W)
#         # in_ch로 내려서 Δ_BG 만들기
#         delta_BG = self.bg_out(bg_attn_d)                    # (B,in_ch,H,W)

#         # -------------------------------------------------
#         # ROI branch: ROI <- BG (Q=ROI, K/V=BG)
#         # -------------------------------------------------
#         q_r  = self.ln_roi_q(roi_tok)
#         k_bg = self.ln_bg_k(bg_tok)
#         v_bg = k_bg
#         roi_attn_tok, _ = self.mha_roi_q__bg_kv(q_r, k_bg, v_bg, need_weights=False)
#         roi_attn_d = roi_attn_tok.transpose(1, 2).reshape(B, self.d_model, H, W)
#         delta_ROI = self.roi_out(roi_attn_d)                  # (B,in_ch,H,W)

#         # -------------------------------------------------
#         # ✅ 원본 latent에 residual로 더하기 (A방식)
#         # -------------------------------------------------
#         BG_out  = BG  + self.alpha_bg  * delta_BG
#         ROI_out = ROI + self.alpha_roi * delta_ROI
#         return BG_out, ROI_out


# 다음버전~
# # -------------------------------------------------
# # 2) 마스크 사용: 전역 양방향 cross-attention + 공간 제어
# #    roi_mask: (B,1,H,W) in [0,1]
# #    - BG<-ROI: ROI 영역만 키/밸류로 사용 (선택)
# #    - ROI<-BG: BG 영역만 키/밸류로 사용 (선택)
# #    쿼리 행 출력도 해당 마스크로 곱해 바깥 쿼리를 억제
# # -------------------------------------------------
# class BiDirCrossAttentionVCM(_BiDirBase):
#     def __init__(self, *args, restrict_keys=True, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.restrict_keys = restrict_keys

#     def forward(self, BG, ROI, roi_mask, bg_mask=None):
#         """
#         roi_mask: (B,1,H,W) — ROI 쿼리/키 영역
#         bg_mask : (B,1,H,W) — BG 쿼리/키 영역 (없으면 ~roi_mask로 간주 가능)
#         """
#         B, _, H, W = BG.shape
#         bg  = self.bg_in(BG)
#         roi = self.roi_in(ROI)

#         qm_roi = roi_mask.to(BG.dtype).clamp(0,1)
#         if bg_mask is None:
#             qm_bg = (1.0 - qm_roi)
#         else:
#             qm_bg = bg_mask.to(BG.dtype).clamp(0,1)

#         # -------- BG <- ROI --------
#         q_bg = self.ln_bg_q(self._tokens(bg))
#         k_r  = self.ln_roi_k(self._tokens(roi))
#         v_r  = k_r

#         if self.restrict_keys:
#             k_mask_r = (qm_roi.flatten(2).squeeze(1) < 0.5)   # True=mask
#         else:
#             k_mask_r = None

#         bg_attn, _ = self.mha_bg_q__roi_kv(q_bg, k_r, v_r, need_weights=False, key_padding_mask=k_mask_r)
#         bg_f = (q_bg + bg_attn).transpose(1, 2).reshape(B, self.d_model, H, W)
#         # 쿼리(=BG) 출력 마스킹
#         bg_f = bg_f * qm_bg

#         # -------- ROI <- BG --------
#         q_r  = self.ln_roi_q(self._tokens(roi))
#         k_bg = self.ln_bg_k(self._tokens(bg))
#         v_bg = k_bg

#         if self.restrict_keys:
#             k_mask_bg = (qm_bg.flatten(2).squeeze(1) < 0.5)
#         else:
#             k_mask_bg = None

#         roi_attn, _ = self.mha_roi_q__bg_kv(q_r, k_bg, v_bg, need_weights=False, key_padding_mask=k_mask_bg)
#         roi_f = (q_r + roi_attn).transpose(1, 2).reshape(B, self.d_model, H, W)
#         roi_f = roi_f * qm_roi

#         BG_out  = self.bg_out(bg_f)
#         ROI_out = self.roi_out(roi_f)
#         return BG_out, ROI_out