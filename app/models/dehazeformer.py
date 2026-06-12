"""
app/models/dehazeformer.py
==========================
DehazeFormer Architecture Implementation
-----------------------------------------
DehazeFormer is a Transformer-based image dehazing network proposed in:
  "Vision Transformers for Single Image Dehazing" (Song et al., 2023)

Architecture overview:
  Input (hazy image)
      │
      ▼
  PatchEmbed  ──► splits image into overlapping patches and projects to C dims
      │
      ▼
  Encoder (3 stages, each with N DehazeFormer Blocks + PatchMerging downsample)
      │
      ▼
  Bottleneck  (DehazeFormer Blocks at lowest resolution)
      │
      ▼
  Decoder (3 stages, each with PatchExpand upsample + N DehazeFormer Blocks)
      │
      ▼
  PatchUnEmbed + Conv head  ──► residual added to hazy input
      │
      ▼
  Output (clean image)

Each DehazeFormer Block contains:
  1. RLN  (Rescaled Layer Norm) — normalises features
  2. SKFF (Selective Kernel Feature Fusion) — multi-scale attention
  3. MDTA (Multi-Dconv Head Transposed Attention) — efficient self-attention
  4. GDFN (Gated-Dconv Feed-forward Network) — gated MLP

This file is intentionally verbose with comments for FYP explanation purposes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────────────────────
# 1. RESCALED LAYER NORM (RLN)
#    Standard LayerNorm but with a learnable rescaling factor applied to the
#    weight. This helps stabilise training on low-light / hazy images.
# ─────────────────────────────────────────────────────────────────────────────
class RLN(nn.Module):
    """
    Rescaled Layer Normalisation.
    Adds a learnable scalar `alpha` that rescales the LN weight,
    giving the network more flexibility to control feature magnitudes.
    """
    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super().__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        # Learnable affine parameters (same as standard LN)
        self.weight = nn.Parameter(torch.ones((1, 1, dim)))
        self.bias   = nn.Parameter(torch.zeros((1, 1, dim)))

        # Extra rescaling factor — the key difference from standard LN
        self.var    = nn.Parameter(torch.ones((1, 1, dim)))

    def forward(self, x):
        # x: (B, N, C)  where N = number of tokens, C = channels
        std = x.var(dim=-1, keepdim=True, unbiased=False).sqrt()
        mean = x.mean(dim=-1, keepdim=True)

        # Normalise
        x_norm = (x - mean) / (std + self.eps)

        # Rescale: var acts as an extra learnable scale on top of weight
        if self.detach_grad:
            factor = (self.var / (std.detach() + self.eps))
        else:
            factor = (self.var / (std + self.eps))

        weight = factor * self.weight
        return x_norm * weight + self.bias


# ─────────────────────────────────────────────────────────────────────────────
# 2. MULTI-DCONV HEAD TRANSPOSED ATTENTION (MDTA)
#    Instead of standard Q·K^T attention (O(N²)), MDTA computes attention
#    across the channel dimension (O(C²)), making it efficient for high-res
#    images. Depth-wise convolutions capture local spatial context.
# ─────────────────────────────────────────────────────────────────────────────
class MDTA(nn.Module):
    """
    Multi-Dconv Head Transposed Attention.

    Key idea: transpose the attention matrix from (N×N) to (C×C).
    This reduces complexity from O(N²C) to O(NC²), which is much cheaper
    when N (number of pixels) >> C (number of channels).
    """
    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        # Temperature parameter scales the attention logits
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Project input to Q, K, V — 1×1 conv followed by 3×3 depth-wise conv
        self.qkv      = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(
            channels * 3, channels * 3,
            kernel_size=3, stride=1, padding=1,
            groups=channels * 3, bias=False          # depth-wise
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # Compute Q, K, V with depth-wise conv for local context
        qkv = self.qkv_dwconv(self.qkv(x))          # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)               # each (B, C, H, W)

        # Reshape for multi-head: (B, heads, C//heads, H*W)
        head_dim = C // self.num_heads
        q = q.reshape(B, self.num_heads, head_dim, H * W)
        k = k.reshape(B, self.num_heads, head_dim, H * W)
        v = v.reshape(B, self.num_heads, head_dim, H * W)

        # L2-normalise Q and K (stabilises training)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Transposed attention: (C//heads × C//heads) instead of (N × N)
        attn = (q @ k.transpose(-2, -1)) * self.temperature   # (B, heads, C//h, C//h)
        attn = attn.softmax(dim=-1)

        # Apply attention to V
        out = (attn @ v)                             # (B, heads, C//h, H*W)
        out = out.reshape(B, C, H, W)
        return self.project_out(out)


# ─────────────────────────────────────────────────────────────────────────────
# 3. GATED DCONV FEED-FORWARD NETWORK (GDFN)
#    A gated variant of the standard FFN. The gate mechanism (element-wise
#    product of two parallel branches) allows the network to selectively
#    pass or suppress features — useful for separating haze from content.
# ─────────────────────────────────────────────────────────────────────────────
class GDFN(nn.Module):
    """
    Gated Depth-wise Convolutional Feed-Forward Network.

    Structure:
        x → Conv1×1 → split into (gate, features)
                gate   → depth-wise 3×3 → GELU
                features → depth-wise 3×3
        output = gate ⊙ features → Conv1×1
    """
    def __init__(self, channels, expansion_factor=2.66):
        super().__init__()
        hidden = int(channels * expansion_factor)

        # Input projection — outputs 2× hidden for the gating split
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=False)

        # Depth-wise 3×3 convolutions for both branches
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2,
            kernel_size=3, stride=1, padding=1,
            groups=hidden * 2, bias=False
        )

        # Output projection back to original channels
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.project_in(x)          # (B, 2*hidden, H, W)
        x = self.dwconv(x)              # local context via depth-wise conv
        x1, x2 = x.chunk(2, dim=1)     # split into two branches
        # Gating: GELU activation on one branch, multiply with the other
        x = F.gelu(x1) * x2
        return self.project_out(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4. DEHAZEFORMER BLOCK
#    One complete transformer block combining RLN + MDTA + GDFN with
#    residual connections. This is the core repeating unit of the network.
# ─────────────────────────────────────────────────────────────────────────────
class DehazeFormerBlock(nn.Module):
    """
    One DehazeFormer Transformer Block.

    Forward pass:
        x_norm = RLN(x)                         # normalise
        x = x + MDTA(x_norm)                    # attention + residual
        x_norm = RLN(x)                         # normalise again
        x = x + GDFN(x_norm)                    # feed-forward + residual
    """
    def __init__(self, channels, num_heads, expansion_factor=2.66):
        super().__init__()
        self.norm1 = RLN(channels)
        self.norm2 = RLN(channels)
        self.attn  = MDTA(channels, num_heads)
        self.ffn   = GDFN(channels, expansion_factor)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # ── Attention branch ──────────────────────────────────────────────
        # Flatten spatial dims for LayerNorm: (B, H*W, C)
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm1(x_flat)
        # Reshape back to (B, C, H, W) for conv-based attention
        x_norm = x_norm.transpose(1, 2).reshape(B, C, H, W)
        x = x + self.attn(x_norm)

        # ── Feed-forward branch ───────────────────────────────────────────
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm2(x_flat)
        x_norm = x_norm.transpose(1, 2).reshape(B, C, H, W)
        x = x + self.ffn(x_norm)

        return x


# ─────────────────────────────────────────────────────────────────────────────
# 5. PATCH EMBED / UNEMBED
#    Converts between pixel space and token space.
# ─────────────────────────────────────────────────────────────────────────────
class PatchEmbed(nn.Module):
    """
    Overlapping patch embedding using a 3×3 conv.
    Maps (B, 3, H, W) → (B, C, H, W) feature map.
    """
    def __init__(self, in_channels=3, embed_dim=32):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.proj(x)


class PatchUnEmbed(nn.Module):
    """
    Inverse of PatchEmbed — maps feature map back to image space.
    Maps (B, C, H, W) → (B, 3, H, W).
    """
    def __init__(self, embed_dim=32, out_channels=3):
        super().__init__()
        self.proj = nn.Conv2d(embed_dim, out_channels,
                              kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# 6. DOWNSAMPLE / UPSAMPLE
#    Encoder uses strided conv to halve spatial resolution.
#    Decoder uses pixel-shuffle to double spatial resolution.
# ─────────────────────────────────────────────────────────────────────────────
class Downsample(nn.Module):
    """2× spatial downsampling via strided 2×2 conv, doubles channels."""
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2)   # (B, C//2, H, W) → (B, 2C, H/2, W/2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """2× spatial upsampling via pixel-shuffle, halves channels."""
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2)     # (B, 4C, H, W) → (B, C, 2H, 2W)
        )

    def forward(self, x):
        return self.body(x)


# ─────────────────────────────────────────────────────────────────────────────
# 7. DEHAZEFORMER — FULL U-NET STYLE NETWORK
#    Encoder-Bottleneck-Decoder with skip connections.
#    Default config matches DehazeFormer-S (small) for fast inference.
# ─────────────────────────────────────────────────────────────────────────────
class DehazeFormer(nn.Module):
    """
    Full DehazeFormer network.

    U-Net style encoder-decoder:
      Encoder:    3 stages, each doubles channels and halves resolution
      Bottleneck: transformer blocks at lowest resolution
      Decoder:    3 stages, each halves channels and doubles resolution
                  with skip connections from encoder

    Default config = DehazeFormer-S:
      embed_dim=24, depths=[1,1,2], num_heads=[1,2,4], mlp_ratio=2.66
    """
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        embed_dim=24,
        depths=(1, 1, 2),          # blocks per encoder stage
        num_heads=(1, 2, 4),       # attention heads per stage
        mlp_ratio=2.66,
        **kwargs
    ):
        super().__init__()

        # ── Patch embedding ───────────────────────────────────────────────
        self.patch_embed = PatchEmbed(inp_channels, embed_dim)

        # ── Encoder ───────────────────────────────────────────────────────
        # Stage 1: embed_dim channels, full resolution
        self.encoder1 = nn.Sequential(*[
            DehazeFormerBlock(embed_dim, num_heads[0], mlp_ratio)
            for _ in range(depths[0])
        ])
        self.down1 = Downsample(embed_dim)          # → embed_dim*2, H/2

        # Stage 2: embed_dim*2 channels, half resolution
        self.encoder2 = nn.Sequential(*[
            DehazeFormerBlock(embed_dim * 2, num_heads[1], mlp_ratio)
            for _ in range(depths[1])
        ])
        self.down2 = Downsample(embed_dim * 2)      # → embed_dim*4, H/4

        # Stage 3: embed_dim*4 channels, quarter resolution
        self.encoder3 = nn.Sequential(*[
            DehazeFormerBlock(embed_dim * 4, num_heads[2], mlp_ratio)
            for _ in range(depths[2])
        ])
        self.down3 = Downsample(embed_dim * 4)      # → embed_dim*8, H/8

        # ── Bottleneck ────────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(*[
            DehazeFormerBlock(embed_dim * 8, num_heads[2], mlp_ratio)
            for _ in range(depths[2])
        ])

        # ── Decoder ───────────────────────────────────────────────────────
        self.up3 = Upsample(embed_dim * 8)          # → embed_dim*4, H/4
        # After concat with skip: embed_dim*8 → embed_dim*4
        self.reduce3 = nn.Conv2d(embed_dim * 8, embed_dim * 4, 1, bias=False)
        self.decoder3 = nn.Sequential(*[
            DehazeFormerBlock(embed_dim * 4, num_heads[2], mlp_ratio)
            for _ in range(depths[2])
        ])

        self.up2 = Upsample(embed_dim * 4)          # → embed_dim*2, H/2
        self.reduce2 = nn.Conv2d(embed_dim * 4, embed_dim * 2, 1, bias=False)
        self.decoder2 = nn.Sequential(*[
            DehazeFormerBlock(embed_dim * 2, num_heads[1], mlp_ratio)
            for _ in range(depths[1])
        ])

        self.up1 = Upsample(embed_dim * 2)          # → embed_dim, H
        self.reduce1 = nn.Conv2d(embed_dim * 2, embed_dim, 1, bias=False)
        self.decoder1 = nn.Sequential(*[
            DehazeFormerBlock(embed_dim, num_heads[0], mlp_ratio)
            for _ in range(depths[0])
        ])

        # ── Output head ───────────────────────────────────────────────────
        self.patch_unembed = PatchUnEmbed(embed_dim, out_channels)

    def forward(self, x):
        """
        x: (B, 3, H, W) — normalised hazy input image [0, 1]
        returns: (B, 3, H, W) — dehazed image [0, 1]
        """
        # Remember input for residual connection
        inp = x

        # ── Encode ────────────────────────────────────────────────────────
        x = self.patch_embed(x)          # (B, C, H, W)

        enc1 = self.encoder1(x)          # (B, C, H, W)
        x    = self.down1(enc1)          # (B, 2C, H/2, W/2)

        enc2 = self.encoder2(x)          # (B, 2C, H/2, W/2)
        x    = self.down2(enc2)          # (B, 4C, H/4, W/4)

        enc3 = self.encoder3(x)          # (B, 4C, H/4, W/4)
        x    = self.down3(enc3)          # (B, 8C, H/8, W/8)

        # ── Bottleneck ────────────────────────────────────────────────────
        x = self.bottleneck(x)           # (B, 8C, H/8, W/8)

        # ── Decode with skip connections ──────────────────────────────────
        x = self.up3(x)                  # (B, 4C, H/4, W/4)
        x = self.reduce3(torch.cat([x, enc3], dim=1))
        x = self.decoder3(x)

        x = self.up2(x)                  # (B, 2C, H/2, W/2)
        x = self.reduce2(torch.cat([x, enc2], dim=1))
        x = self.decoder2(x)

        x = self.up1(x)                  # (B, C, H, W)
        x = self.reduce1(torch.cat([x, enc1], dim=1))
        x = self.decoder1(x)

        # ── Output: residual learning (predict the clean residual) ────────
        out = self.patch_unembed(x)      # (B, 3, H, W)
        return torch.clamp(inp + out, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 8. MODEL FACTORY
#    Convenience functions to build different size variants.
# ─────────────────────────────────────────────────────────────────────────────

def build_dehazeformer_s():
    """DehazeFormer-S (Small) — fast, good for FYP demo."""
    return DehazeFormer(
        embed_dim=24,
        depths=(1, 1, 2),
        num_heads=(1, 2, 4),
        mlp_ratio=2.66
    )


def build_dehazeformer_b():
    """DehazeFormer-B (Base) — balanced accuracy/speed."""
    return DehazeFormer(
        embed_dim=32,
        depths=(2, 2, 4),
        num_heads=(1, 2, 4),
        mlp_ratio=2.66
    )


def build_dehazeformer_l():
    """DehazeFormer-L (Large) — highest accuracy, slower."""
    return DehazeFormer(
        embed_dim=48,
        depths=(4, 4, 8),
        num_heads=(2, 4, 8),
        mlp_ratio=2.66
    )
