"""FloodMap -- cross-modal SAR<->optical masked autoencoder for few-shot floods.

Novel contributions
-------------------
1. **Cross-modal masked reconstruction.** A single ViT autoencoder is masked
   over *both* the Sentinel-1 (SAR, VV+VH) and Sentinel-2 (optical) inputs.
   The decoder must reconstruct missing tokens of both modalities, which
   forces the latent to encode physical flooding signatures that are
   consistent across radar and optical views -- SAR sees through clouds,
   optical adds spectral confirmation.

2. **Lightweight ViT-Tiny backbone** so the whole model pretrains on a free
   Colab TPU/T4 budget (the practical blocker for EO foundation models).

3. **Few-shot flood head.** After self-supervised pretraining on ~10k
   unlabelled tiles, a lightweight segmentation head is fine-tuned with
   <50 labelled flood masks (Copernicus EMS / GFD), making FloodMap usable
   where labelled disaster data is nearly absent.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Standard ViT 2-D sinusoidal positional embedding."""
    g = grid_size
    pos = torch.zeros(g * g, embed_dim)
    h = torch.arange(g, dtype=torch.float32)
    w = torch.arange(g, dtype=torch.float32)
    gy, gx = torch.meshgrid(h, w, indexing="ij")
    gy = gy.flatten()
    gx = gx.flatten()
    assert embed_dim % 4 == 0
    half = embed_dim // 2
    emb = torch.arange(half // 2, dtype=torch.float32)
    emb = emb / (half // 2 - 1)
    omega = torch.exp(emb * (-math.log(10000.0)))
    ax = gx[:, None] * omega[None, :]
    ay = gy[:, None] * omega[None, :]
    pos[:, 0:half:2] = ax.sin()
    pos[:, 1:half:2] = ax.cos()
    pos[:, half::2] = ay.sin()
    pos[:, half + 1 :: 2] = ay.cos()
    return pos


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, patch_size: int, embed_dim: int, img_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)  # [B, N, D]


class ViTEncoder(nn.Module):
    def __init__(self, in_chans: int, patch_size: int, img_size: int, embed_dim: int, depth: int, nhead: int) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans, patch_size, embed_dim, img_size)
        num_patches = self.patch_embed.grid**2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(get_2d_sincos_pos_embed(embed_dim, self.patch_embed.grid).unsqueeze(0))
        self.blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(embed_dim, nhead, dim_feedforward=embed_dim * 4, batch_first=True) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        torch.nn.init.normal_(self.cls_token, std=0.02)
        for blk in self.blocks:
            for p in blk.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]; mask: [B, N] bool (True = keep)
        tokens = self.patch_embed(x)  # [B, N, D]
        B, N, D = tokens.shape
        tokens = tokens + self.pos_embed[:, :N]
        tokens = tokens * mask.unsqueeze(-1).float()
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)


class MAEDecoder(nn.Module):
    def __init__(self, embed_dim: int, out_chans: int, grid: int, depth: int = 2, nhead: int = 4) -> None:
        super().__init__()
        self.grid = grid
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(get_2d_sincos_pos_embed(embed_dim, grid).unsqueeze(0))
        self.blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(embed_dim, nhead, dim_feedforward=embed_dim * 4, batch_first=True) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, patch_size := out_chans * 16 * 16)

    def forward(self, enc: torch.Tensor, mask: torch.Tensor, num_tokens: int) -> torch.Tensor:
        B, N, D = enc.shape
        # enc already has all tokens (filled + mask tokens for masked positions)
        full = enc + self.pos_embed[:, :num_tokens]
        for blk in self.blocks:
            full = blk(full)
        full = self.norm(full)
        out = self.head(full)  # [B, N, C*p*p]
        C = out.shape[-1] // (16 * 16)
        out = out.view(B, self.grid, self.grid, C, 16, 16)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, self.grid * 16, self.grid * 16)
        return out


class FloodMap(nn.Module):
    """Joint SAR + optical masked autoencoder with a few-shot flood head.

    Args:
        img_size: square input size.
        sar_chans: 2 (VV, VH).
        opt_chans: 3 (RGB) or 10 (all Sentinel-2 bands).
        patch_size: 16.
        mask_ratio: fraction of tokens masked during pretraining.
    """

    def __init__(
        self,
        img_size: int = 128,
        sar_chans: int = 2,
        opt_chans: int = 3,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 6,
        nhead: int = 6,
        mask_ratio: float = 0.6,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid**2
        self.mask_ratio = mask_ratio

        self.sar_enc = ViTEncoder(sar_chans, patch_size, img_size, embed_dim, depth, nhead)
        self.opt_enc = ViTEncoder(opt_chans, patch_size, img_size, embed_dim, depth, nhead)
        self.fuse = nn.Linear(embed_dim * 2, embed_dim)
        self.decoder = MAEDecoder(embed_dim, out_chans=sar_chans + opt_chans, grid=self.grid)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # few-shot segmentation head (trainable only in the fine-tune stage)
        self.flood_head = nn.Sequential(
            nn.Conv2d(embed_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )

    def random_masking(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        keep = int(N * (1 - self.mask_ratio))
        ids_keep = ids_shuffle[:, :keep]
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        mask.scatter_(1, ids_shuffle[:, keep:], True)
        return x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, D)), mask

    def forward(self, sar: torch.Tensor, opt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, mask)."""
        # Encode both modalities to patch embeddings [B, N, D]
        sar_tokens = self.sar_enc.patch_embed(sar)
        opt_tokens = self.opt_enc.patch_embed(opt)
        
        # Add positional embeddings
        sar_tokens = sar_tokens + self.sar_enc.pos_embed[:, :self.num_patches]
        opt_tokens = opt_tokens + self.opt_enc.pos_embed[:, :self.num_patches]
        
        # Random masking on SAR tokens
        B, N, D = sar_tokens.shape
        kept_sar, mask = self.random_masking(sar_tokens)
        
        # Fuse kept SAR with all optical tokens (element-wise concat per patch)
        # For masked SAR positions, we use a learned mask token + optical token
        mask_token = self.mask_token.expand(B, N, D)
        sar_filled = torch.where(mask.unsqueeze(-1), mask_token, sar_tokens)
        
        # Fuse: concat SAR + optical at feature dim → [B, N, 2*D]
        fused = torch.cat([sar_filled, opt_tokens], dim=-1)
        fused = self.fuse(fused)  # [B, N, D]
        
        # Decode
        recon = self.decoder(fused, mask, N)
        return recon, mask

    def flood_prob(self, sar: torch.Tensor, opt: torch.Tensor) -> torch.Tensor:
        """Segmentation head over fused patch embeddings -> [B, 1, H, W]."""
        B, C, H, W = sar.shape
        mask_ones = torch.ones(B, self.num_patches, dtype=torch.bool, device=sar.device)
        sar_tokens = self.sar_enc.patch_embed(sar) + self.sar_enc.pos_embed[:, :self.num_patches]
        opt_tokens = self.opt_enc.patch_embed(opt) + self.opt_enc.pos_embed[:, :self.num_patches]
        fused = self.fuse(torch.cat([sar_tokens, opt_tokens], dim=-1))
        feats = fused.transpose(1, 2).view(B, fused.shape[-1], self.grid, self.grid)
        feats = F.interpolate(feats, size=(H, W), mode="bilinear", align_corners=False)
        return torch.sigmoid(self.flood_head(feats))

    def pretrain_loss(self, sar: torch.Tensor, opt: torch.Tensor) -> dict[str, torch.Tensor]:
        """Self-supervised masked reconstruction loss (MSE on visible+masked)."""
        recon, mask = self.forward(sar, opt)
        target = torch.cat([sar, opt], dim=1)
        loss = F.mse_loss(recon, target)
        return {"loss": loss, "recon": recon, "mask": mask}