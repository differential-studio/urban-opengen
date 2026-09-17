"""
unet.py

A plain conditional UNet for the inpainter, self-contained so the project does not pick
up a diffusion library dependency for a 4-channel 128 px model.

Input  [B, 9, S, S]  = noisy fields (4) + context fields with the hole zeroed (4) + mask (1)
Output [B, 4, S, S]  = v-prediction for the four fields

Conditioning enters every residual block through adaptive group norm (scale and shift
from one embedding vector), and the embedding is the sum of three parts: the sinusoidal
timestep, the sinusoidal log2(scale) token, and an MLP over the 12-dim metric vector
(six values, six given/withheld flags). Attention runs at the two coarsest resolutions.

Default size (base 64, mults 1 2 3 4) is about 30M parameters: hours, not days, on a
consumer GPU at 128 px.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def sinusoidal(x: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """[B] -> [B, dim]"""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=x.device, dtype=torch.float32) / half)
    args = x.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], 1)


def _gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, max(1, ch // 8)), ch)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb: int, dropout: float = 0.0):
        super().__init__()
        self.n1 = _gn(cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.ada = nn.Linear(emb, cout * 2)
        self.n2 = _gn(cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.ada(emb).unsqueeze(-1).unsqueeze(-1).chunk(2, 1)
        h = self.n2(h) * (1 + scale) + shift
        h = self.c2(self.drop(F.silu(h)))
        return self.skip(x) + h


class Attention(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = _gn(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.out = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        q, k, v = (t.transpose(-1, -2) for t in (q, k, v))          # [b, heads, hw, d]
        if hasattr(F, "scaled_dot_product_attention"):
            o = F.scaled_dot_product_attention(q, k, v)
        else:
            a = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            o = a.softmax(-1) @ v
        o = o.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.out(o)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 9, out_ch: int = 4, base: int = 64, mults=(1, 2, 3, 4),
                 blocks: int = 2, attn_levels=(2, 3), cond_dim: int = 12, dropout: float = 0.0, heads: int = 4):
        super().__init__()
        self.cfg = dict(in_ch=in_ch, out_ch=out_ch, base=base, mults=list(mults), blocks=blocks,
                        attn_levels=list(attn_levels), cond_dim=cond_dim, dropout=dropout, heads=heads)
        # recompute block activations in the backward pass instead of storing them: roughly a
        # third of the memory for about 30% more compute. Set by train.py; sampling never needs it.
        self.grad_ckpt = False
        emb = base * 4
        self.emb = emb
        self.t_mlp = nn.Sequential(nn.Linear(base, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.s_mlp = nn.Sequential(nn.Linear(base, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.c_mlp = nn.Sequential(nn.Linear(cond_dim, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.null_cond = nn.Parameter(torch.zeros(emb))

        self.inp = nn.Conv2d(in_ch, base, 3, padding=1)
        chs = [base * m for m in mults]
        self.down = nn.ModuleList()
        skip_chs = [base]
        c = base
        for lvl, co in enumerate(chs):
            for _ in range(blocks):
                mods = [ResBlock(c, co, emb, dropout)]
                if lvl in attn_levels:
                    mods.append(Attention(co, heads))
                self.down.append(nn.ModuleList(mods))
                c = co
                skip_chs.append(c)
            if lvl < len(chs) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(c, c, 3, stride=2, padding=1)]))
                skip_chs.append(c)
        self.mid = nn.ModuleList([ResBlock(c, c, emb, dropout), Attention(c, heads), ResBlock(c, c, emb, dropout)])
        self.up = nn.ModuleList()
        for lvl, co in reversed(list(enumerate(chs))):
            for _ in range(blocks + 1):
                mods = [ResBlock(c + skip_chs.pop(), co, emb, dropout)]
                if lvl in attn_levels:
                    mods.append(Attention(co, heads))
                self.up.append(nn.ModuleList(mods))
                c = co
            if lvl > 0:
                self.up.append(nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c, 3, padding=1)]))
        self.out = nn.Sequential(_gn(c), nn.SiLU(), nn.Conv2d(c, out_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    def embed(self, t: torch.Tensor, cond: torch.Tensor | None, log2scale: torch.Tensor) -> torch.Tensor:
        base = self.cfg["base"]
        e = self.t_mlp(sinusoidal(t, base)) + self.s_mlp(sinusoidal(log2scale * 100.0, base))
        if cond is None:
            e = e + self.null_cond.unsqueeze(0)
        else:
            # a sample whose flags are all zero is the unconditional case, give it the null token
            given = (cond[:, cond.shape[1] // 2:].sum(1, keepdim=True) > 0).float()
            e = e + given * self.c_mlp(cond) + (1 - given) * self.null_cond.unsqueeze(0)
        return e

    def _block(self, mods, h, emb):
        for m in mods:
            h = m(h, emb) if isinstance(m, ResBlock) else m(h)
        return h

    def _run(self, mods, h, emb):
        if self.grad_ckpt and self.training and torch.is_grad_enabled():
            return checkpoint(self._block, mods, h, emb, use_reentrant=False)
        return self._block(mods, h, emb)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None, log2scale: torch.Tensor):
        emb = self.embed(t, cond, log2scale)
        h = self.inp(x)
        skips = [h]
        for mods in self.down:
            if isinstance(mods[0], ResBlock):
                h = self._run(mods, h, emb)
            else:
                h = mods[0](h)
            skips.append(h)
        h = self._run(self.mid, h, emb)
        for mods in self.up:
            if isinstance(mods[0], ResBlock):
                h = self._run(mods, torch.cat([h, skips.pop()], 1), emb)
            else:
                h = mods[1](mods[0](h))
        return self.out(h)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    net = UNet()
    print(f"{count_params(net) / 1e6:.1f} M params")
    x = torch.randn(2, 9, 128, 128)
    y = net(x, torch.tensor([10, 500]), torch.zeros(2, 12), torch.zeros(2))
    print(y.shape)
