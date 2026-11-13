"""Untrained matched temporal CNN / local-attention research candidates.

Inputs are six *observed* corresponding visible/NIR/SWIR bands, never the
legacy five-band-plus-EMIT label stack. Radiometry, native physics computation,
units, georegistration and inference-available QA are caller responsibilities.
Native physics rasters must cover the same footprint; nearest alignment here
does not create finer physical measurements. Annotation availability is not QA.

This compact pilot uses local temporal cross-attention, not an additional
shifted-window spatial transformer. Both variants use the same CNN/decoder and
independent scene path. Channel-only normalization avoids spatial padding in
normalization statistics. Invalid VALUES cannot leak; changing support or
canvas can still change convolutional boundary/context effects.
No-reference inputs are supported; they do not certify methane observability.
No training, calibrated alert policy, or performance claim is supplied here.
"""
from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import nn
from torch.nn import functional as F


#### SHARED CONVOLUTIONAL AND TEMPORAL COMPONENTS ####

class ChannelNorm(nn.Module):
    """LayerNorm over channels independently at every pixel, never spatially."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ResidualBlock(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            ChannelNorm(cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            ChannelNorm(cout),
        )
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.layers(x) + self.skip(x))


class LocalTemporalAttention(nn.Module):
    """Target queries attend to their 3x3 reference neighborhood plus self.

    The self key prevents all-masked softmax. An explicit gate makes the whole
    block the identity wherever no valid reference key exists, including its
    feedforward and bias terms. Reference values are sanitized before encoding.
    """
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.heads, self.depth = heads, channels // heads
        self.norm = ChannelNorm(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.kv = nn.Conv2d(channels, channels * 2, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        self.ff = nn.Sequential(ChannelNorm(channels),
                                nn.Conv2d(channels, channels * 2, 1), nn.GELU(),
                                nn.Conv2d(channels * 2, channels, 1))

    def forward(self, target: torch.Tensor, reference: torch.Tensor,
                reference_valid: torch.Tensor, target_valid: torch.Tensor) -> torch.Tensor:
        b, c, h, w = target.shape
        q = self.q(self.norm(target)).reshape(b, self.heads, self.depth, 1, h * w)
        rk, rv = self.kv(self.norm(reference)).chunk(2, dim=1)
        tk, tv = self.kv(self.norm(target)).chunk(2, dim=1)
        def neighborhood(x: torch.Tensor) -> torch.Tensor:
            return F.unfold(x, 3, padding=1).reshape(b, self.heads, self.depth, 9, h * w)
        k = torch.cat((tk.reshape(b, self.heads, self.depth, 1, h * w), neighborhood(rk)), dim=3)
        v = torch.cat((tv.reshape(b, self.heads, self.depth, 1, h * w), neighborhood(rv)), dim=3)
        valid = F.unfold(reference_valid.float(), 3, padding=1).bool()
        key_valid = torch.cat((torch.ones_like(valid[:, :1]), valid), dim=1)
        scores = (q * k).sum(dim=2) / math.sqrt(self.depth)
        scores = scores.masked_fill(~key_valid[:, None], float("-inf"))
        mixed = (scores.softmax(dim=2).unsqueeze(2) * v).sum(dim=3).reshape(b, c, h, w)
        gate = valid.any(dim=1).reshape(b, 1, h, w) & target_valid
        updated = target + self.out(mixed)
        updated = updated + self.ff(updated)
        return torch.where(gate, updated, target)


#### MODEL INPUT CONTRACT AND MATCHED PAIR ####

class TemporalAttentionCandidate(nn.Module):
    """Same-sensor/product temporal pair; sensor IDs index configured stems.

    ``physics_channel_names`` fixes caller-defined channel semantics and order.
    Each call must provide the identical names and native validity mask. No
    synthetic MBMP/units or label-derived features are manufactured internally.
    Outputs are logits and inference QA support, not calibrated probabilities.
    """
    widths = (32, 64, 128, 256)

    def __init__(self, physics_channel_names: Sequence[str], *,
                 sensor_count: int = 2, attention: bool = False) -> None:
        super().__init__()
        names = tuple(physics_channel_names)
        if not names or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
            raise ValueError("physics_channel_names must be nonempty, unique names")
        if not isinstance(sensor_count, int) or isinstance(sensor_count, bool) or sensor_count < 1:
            raise ValueError("sensor_count must be a positive integer")
        self.physics_channel_names, self.sensor_count = names, sensor_count
        self.stems = nn.ModuleList([ResidualBlock(7, 32) for _ in range(sensor_count)])
        self.encoder = nn.ModuleList([
            nn.Sequential(ResidualBlock(cin, cout), ResidualBlock(cout, cout))
            for cin, cout in zip((32, 32, 64, 128), self.widths)
        ])
        p = len(names)
        # Identical temporal and native-evidence routes in both variants.
        self.fusion = nn.ModuleList([nn.Conv2d(4 * c + p + 3, c, 1) for c in self.widths])
        self.decoder = nn.ModuleList([ResidualBlock(384, 128), ResidualBlock(192, 64), ResidualBlock(96, 32)])
        self.mask_head = nn.Conv2d(32, 1, 1)
        self.scene_encoder = nn.Sequential(ResidualBlock(24 + p + 3, 32),
                                           nn.AvgPool2d(2), ResidualBlock(32, 64))
        self.scene_head = nn.Sequential(nn.Linear(2 * (64 + 32), 64), nn.SiLU(), nn.Linear(64, 1))
        # Construct last so a common seed initializes shared modules identically.
        self.attention = nn.ModuleDict({
            str(i): nn.ModuleList([LocalTemporalAttention(self.widths[i], heads) for _ in range(2)])
            for i, heads in ((2, 4), (3, 8))
        } if attention else {})

    def forward(self, target: torch.Tensor, reference: torch.Tensor,
                sensor_index: torch.Tensor, target_valid: torch.Tensor,
                reference_valid: torch.Tensor, native_physics: torch.Tensor,
                native_physics_valid: torch.Tensor, *,
                physics_channel_names: Sequence[str]) -> dict[str, torch.Tensor]:
        if tuple(physics_channel_names) != self.physics_channel_names:
            raise ValueError("native physics channel names/order differ from model contract")
        if target.ndim != 4 or target.shape[1] != 6 or min(target.shape[0], *target.shape[-2:]) < 1:
            raise ValueError("target must be nonempty Bx6xHxW observed bands")
        if reference.shape != target.shape:
            raise ValueError("reference must match target Bx6xHxW")
        b, _, h, w = target.shape
        if sensor_index.shape != (b,) or sensor_index.dtype != torch.long:
            raise ValueError("sensor_index must be B integer (torch.long) indices")
        if bool(((sensor_index < 0) | (sensor_index >= self.sensor_count)).any()):
            raise ValueError("sensor_index outside configured stems")
        def check_mask(mask: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
            if mask.shape != shape or mask.dtype != torch.bool:
                raise ValueError(f"{name} must be boolean with shape {shape}")
        check_mask(target_valid, (b, 1, h, w), "target_valid")
        check_mask(reference_valid, (b, 1, h, w), "reference_valid")
        if (native_physics.ndim != 4 or native_physics.shape[:2] != (b, len(self.physics_channel_names))
                or min(native_physics.shape[-2:]) < 1):
            raise ValueError("native_physics must be BxPxHp xWp with exactly the configured channels")
        check_mask(native_physics_valid, (b, 1, *native_physics.shape[-2:]), "native_physics_valid")
        tensors = (reference, sensor_index, target_valid, reference_valid, native_physics, native_physics_valid)
        if any(x.device != target.device for x in tensors):
            raise ValueError("all inputs must be on the same device")
        if not target.is_floating_point() or reference.dtype != target.dtype or native_physics.dtype != target.dtype:
            raise ValueError("observations and physics must use the same floating dtype")
        def clean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            if bool((~torch.isfinite(x) & valid).any()):
                raise ValueError("nonfinite input marked valid")
            return torch.where(valid, x, torch.zeros_like(x))
        target, reference = clean(target, target_valid), clean(reference, reference_valid)
        physics = F.interpolate(clean(native_physics, native_physics_valid), size=(h, w), mode="nearest")
        pv = F.interpolate(native_physics_valid.float(), size=(h, w), mode="nearest").bool()
        # Zero-pad once to a multiple of eight for three downsamplings.
        pad = (0, (-w) % 8, 0, (-h) % 8)
        target, reference, physics = (F.pad(x, pad) for x in (target, reference, physics))
        tv, rv, pv = (F.pad(x, pad) for x in (target_valid, reference_valid, pv))
        pair_valid = tv & rv
        delta = torch.where(pair_valid, target - reference, torch.zeros_like(target))
        raw = torch.cat((target, reference, delta, delta.abs(), physics, tv, rv, pv), dim=1)
        def stem(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            x = torch.cat((x, valid), dim=1)
            out = x.new_zeros((b, 32, *x.shape[-2:]))
            for i, layer in enumerate(self.stems):
                selected = sensor_index == i
                if bool(selected.any()):
                    out[selected] = layer(x[selected])
            return out * valid
        t, r = stem(target, tv), stem(reference, rv)
        skips = []
        for i, layer in enumerate(self.encoder):
            if i:
                t, r = F.avg_pool2d(t, 2), F.avg_pool2d(r, 2)
                # Strict support: reference keys require a wholly valid source cell.
                tv = F.avg_pool2d(tv.float(), 2) == 1
                rv = F.avg_pool2d(rv.float(), 2) == 1
            t, r = layer(t) * tv, layer(r) * rv
            attended = t
            if str(i) in self.attention:
                for block in self.attention[str(i)]:
                    attended = block(attended, r, rv, tv)
            d = (t - r) * (tv & rv)
            evidence = F.interpolate(torch.cat((physics, pv), dim=1), size=t.shape[-2:], mode="nearest")
            skips.append(self.fusion[i](torch.cat((attended, r, d, d.abs(), evidence, tv, rv), dim=1)))
        dense = skips[-1]
        for layer, skip in zip(self.decoder, reversed(skips[:-1])):
            dense = layer(torch.cat((F.interpolate(dense, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip), dim=1))
        mask_logits = self.mask_head(dense)[..., :h, :w]
        scene = self.scene_encoder(raw)
        scene = F.interpolate(scene, size=dense.shape[-2:], mode="bilinear", align_corners=False)
        scene = torch.cat((scene, dense.detach()), dim=1)[..., :h, :w]
        valid = target_valid.to(scene.dtype)
        mean = (scene * valid).sum(dim=(-2, -1)) / valid.sum(dim=(-2, -1)).clamp_min(1)
        maximum = scene.masked_fill(~target_valid, float("-inf")).amax(dim=(-2, -1))
        maximum = torch.where(target_valid.any(dim=(-2, -1)), maximum, torch.zeros_like(maximum))
        return {"mask_logits": mask_logits,
                "scene_logits": self.scene_head(torch.cat((mean, maximum), dim=1)).squeeze(1),
                "support": target_valid}


def matched_pair(physics_channel_names: Sequence[str], *, sensor_count: int = 2
                 ) -> tuple[TemporalAttentionCandidate, TemporalAttentionCandidate]:
    """Create CNN/hybrid with identical shared initial weights, no global reseed."""
    cnn = TemporalAttentionCandidate(physics_channel_names, sensor_count=sensor_count)
    hybrid = TemporalAttentionCandidate(physics_channel_names, sensor_count=sensor_count, attention=True)
    incompatible = hybrid.load_state_dict(cnn.state_dict(), strict=False)
    if incompatible.unexpected_keys or any(not k.startswith("attention.") for k in incompatible.missing_keys):
        raise RuntimeError("candidate architectures no longer have matched shared parameters")
    return cnn, hybrid
