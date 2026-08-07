"""Pure PyTorch inference model for the released Swin-H checkpoint."""

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _to_pair(value):
    return value if isinstance(value, tuple) else (value, value)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * random_tensor.floor()


class PatchEmbed(nn.Module):
    def __init__(self, embed_dims=384, patch_size=4):
        super().__init__()
        self.patch_size = _to_pair(patch_size)
        self.projection = nn.Conv2d(
            3, embed_dims, kernel_size=self.patch_size,
            stride=self.patch_size)
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, x):
        height, width = x.shape[-2:]
        pad_h = (self.patch_size[0] - height % self.patch_size[0]) % self.patch_size[0]
        pad_w = (self.patch_size[1] - width % self.patch_size[1]) % self.patch_size[1]
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.projection(x)
        shape = x.shape[-2:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, shape


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(4 * in_channels)
        self.reduction = nn.Linear(4 * in_channels, out_channels, bias=False)
        self.sampler = nn.Unfold(kernel_size=2, stride=2)

    def forward(self, x, input_size):
        batch, length, channels = x.shape
        height, width = input_size
        if length != height * width:
            raise ValueError("Patch sequence and spatial size do not match")
        x = x.view(batch, height, width, channels).permute(0, 3, 1, 2)
        pad_h = height % 2
        pad_w = width % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            height += pad_h
            width += pad_w
        x = self.sampler(x).transpose(1, 2)
        x = self.reduction(self.norm(x))
        return x, (height // 2, width // 2)


class WindowMSA(nn.Module):
    def __init__(self, embed_dims, window_size, num_heads):
        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = _to_pair(window_size)
        self.num_heads = num_heads
        self.scale = (embed_dims // num_heads) ** -0.5

        table_size = ((2 * self.window_size[0] - 1)
                      * (2 * self.window_size[1] - 1))
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_size, num_heads))

        height, width = self.window_size
        seq1 = torch.arange(0, (2 * width - 1) * height, 2 * width - 1)
        seq2 = torch.arange(0, width)
        relative_index = (seq1[:, None] + seq2[None, :]).reshape(1, -1)
        relative_index = (relative_index + relative_index.T).flip(1).contiguous()
        self.register_buffer("relative_position_index", relative_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(0.0)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_windows, tokens, 3, self.num_heads,
            channels // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query * self.scale) @ key.transpose(-2, -1)

        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)]
        relative_bias = relative_bias.view(
            tokens, tokens, self.num_heads).permute(2, 0, 1).contiguous()
        attention = attention + relative_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attention = attention.view(
                batch_windows // num_windows, num_windows,
                self.num_heads, tokens, tokens)
            attention = attention + mask.unsqueeze(0).unsqueeze(2)
            attention = attention.view(
                -1, self.num_heads, tokens, tokens)
        attention = self.softmax(attention)
        attention = self.attn_drop(attention)

        x = (attention @ value).transpose(1, 2).reshape(
            batch_windows, tokens, channels)
        return self.proj_drop(self.proj(x))


class ShiftWindowMSA(nn.Module):
    def __init__(self, embed_dims, num_heads, window_size=7, shift_size=0,
                 drop_path=0.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.w_msa = WindowMSA(embed_dims, window_size, num_heads)
        self.drop = DropPath(drop_path)

    @staticmethod
    def _partition(x, window_size):
        batch, height, width, channels = x.shape
        x = x.view(
            batch, height // window_size, window_size,
            width // window_size, window_size, channels)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            -1, window_size, window_size, channels)

    @staticmethod
    def _reverse(windows, height, width, window_size):
        batch = int(windows.shape[0] / (height * width / window_size**2))
        x = windows.view(
            batch, height // window_size, width // window_size,
            window_size, window_size, -1)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            batch, height, width, -1)

    @classmethod
    def _attention_mask(cls, shape, window_size, shift_size, device):
        if shift_size == 0:
            return None
        image_mask = torch.zeros(1, *shape, 1, device=device)
        slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        count = 0
        for height_slice in slices:
            for width_slice in slices:
                image_mask[:, height_slice, width_slice, :] = count
                count += 1
        mask_windows = cls._partition(image_mask, window_size).view(
            -1, window_size**2)
        mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def forward(self, query, shape):
        batch, length, channels = query.shape
        height, width = shape
        if length != height * width:
            raise ValueError("Attention sequence and spatial size do not match")
        query = query.view(batch, height, width, channels)

        shift_size = self.shift_size
        if min(height, width) == self.window_size:
            shift_size = 0
        if min(height, width) < self.window_size:
            raise ValueError("Input is smaller than the attention window")

        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_w, 0, pad_h))
        padded_height, padded_width = query.shape[1:3]

        if shift_size:
            query = torch.roll(
                query, shifts=(-shift_size, -shift_size), dims=(1, 2))
        mask = self._attention_mask(
            (padded_height, padded_width), self.window_size,
            shift_size, query.device)
        windows = self._partition(query, self.window_size).view(
            -1, self.window_size**2, channels)
        windows = self.w_msa(windows, mask).view(
            -1, self.window_size, self.window_size, channels)
        x = self._reverse(
            windows, padded_height, padded_width, self.window_size)

        if shift_size:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))
        if pad_h or pad_w:
            x = x[:, :height, :width, :].contiguous()
        return self.drop(x.view(batch, height * width, channels))


class FFN(nn.Module):
    def __init__(self, embed_dims, drop_path=0.0):
        super().__init__()
        hidden_dims = embed_dims * 4
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(embed_dims, hidden_dims),
                nn.GELU(),
                nn.Dropout(0.0),
            ),
            nn.Linear(hidden_dims, embed_dims),
            nn.Dropout(0.0),
        )
        self.dropout_layer = DropPath(drop_path)

    def forward(self, x, identity=None):
        if identity is None:
            identity = x
        return identity + self.dropout_layer(self.layers(x))


class SwinBlock(nn.Module):
    def __init__(self, embed_dims, num_heads, shift, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dims)
        self.attn = ShiftWindowMSA(
            embed_dims, num_heads, window_size=7,
            shift_size=3 if shift else 0, drop_path=drop_path)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = FFN(embed_dims, drop_path=drop_path)

    def forward(self, x, shape):
        x = x + self.attn(self.norm1(x), shape)
        return self.ffn(self.norm2(x), identity=x)


class SwinStage(nn.Module):
    def __init__(self, embed_dims, depth, num_heads, drop_paths, downsample):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(embed_dims, num_heads, index % 2 == 1, drop_paths[index])
            for index in range(depth)
        ])
        self.downsample = (
            PatchMerging(embed_dims, embed_dims * 2) if downsample else None)

    def forward(self, x, shape):
        for block in self.blocks:
            x = block(x, shape)
        return x, shape


class SwinTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        embed_dims = 384
        depths = (2, 2, 18, 2)
        num_heads = (12, 24, 48, 96)
        total_depth = sum(depths)
        drop_paths = [0.2 * index / (total_depth - 1)
                      for index in range(total_depth)]

        self.patch_embed = PatchEmbed(embed_dims=embed_dims)
        self.drop_after_pos = nn.Dropout(0.0)
        self.stages = nn.ModuleList()
        offset = 0
        for index, (depth, heads) in enumerate(zip(depths, num_heads)):
            channels = embed_dims * 2**index
            self.stages.append(SwinStage(
                channels, depth, heads,
                drop_paths[offset:offset + depth],
                downsample=index < len(depths) - 1))
            offset += depth
        self.norm0 = nn.LayerNorm(384)
        self.norm1 = nn.LayerNorm(768)
        self.norm2 = nn.LayerNorm(1536)
        self.norm3 = nn.LayerNorm(3072)

    def forward(self, x):
        x, shape = self.patch_embed(x)
        x = self.drop_after_pos(x)
        outputs = []
        for index, stage in enumerate(self.stages):
            x, shape = stage(x, shape)
            norm = getattr(self, f"norm{index}")
            output = norm(x).view(
                -1, shape[0], shape[1], x.shape[-1])
            outputs.append(output.permute(0, 3, 1, 2).contiguous())
            if stage.downsample is not None:
                x, shape = stage.downsample(x, shape)
        return tuple(outputs)


class ConvOnly(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        return self.conv(x)


class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = (384, 768, 1536, 3072)
        self.lateral_convs = nn.ModuleList([
            ConvOnly(channels, 256, 1) for channels in in_channels
        ])
        self.fpn_convs = nn.ModuleList([
            ConvOnly(256, 256, 3, padding=1) for _ in in_channels
        ])

    def forward(self, inputs):
        laterals = [layer(value) for layer, value in zip(
            self.lateral_convs, inputs)]
        for index in range(len(laterals) - 1, 0, -1):
            laterals[index - 1] = laterals[index - 1] + F.interpolate(
                laterals[index], size=laterals[index - 1].shape[-2:],
                mode="nearest")
        return tuple(layer(value) for layer, value in zip(
            self.fpn_convs, laterals))


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class SemanticFPNHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_heads = nn.ModuleList([
            nn.Sequential(ConvBNReLU(256, 64)),
            nn.Sequential(
                ConvBNReLU(256, 64),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ),
            nn.Sequential(
                ConvBNReLU(256, 64),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ConvBNReLU(64, 64),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ),
            nn.Sequential(
                ConvBNReLU(256, 64),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ConvBNReLU(64, 64),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ConvBNReLU(64, 64),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ),
        ])
        self.conv_seg = nn.Conv2d(64, 64, 1)

    def forward(self, inputs):
        output = self.scale_heads[0](inputs[0])
        for index in range(1, len(self.scale_heads)):
            scaled = self.scale_heads[index](inputs[index])
            output = output + F.interpolate(
                scaled, size=output.shape[-2:], mode="bilinear",
                align_corners=False)
        return self.conv_seg(output)


class FlowSegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones(()) * 2.659260036932778)
        self.backbone = SwinTransformer()
        self.neck = FPN()
        self.head = SemanticFPNHead()
        self.with_head = True

    def extract_feat(self, inputs):
        return self.neck(self.backbone(inputs))

    def forward(self, inputs):
        return self.head(self.extract_feat(inputs))


__all__ = ["FlowSegModel"]
