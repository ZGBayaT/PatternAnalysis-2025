# modules.py
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# 基础工具
# ----------------------------
class LayerNorm2d(nn.Module):
    """LayerNorm over channels for NCHW, 等价于对 C 做 LN。"""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class DropPath(nn.Module):
    """Stochastic Depth: 按路径丢弃残差分支（仅训练时）。"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # (N, 1, 1, 1) broadcast
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor


# ----------------------------
# ConvNeXt 组件
# ----------------------------
class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block:
    - Depthwise Conv 7x7 保持通道不变
    - Permute 到 channels-last 做 LayerNorm + 1x1 MLP (4x扩张 → GELU → 1x1还原)
    - γ 缩放参数（可学习标量 per-channel）
    - DropPath + 残差
    """
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise
        self.ln = nn.LayerNorm(dim, eps=1e-6)  # channels-last
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) \
            if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)  # (N,C,H,W)
        # NCHW -> NHWC
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        # NHWC -> NCHW
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x


class DownsampleLayer(nn.Module):
    """
    ConvNeXt 下采样层：
    - LayerNorm (2d 版或 channels-last 版都可)
    - 2x2, stride=2 的 Conv 降采样，通道提升
    原文实现通常用 LN(channels-last) + Conv2d(stride=2)。
    这里用 LayerNorm2d 简洁且稳定。
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm = LayerNorm2d(in_ch)
        self.reduction = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.reduction(x)
        return x


class ConvNeXtStage(nn.Module):
    """由若干 ConvNeXtBlock 组成的一个 stage。"""
    def __init__(self, dim: int, depth: int, drop_path: List[float], layer_scale_init_value: float = 1e-6):
        super().__init__()
        blocks = []
        for i in range(depth):
            blocks.append(ConvNeXtBlock(dim, drop_path=drop_path[i], layer_scale_init_value=layer_scale_init_value))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


# ----------------------------
# 完整 ConvNeXt
# ----------------------------
class ConvNeXt(nn.Module):
    """
    ConvNeXt 主干 + 分类头。
    参数:
        depths: 各 stage 的 block 数，如 [3, 3, 9, 3] (tiny)
        dims:   各 stage 的通道数，如 [96, 192, 384, 768] (tiny)
        num_classes: 分类数
        drop_path_rate: 总体 StochasticDepth 最大值，按深度线性分配
    """
    def __init__(
        self,
        depths: List[int],
        dims: List[int],
        num_classes: int = 2,
        in_chans: int = 3,
        drop_path_rate: float = 0.1,
        layer_scale_init_value: float = 1e-6
    ):
        super().__init__()
        assert len(depths) == 4 and len(dims) == 4

        # Stem: 4x 下采样
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )

        # 为所有 blocks 生成分布式 drop_path 概率（线性增长）
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # Stages
        cur = 0
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        # stage 0
        self.stages.append(ConvNeXtStage(dims[0], depths[0], dpr[cur:cur+depths[0]], layer_scale_init_value))
        cur += depths[0]

        # stage 1..3: 先下采样再堆 block
        for i in range(3):
            self.downsample_layers.append(DownsampleLayer(dims[i], dims[i+1]))
            self.stages.append(ConvNeXtStage(dims[i+1], depths[i+1], dpr[cur:cur+depths[i+1]], layer_scale_init_value))
            cur += depths[i+1]

        # Head
        self.head_norm = nn.LayerNorm(dims[-1], eps=1e-6)  # channels-last
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,3,224,224)
        x = self.stem(x)
        # stage 0
        x = self.stages[0](x)
        # stage 1..3 with downsample
        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i+1](x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)          # (N,C,H,W)
        # Global AvgPool -> (N,C)
        x = x.mean(dim=[2, 3])
        # channels-last LN
        x = self.head_norm(x)
        x = self.head(x)
        return x


# ----------------------------
# 工厂函数
# ----------------------------
_VARIANTS = {
    # depths, dims
    "tiny":  ([3, 3, 9, 3],   [96, 192, 384, 768]),
    "small": ([3, 3, 27, 3],  [96, 192, 384, 768]),
    "base":  ([3, 3, 27, 3],  [128, 256, 512, 1024]),
    "large": ([3, 3, 27, 3],  [192, 384, 768, 1536]),
}

def build_convnext(variant: str = "tiny", num_classes: int = 2, drop_path_rate: float = 0.1) -> ConvNeXt:
    if variant not in _VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(_VARIANTS.keys())}")
    depths, dims = _VARIANTS[variant]
    model = ConvNeXt(depths=depths, dims=dims, num_classes=num_classes, drop_path_rate=drop_path_rate)
    return model


# ----------------------------
# 可选：损失/优化器工厂
# ----------------------------
def create_criterion():
    return nn.CrossEntropyLoss()

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
