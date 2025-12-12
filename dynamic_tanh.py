import torch
import torch.nn as nn
from timm.layers import LayerNorm2d

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5, variant='scalar'):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last
        self.variant = variant

        # [RQ2 关键修改] 根据 variant 决定 alpha 的形状
        if variant == 'scalar':
            # 标量模式: 整个层共用一个 alpha (参数量: 1)
            self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        elif variant == 'channel':
            # 通道模式: 每个特征通道一个 alpha (参数量: C)
            # normalized_shape 通常是一个 int (C) 或者 tuple (C, H, W)
            # 这里简化处理，假设 normalized_shape 是特征维度 C
            dim = normalized_shape if isinstance(normalized_shape, int) else normalized_shape[0]
            self.alpha = nn.Parameter(torch.ones(dim) * alpha_init_value)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # 处理 alpha 的广播机制
        if self.variant == 'channel' and not self.channels_last:
             # 如果是 ConvNeXt (B, C, H, W) 且是 channel 模式，alpha 需要变成 (C, 1, 1)
             alpha = self.alpha.view(-1, 1, 1)
        else:
             # 其他情况 (ViT 或 标量模式)，直接广播即可
             alpha = self.alpha

        x = torch.tanh(alpha * x)
        
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, variant={self.variant}, alpha_init={self.alpha_init_value}"

# [关键修改] 转换函数增加 variant 参数
def convert_ln_to_dyt(module, variant='scalar', alpha_init_value=0.5):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        # 将 variant 和 alpha_init_value 传递给 DynamicTanh
        module_output = DynamicTanh(
            module.normalized_shape, 
            not isinstance(module, LayerNorm2d),
            alpha_init_value=alpha_init_value,
            variant=variant
        )
    
    # 递归替换子模块
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child, variant=variant, alpha_init_value=alpha_init_value))
    
    del module
    return module_output