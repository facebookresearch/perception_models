import torch
from typing import Optional
from .config import TransformerConfig


class MaskedGroupNorm(torch.nn.GroupNorm):
    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if padding_mask is None:
            return super().forward(x)
        B, C, L = x.shape
        G = self.num_groups
        x_grouped = x.view(B, G, C // G, L)
        padding_mask_grouped = padding_mask.reshape(B, G, C // G, L).bool()
        mean = torch.masked.mean(
            x_grouped, mask=padding_mask_grouped, dim=(2, 3), keepdim=True
        )
        var = torch.masked.var(
            x_grouped,
            mask=padding_mask_grouped,
            dim=(2, 3),
            keepdim=True,
            unbiased=False,
        )
        x_norm = (x_grouped - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.view(B, C, L)
        if self.affine:
            x_norm = x_norm * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return x_norm * padding_mask


class ConvBlock1d(torch.nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.groupnorm = MaskedGroupNorm(num_groups=1, num_channels=config.hidden_size)
        self.activation = torch.nn.SiLU()
        self.project = torch.nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=3,
            padding="same",
        )

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.groupnorm(x, padding_mask=padding_mask)
        x = self.activation(x)
        return self.project(x)


class ResnetBlock1d(torch.nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.block1 = ConvBlock1d(config)
        self.block2 = ConvBlock1d(config)

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).expand_as(x)
        h = self.block1(x, padding_mask=padding_mask)
        h = self.block2(h, padding_mask=padding_mask)
        return h + x
