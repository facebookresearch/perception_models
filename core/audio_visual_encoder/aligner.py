import torch
import torch.nn.functional as F


class AlignModalities(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalize: bool = True,
        btc: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.btc = btc
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1
        )
        if self.normalize:
            self.layer_norm = torch.nn.LayerNorm(self.out_channels)

    def get_sizes(self, seq, mask):
        if mask is not None:
            sizes = mask.sum(-1)
        else:
            sizes = torch.full((seq.size(0),), seq.size(-1), device=seq.device)
        if sizes.dim() > 1:
            sizes = sizes.squeeze(1)
        return sizes.long()

    def interpolate(self, tgt, tgt_sizes, src_sizes) -> torch.Tensor:
        result = torch.zeros(
            tgt.size(0), tgt.size(1), src_sizes.max(), device=tgt.device
        )
        for i, (tgt_row, tgt_size, src_size) in enumerate(
            zip(tgt, tgt_sizes, src_sizes)
        ):
            tgt_row = tgt_row[:, :tgt_size]
            interpolated = F.interpolate(tgt_row[None], size=src_size, mode="nearest")
            result[i, :, :src_size] = interpolated[0]
        return result

    def forward(self, src, src_mask, tgt, tgt_mask):
        # BxTxC -> BxCxT
        src = src.transpose(1, 2)
        tgt = tgt.transpose(1, 2)

        tgt = self.conv(tgt)

        src_sizes = self.get_sizes(src, src_mask)
        tgt_sizes = self.get_sizes(tgt, tgt_mask)
        if all(src_sizes == tgt_sizes):
            upsampled = tgt
        else:
            upsampled = self.interpolate(tgt, tgt_sizes, src_sizes)

        upsampled = upsampled.permute(0, 2, 1)  # BxCxT -> BxTxC
        if self.normalize:
            upsampled = self.layer_norm(upsampled)
        return upsampled, src_mask
