import torch
from .config import DacConfig
from transformers.models.dac.modeling_dac import DacEncoder


class VAEBottleneck(torch.nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        bottleneck_dim: int = 512,
    ):
        super().__init__()
        self.in_proj = torch.nn.Conv1d(input_dim, bottleneck_dim * 2, kernel_size=1)
        self.out_proj = torch.nn.Conv1d(bottleneck_dim, input_dim, kernel_size=1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, _ = self.in_proj(z).chunk(2, dim=1)
        return mean


class DacEncoderVAE(torch.nn.Module):
    def __init__(self, config: DacConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = DacEncoder(config)
        self.bottleneck = VAEBottleneck(config.codebook_size, config.codebook_dim)
        self.hop_length = config.hop_length
        self.sampling_rate = config.sampling_rate

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
            z = self.encoder(self._pad(waveform))
            encoded_frames = self.bottleneck(z)
        return encoded_frames

    def _pad(self, wavs):
        length = wavs.size(-1)
        if length % self.hop_length:
            p1d = (0, self.hop_length - (length % self.hop_length))
            return torch.nn.functional.pad(wavs, p1d, "reflect")
        else:
            return wavs
