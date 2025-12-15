import json
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import safe_open
from transformers import AutoModel

from ..vision_encoder.pe import CLIP as PeEncoder
from .aligner import AlignModalities
from .audio_codec import DacEncoderVAE
from .config import (
    AudioEncoderConfig,
    PEAudioFrameConfig,
    PEAudioVisualConfig,
    PEAudioVisualEncoderConfig,
    VisualEncoderConfig,
)
from .transformer import BaseModelOutputWithPooling, Transformer


@dataclass
class AudioOutput(BaseModelOutputWithPooling):
    audio_feature_padding_mask: Optional[torch.Tensor] = None
    dac_vae_features: Optional[torch.Tensor] = None


@dataclass
class VisualOutput(BaseModelOutputWithPooling):
    pe_output: Optional[torch.Tensor] = None


@dataclass
class AudioVisualOutput(BaseModelOutputWithPooling):
    audio_output: Optional[AudioOutput] = None
    visual_output: Optional[VisualOutput] = None


@dataclass
class PEAudioFrameOutput:
    audio_embeds: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    spans: Optional[list[list[list[float]]]] = None
    audio_output: Optional[AudioOutput] = None
    text_output: Optional[BaseModelOutputWithPooling] = None


@dataclass
class PEAudioVisualOutput:
    """
    Output embeddings and intermediate results from the PEAudioVisual model.

    Attributes:
        audio_embeds (Optional[torch.FloatTensor]): Embeddings for the audio modality.
        audio_visual_embeds (Optional[torch.FloatTensor]): Embeddings for the combined audio-visual modality.
        visual_embeds (Optional[torch.FloatTensor]): Embeddings for the visual modality.
        audio_text_embeds (Optional[torch.FloatTensor]): Embeddings for the audio-text modality.  This should be used for Audio <-> Text retrieval.
        audio_visual_text_embeds (Optional[torch.FloatTensor]): Embeddings for the audio-visual-text modality.  This should be used for Audio/Video <-> Text retrieval.
        visual_text_embeds (Optional[torch.FloatTensor]): Embeddings for the visual-text modality.  This should be used for Video <-> Text retrieval.
        audio_plus_text_embeds (Optional[torch.FloatTensor]): Embeddings for combined audio and text features.
        visual_plus_text_embeds (Optional[torch.FloatTensor]): Embeddings for combined visual and text features.
        audio_visual_output (Optional[AudioVisualOutput]): Intermediate outputs from the audio-visual encoder.
        text_output (Optional[BaseModelOutputWithPooling]): Intermediate outputs from the text encoder.
    """

    audio_embeds: Optional[torch.FloatTensor] = None
    audio_visual_embeds: Optional[torch.FloatTensor] = None
    visual_embeds: Optional[torch.FloatTensor] = None
    audio_text_embeds: Optional[torch.FloatTensor] = None
    audio_visual_text_embeds: Optional[torch.FloatTensor] = None
    visual_text_embeds: Optional[torch.FloatTensor] = None
    audio_plus_text_embeds: Optional[torch.FloatTensor] = None
    visual_plus_text_embeds: Optional[torch.FloatTensor] = None
    audio_visual_output: Optional[AudioVisualOutput] = None
    text_output: Optional[BaseModelOutputWithPooling] = None


class ContrastiveHead(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=in_dim, eps=1e-6)
        self.proj = torch.nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.layer_norm(x))


class AVTransformer(Transformer):
    def __init__(self, config):
        super().__init__(config)
        self.modality_aligner = AlignModalities(
            self.config.hidden_size, self.config.hidden_size, normalize=True, btc=True
        )
        self.concat_modality_proj = torch.nn.Linear(
            self.config.hidden_size * 2, self.config.hidden_size
        )
        self.data_proj = torch.nn.Linear(
            self.config.hidden_size, self.config.hidden_size
        )

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        audio_padding_mask: Optional[torch.Tensor] = None,
        video_padding_mask: Optional[torch.Tensor] = None,
    ):
        video, video_padding_mask = self.modality_aligner(
            audio, audio_padding_mask, video, video_padding_mask
        )
        x = torch.cat([audio, video], dim=-1)
        x = self.concat_modality_proj(x)
        return super().forward(self.data_proj(x), attention_mask=video_padding_mask)


class AudioEncoder(torch.nn.Module):
    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.data_proj = torch.nn.Linear(
            config.dac_vae_encoder.codebook_dim, config.audio_transformer.hidden_size
        )
        self.dac_vae_encoder = DacEncoderVAE(config.dac_vae_encoder)
        self.audio_transformer = Transformer(config.audio_transformer)

    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,  # codec_features
    ) -> AudioOutput:
        if input_features is None:
            codec_features = self.dac_vae_encoder(input_values).transpose(1, 2)
            feature_padding_mask = None
            if padding_mask is not None:
                feature_padding_mask = padding_mask[
                    :, :: self.dac_vae_encoder.config.hop_length
                ]
        else:
            codec_features = input_features
            feature_padding_mask = padding_mask
        outputs = self.audio_transformer(
            self.data_proj(codec_features), attention_mask=feature_padding_mask
        )
        return AudioOutput(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=outputs.pooler_output,
            audio_feature_padding_mask=feature_padding_mask,
            dac_vae_features=codec_features,
        )


class VisualEncoder(torch.nn.Module):
    def __init__(self, config: VisualEncoderConfig):
        super().__init__()
        # Note we only use the visual branch of the model.  Throw the rest away to save space
        self.pe_encoder = PeEncoder.from_config(
            config.pe_encoder, pretrained=False
        ).visual
        self.proj = torch.nn.Linear(
            self.pe_encoder.output_dim,
            config.visual_transformer.hidden_size,
            bias=False,
        )
        self.data_proj = torch.nn.Linear(
            config.visual_transformer.hidden_size, config.visual_transformer.hidden_size
        )
        self.visual_transformer = Transformer(config.visual_transformer)

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        padding_mask_videos: Optional[torch.Tensor] = None,
        pe_features: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithPooling:
        B, N, C, H, W = pixel_values_videos.shape
        if pe_features is None:
            backbone_output = self.pe_encoder(
                pixel_values_videos.view(B * N, C, H, W)
            ).view(B, N, -1)
            pe_features = F.normalize(backbone_output, dim=-1)
        projected = self.proj(pe_features)
        output = self.visual_transformer(
            self.data_proj(projected), attention_mask=padding_mask_videos
        )
        return VisualOutput(
            last_hidden_state=output.last_hidden_state,
            pooler_output=output.pooler_output,
            pe_output=pe_features,
        )


class AudioVisualEncoder(torch.nn.Module):
    def __init__(self, config: PEAudioVisualEncoderConfig):
        super().__init__()
        self.audio_model = AudioEncoder(config.audio_model)
        self.visual_model = VisualEncoder(config.visual_model)
        self.audio_visual_transformer = AVTransformer(config.audio_visual_transformer)

    def forward(
        self,
        input_values: torch.Tensor,
        pixel_values_videos: torch.Tensor,
        pe_features: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        padding_mask_videos: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,  # codec_features
    ) -> AudioVisualOutput:
        audio_output = self.audio_model(input_values, padding_mask=padding_mask, input_features=input_features)
        video_output = self.visual_model(
            pixel_values_videos, padding_mask_videos=padding_mask_videos, pe_features=pe_features
        )
        av_output = self.audio_visual_transformer(
            audio_output.last_hidden_state,
            video_output.last_hidden_state,
            audio_padding_mask=audio_output.audio_feature_padding_mask,
            video_padding_mask=padding_mask_videos,
        )
        return AudioVisualOutput(
            last_hidden_state=av_output.last_hidden_state,
            pooler_output=av_output.pooler_output,
            audio_output=audio_output,
            visual_output=video_output,
        )


class BasePEAudio(torch.nn.Module):
    @classmethod
    def from_config(cls, name_or_checkpoint: str, pretrained: bool = False):
        if os.path.isdir(name_or_checkpoint):
            checkpoint_dir = name_or_checkpoint
        else:
            checkpoint_dir = snapshot_download(
                repo_id=f"facebook/{name_or_checkpoint}", revision="perception_models"
            )
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path) as fin:
            config_dict = json.load(fin)
        config = cls.config_cls(**config_dict)
        model = cls(config)
        if pretrained:
            checkpoint_path = os.path.join(checkpoint_dir, "model.safetensors")
            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                model.load_state_dict({k: f.get_tensor(k) for k in f.keys()})
        return model


class PEAudioVisual(BasePEAudio):
    config_cls = PEAudioVisualConfig

    def __init__(self, config: PEAudioVisualConfig):
        super().__init__()
        self.config = config
        self.audio_visual_model = AudioVisualEncoder(config.audio_visual_model)
        self.text_model = AutoModel.from_config(config.text_model)
        self.audio_visual_text_head = ContrastiveHead(
            config.text_model.hidden_size, config.output_dim
        )
        self.audio_text_head = ContrastiveHead(
            config.text_model.hidden_size, config.output_dim
        )
        self.visual_text_head = ContrastiveHead(
            config.text_model.hidden_size, config.output_dim
        )
        self.audio_visual_head = ContrastiveHead(
            config.audio_visual_model.audio_visual_transformer.hidden_size,
            config.output_dim,
        )
        self.audio_head = ContrastiveHead(
            config.audio_visual_model.audio_model.audio_transformer.hidden_size,
            config.output_dim,
        )
        self.visual_head = ContrastiveHead(
            config.audio_visual_model.visual_model.visual_transformer.hidden_size,
            config.output_dim,
        )
        self.visual_plus_text_head = ContrastiveHead(
            config.audio_visual_model.visual_model.visual_transformer.hidden_size
            + config.text_model.hidden_size,
            config.output_dim,
        )
        self.audio_plus_text_head = ContrastiveHead(
            config.audio_visual_model.audio_model.audio_transformer.hidden_size
            + config.text_model.hidden_size,
            config.output_dim,
        )

    def _get_text_output(self, input_ids, attention_mask):
        nth_layer = self.config.nth_text_layer
        output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=nth_layer is not None,
        )
        if nth_layer is None:
            text_model_output = output.last_hidden_state
        else:
            text_model_output = output.hidden_states[nth_layer]

        return BaseModelOutputWithPooling(
            last_hidden_state=text_model_output, pooler_output=text_model_output[:, 0]
        )

    def encode_video_text(self, input_ids, attention_mask=None):
        text_outputs = self._get_text_output(input_ids, attention_mask)
        return self.visual_text_head(text_outputs.pooler_output)

    def encode_audio_text(self, input_ids, attention_mask=None):
        text_outputs = self._get_text_output(input_ids, attention_mask)
        return self.audio_text_head(text_outputs.pooler_output)

    def encode_audio_video_text(self, input_ids, attention_mask=None):
        text_outputs = self._get_text_output(input_ids, attention_mask)
        return self.audio_visual_text_head(text_outputs.pooler_output)

    def encode_audio(self, input_values, padding_mask=None, input_features=None):
        audio_outputs = self.audio_visual_model.audio_model(
            input_values, padding_mask=padding_mask, input_features=input_features
        )
        return self.audio_head(audio_outputs.pooler_output)

    def encode_video(self, pixel_values_videos, padding_mask_videos=None, pe_features=None):
        video_outputs = self.audio_visual_model.visual_model(
            pixel_values_videos, padding_mask_videos=padding_mask_videos, pe_features=pe_features
        )
        return self.visual_head(video_outputs.pooler_output)

    def encode_audio_video(
        self,
        input_values,
        pixel_values_videos,
        padding_mask=None,
        padding_mask_videos=None,
        pe_features=None,
        input_features=None,
    ):
        audio_video_outputs = self.audio_visual_model(
            input_values,
            pixel_values_videos,
            padding_mask=padding_mask,
            padding_mask_videos=padding_mask_videos,
            pe_features=pe_features,
            input_features=input_features,
        )
        return self.audio_visual_head(audio_video_outputs.pooler_output)

    def encode_audio_plus_text(
        self, input_ids, input_values, attention_mask=None, padding_mask=None, input_features=None
    ):
        text_outputs = self._get_text_output(input_ids, attention_mask)
        audio_outputs = self.audio_visual_model.audio_model(
            input_values, padding_mask=padding_mask, input_features=input_features
        )
        return self.audio_plus_text_head(
            torch.cat(
                [audio_outputs.pooler_output, text_outputs.pooler_output],
                dim=-1,
            )
        )

    def encode_video_plus_text(
        self,
        input_ids,
        pixel_values_videos,
        attention_mask=None,
        padding_mask_videos=None,
        pe_features=None,
    ):
        text_outputs = self._get_text_output(input_ids, attention_mask)
        video_outputs = self.audio_visual_model.visual_model(
            pixel_values_videos, padding_mask_videos=padding_mask_videos, pe_features=pe_features
        )
        return self.visual_plus_text_head(
            torch.cat(
                [video_outputs.pooler_output, text_outputs.pooler_output],
                dim=-1,
            )
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask_videos: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        pe_features: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        return_loss=False,
    ) -> PEAudioVisualOutput:
        # text embeddings
        audio_text_embeds = visual_text_embeds = audio_visual_text_embeds = None
        # media embeddings (audio, video, audio_video)
        audio_embeds = visual_embeds = audio_visual_embeds = None
        # media + text embeddings
        audio_plus_text_embeds = visual_plus_text_embeds = None

        audio_visual_outputs = None

        # Compute model outputs and embeddings for each modality
        text_outputs = None
        if input_ids is not None:
            text_outputs = self._get_text_output(input_ids, attention_mask)
        if input_values is not None and pixel_values_videos is not None:
            # If we compute audio/video outputs, then extract the intermediate audio and video outputs
            audio_visual_outputs = self.audio_visual_model(
                input_values,
                pixel_values_videos,
                padding_mask=padding_mask,
                padding_mask_videos=padding_mask_videos,
                pe_features=pe_features,
                input_features=input_features,
            )
            audio_outputs = audio_visual_outputs.audio_output
            video_outputs = audio_visual_outputs.visual_output

            audio_embeds = self.audio_head(audio_outputs.pooler_output)
            visual_embeds = self.visual_head(video_outputs.pooler_output)
            audio_visual_embeds = self.audio_visual_head(
                audio_visual_outputs.pooler_output
            )
            if text_outputs is not None:
                # Compute the corresponding text embeddings
                audio_text_embeds = self.audio_text_head(text_outputs.pooler_output)
                visual_text_embeds = self.visual_text_head(text_outputs.pooler_output)
                audio_visual_text_embeds = self.audio_visual_text_head(
                    text_outputs.pooler_output
                )
                audio_plus_text_embeds = self.audio_plus_text_head(
                    torch.cat(
                        [audio_outputs.pooler_output, text_outputs.pooler_output],
                        dim=-1,
                    )
                )
                visual_plus_text_embeds = self.visual_plus_text_head(
                    torch.cat(
                        [video_outputs.pooler_output, text_outputs.pooler_output],
                        dim=-1,
                    )
                )
        else:
            if pixel_values_videos is not None:
                video_outputs = self.audio_visual_model.visual_model(
                    pixel_values_videos, padding_mask_videos=padding_mask_videos, pe_features=pe_features
                )
                audio_visual_outputs = AudioVisualOutput(visual_output=video_outputs)
                visual_embeds = self.visual_head(video_outputs.pooler_output)
                if text_outputs is not None:
                    visual_text_embeds = self.visual_text_head(
                        text_outputs.pooler_output
                    )
                    visual_plus_text_embeds = self.visual_plus_text_head(
                        torch.cat(
                            [video_outputs.pooler_output, text_outputs.pooler_output],
                            dim=-1,
                        )
                    )
            elif input_values is not None:
                audio_outputs = self.audio_visual_model.audio_model(
                    input_values, padding_mask=padding_mask, input_features=input_features
                )
                audio_visual_outputs = AudioVisualOutput(audio_output=audio_outputs)
                audio_embeds = self.audio_head(audio_outputs.pooler_output)
                if text_outputs is not None:
                    audio_text_embeds = self.audio_text_head(text_outputs.pooler_output)
                    audio_plus_text_embeds = self.audio_plus_text_head(
                        torch.cat(
                            [audio_outputs.pooler_output, text_outputs.pooler_output],
                            dim=-1,
                        )
                    )
            elif text_outputs is not None:
                # If text is supplied, but no audio or video, use audio_video_text as the default embedding
                audio_visual_text_embeds = self.audio_visual_text_head(
                    text_outputs.pooler_output
                )

        return PEAudioVisualOutput(
            audio_embeds=audio_embeds,
            audio_visual_embeds=audio_visual_embeds,
            visual_embeds=visual_embeds,
            audio_text_embeds=audio_text_embeds,
            audio_visual_text_embeds=audio_visual_text_embeds,
            visual_text_embeds=visual_text_embeds,
            audio_plus_text_embeds=audio_plus_text_embeds,
            visual_plus_text_embeds=visual_plus_text_embeds,
            audio_visual_output=audio_visual_outputs,
            text_output=text_outputs,
        )


class PEAudioFrame(BasePEAudio):
    config_cls = PEAudioFrameConfig

    def __init__(self, config: PEAudioFrameConfig):
        super().__init__()
        self.config = config
        self.text_model = AutoModel.from_config(config.text_model)
        self.audio_model = AudioEncoder(config.audio_model)
        self.text_head = ContrastiveHead(
            config.text_model.hidden_size, config.output_dim
        )
        self.audio_head = ContrastiveHead(
            config.audio_model.audio_transformer.hidden_size, config.output_dim
        )
        self.logit_scale = torch.nn.Parameter(torch.tensor([0.0]))
        self.logit_bias = torch.nn.Parameter(torch.tensor([0.0]))

    def _get_text_output(self, input_ids, attention_mask):
        nth_layer = self.config.nth_text_layer
        output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=nth_layer is not None,
        )
        if nth_layer is None:
            text_model_output = output.last_hidden_state
        else:
            text_model_output = output.hidden_states[nth_layer]

        return BaseModelOutputWithPooling(
            last_hidden_state=text_model_output, pooler_output=text_model_output[:, 0]
        )

    def forward(
        self,
        input_ids: torch.Tensor,  # tokenized text
        input_values: Optional[torch.Tensor] = None,  # audio waveform (may be None if input_features is provided)
        input_features: Optional[torch.Tensor] = None,  # codec_features (if already computed)
        attention_mask: Optional[torch.Tensor] = None,  # text attention mask
        padding_mask: Optional[torch.Tensor] = None,  # audio padding mask
        threshold: float = 0.3,
        return_spans: bool = True,
    ) -> PEAudioFrameOutput:
        audio_output = self.audio_model(input_values, padding_mask, input_features=input_features)
        text_model_output = self._get_text_output(input_ids, attention_mask)

        text_embeds = self.text_head(text_model_output.pooler_output)
        audio_embeds = self.audio_head(audio_output.last_hidden_state)

        spans = None
        if return_spans:
            bsz = input_ids.size(0)
            unscaled_logits = audio_embeds @ text_embeds.unsqueeze(1).transpose(-1, -2)
            logits = unscaled_logits.squeeze(-1) * self.logit_scale + self.logit_bias
            probs = logits.sigmoid()

            preds = probs > threshold
            # Find where predictions changed from False->True and True->False
            changes = torch.diff(F.pad(preds, (1, 1), value=False), dim=1).nonzero()
            span_tensor = torch.cat([changes[::2], changes[1::2, [1]]], dim=1)
            # Convert audio frame index to time
            dac_config = self.config.audio_model.dac_vae_encoder

            spans = [
                (
                    span_tensor[span_tensor[:, 0] == i, 1:]
                    * dac_config.hop_length
                    / dac_config.sampling_rate
                ).tolist()
                for i in range(bsz)
            ]

        return PEAudioFrameOutput(
            text_embeds=text_embeds,
            audio_embeds=audio_embeds,
            spans=spans,
            text_output=text_model_output,
            audio_output=audio_output,
        )


__all__ = ["PEAudioVisual", "PEAudioFrame"]
