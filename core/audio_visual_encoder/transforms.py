from .config import PEAudioVisualConfig, PEAudioFrameConfig
from transformers import AutoTokenizer
from transformers import BatchFeature
from typing import Optional
import logging
import torch
from torchcodec.decoders import AudioDecoder, VideoDecoder
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as T
from ..vision_encoder.config import PE_VISION_CONFIG
from huggingface_hub import snapshot_download
import os
import json


logger = logging.getLogger(__name__)


AudioInput = torch.Tensor | list[torch.Tensor] | str | list[str]
VideoInput = AudioInput


class AudioProcessor:
    def __init__(
        self,
        sampling_rate: int = 48_000,
        hop_length: int = 1920,
        **kwargs,
    ):
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length

    def _reflect_pad(self, wav):
        if wav.size(-1) % self.hop_length == 0:
            return wav
        p1d = (0, self.hop_length - (wav.size(-1) % self.hop_length))
        return torch.nn.functional.pad(wav, p1d, mode="reflect")

    def _load_audio(self, path: str):
        ad = AudioDecoder(path, sample_rate=self.sampling_rate, num_channels=1)
        return ad.get_all_samples().data

    def __call__(
        self,
        raw_audio: AudioInput,
        sampling_rate: Optional[int] = None,
    ) -> BatchFeature:
        from_file = False
        if isinstance(raw_audio, str):
            raw_audio = [raw_audio]

        if isinstance(raw_audio, (list, tuple)) and isinstance(raw_audio[0], str):
            loaded = []
            for audio_file in raw_audio:
                loaded.append(self._load_audio(audio_file))
            raw_audio = loaded
            from_file = True

        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided audio input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        elif not from_file:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        if isinstance(raw_audio, list):
            raw_audio = [self._reflect_pad(x).T for x in raw_audio]
        else:
            raw_audio = self._reflect_pad(raw_audio).T

        # verify inputs are valid
        for example in raw_audio:
            if example.ndim > 2:
                raise ValueError(
                    f"Expected input shape (channels, num_samples), but got shape ({example.shape})"
                )

        lengths = torch.tensor([x.size(0) for x in raw_audio])
        input_values = pad_sequence(raw_audio, batch_first=True).transpose(1, 2)
        padding_mask = torch.arange(lengths.max())[None] < lengths[:, None]

        return BatchFeature(
            {"input_values": input_values, "padding_mask": padding_mask}
        )


def pixel_to_float(x):
    return x.float() / 255.0

class VideoProcessor:
    def __init__(self, fixed_len_video: bool, image_size: int):
        self.fixed_len_video = fixed_len_video
        self.frame_transform = T.Compose(
            [
                T.Resize(
                    (image_size, image_size), interpolation=T.InterpolationMode.BILINEAR
                ),
                T.Lambda(pixel_to_float),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
            ]
        )

    def _sample_frames(self, total_frames=None, tensor=None):
        start_idx = 0
        num_frames = 16
        if tensor is not None:
            total_frames = tensor.size(0)
        idxs = [
            start_idx + int(i * (total_frames - 1) / (num_frames - 1))
            for i in range(num_frames)
        ]
        if tensor is None:
            return idxs
        else:
            return tensor[idxs]

    def _load_video(self, path: str):
        vd = VideoDecoder(path)
        if self.fixed_len_video:
            frame_idxs = self._sample_frames(vd.metadata.num_frames_from_header)
            return vd.get_frames_at(frame_idxs).data
        return vd[:].data

    def __call__(self, raw_video: VideoInput):
        if isinstance(raw_video, str):
            raw_video = [raw_video]

        if isinstance(raw_video, (list, tuple)) and isinstance(raw_video[0], str):
            loaded = []
            for video_file in raw_video:
                loaded.append(self._load_video(video_file))
            raw_video = loaded
        elif self.fixed_len_video:
            # video frames were passed in as tensors, but we need to sample fixed frames
            sampled = []
            for video in raw_video:
                idxs = self._sample_frames(video.size(0))
                assert video.size(0) >= len(idxs), (
                    f"Video is not long enough, 16 frames are needed, but only found {len(video)}"
                )
                sampled.append(video[idxs])
            raw_video = sampled

        transformed = [self.frame_transform(v) for v in raw_video]
        lengths = torch.tensor([v.size(0) for v in transformed])
        padding_mask = torch.arange(lengths.max())[None] < lengths[:, None]
        pixel_values = pad_sequence(transformed, batch_first=True)
        return BatchFeature(
            {"pixel_values_videos": pixel_values, "padding_mask_videos": padding_mask}
        )


class PEAudioVisualTransform:
    def __init__(self, tokenizer, audio_processor, video_processor):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.video_processor = video_processor

    @classmethod
    def from_config(cls, name_or_checkpoint: str):
        if os.path.isdir(name_or_checkpoint):
            checkpoint_dir = name_or_checkpoint
        else:
            checkpoint_dir = snapshot_download(
                repo_id=f"facebook/{name_or_checkpoint}", revision="perception_models"
            )
        config_path = os.path.join(checkpoint_dir, "config.json")
        config_dict = json.load(open(config_path))
        config = PEAudioVisualConfig(**config_dict)
        pe_vision_config = PE_VISION_CONFIG[
            config.audio_visual_model.visual_model.pe_encoder
        ]
        return cls(
            tokenizer=AutoTokenizer.from_pretrained(checkpoint_dir),
            audio_processor=AudioProcessor(
                sampling_rate=config.audio_visual_model.audio_model.dac_vae_encoder.sampling_rate,
                hoplength=config.audio_visual_model.audio_model.dac_vae_encoder.hop_length,
            ),
            video_processor=VideoProcessor(
                fixed_len_video=config.audio_visual_model.visual_model.fixed_len_video,
                image_size=pe_vision_config.image_size,
            ),
        )

    def __call__(
        self,
        text: Optional[str] = None,
        audio: Optional[str | list[str] | torch.Tensor | list[torch.Tensor]] = None,
        videos: Optional[str | list[str] | torch.Tensor | list[torch.Tensor]] = None,
        pe_features: Optional[list[torch.Tensor]] = None,
        audio_codec_features: Optional[list[torch.Tensor]] = None,
        sampling_rate: Optional[int] = None,
    ):
        batch = BatchFeature()
        if text is not None:
            batch.update(
                self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=512,
                )
            )

        if audio is not None:
            batch.update(self.audio_processor(audio, sampling_rate=sampling_rate))

        if audio_codec_features is not None:
            batch["input_features"] = pad_sequence(audio_codec_features, batch_first=True)
            if "padding_mask" in batch:
                batch["padding_mask"] = batch["padding_mask"][:, :: self.audio_processor.hop_length]

        if videos is not None:
            batch.update(self.video_processor(videos))

        if pe_features is not None:
            if self.video_processor.fixed_len_video:
                pe_features = [self.video_processor._sample_frames(tensor=v) for v in pe_features]
            batch["pe_features"] = pad_sequence(pe_features, batch_first=True)

        return batch


class PEAudioFrameTransform:
    def __init__(self, tokenizer, audio_processor):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor

    @classmethod
    def from_config(cls, name_or_checkpoint: str):
        if os.path.isdir(name_or_checkpoint):
            checkpoint_dir = name_or_checkpoint
        else:
            checkpoint_dir = snapshot_download(
                repo_id=f"facebook/{name_or_checkpoint}", revision="perception_models"
            )
        config_path = os.path.join(checkpoint_dir, "config.json")
        config_dict = json.load(open(config_path))
        config = PEAudioFrameConfig(**config_dict)
        return cls(
            tokenizer=AutoTokenizer.from_pretrained(checkpoint_dir),
            audio_processor=AudioProcessor(
                sampling_rate=config.audio_model.dac_vae_encoder.sampling_rate,
                hoplength=config.audio_model.dac_vae_encoder.hop_length,
            ),
        )

    def __call__(
        self,
        text: Optional[str] = None,
        audio: Optional[str | list[str] | torch.Tensor | list[torch.Tensor]] = None,
        sampling_rate: Optional[int] = None,
    ):
        batch = BatchFeature()
        if text is not None:
            batch.update(
                self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=512,
                )
            )

        if audio is not None:
            batch.update(self.audio_processor(audio, sampling_rate=sampling_rate))

        return batch
