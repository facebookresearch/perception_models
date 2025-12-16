from typing import Optional
from transformers import ModernBertConfig, DacConfig


DFLT_DACVAE_CONFIG = {
    "encoder_hidden_size": 64,
    "downsampling_ratios": [2, 8, 10, 12],
    "decoder_hidden_size": 1536,
    "n_codebooks": 16,
    "codebook_size": 1024,
    "codebook_dim": 128,
    "quantizer_dropout": 0,
    "sampling_rate": 48000,
}

DFLT_TEXT_ENCODER_CONFIG = {
    "classifier_pooling": "mean",
    "global_attn_every_n_layers": 3,
    "global_rope_theta": 160000.0,
    "hidden_size": 1024,
    "intermediate_size": 2624,
    "layer_norm_eps": 1e-5,
    "local_rope_theta": 10000.0,
    "model_type": "modernbert",
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "position_embedding_type": "absolute",
    "tie_word_embeddings": True,
    "torch_dtype": "float32",
}


class TransformerConfig:
    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=2752,
        num_hidden_layers=16,
        num_attention_heads=8,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=10_000,
        rms_norm_eps=1e-5,
        rope_theta=20000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout


class AudioEncoderConfig:
    def __init__(
        self,
        dac_vae_encoder: Optional[dict] = None,
        audio_transformer: Optional[dict] = None,
        **kwargs,
    ):
        dac_vae_encoder = dac_vae_encoder or DFLT_DACVAE_CONFIG
        audio_transformer = audio_transformer or {}
        self.dac_vae_encoder = DacConfig(**dac_vae_encoder)
        self.audio_transformer = TransformerConfig(**audio_transformer)


class VisualEncoderConfig:
    def __init__(
        self,
        pe_encoder: str = "PE-Core-L14-336",
        visual_transformer: Optional[dict] = None,
        fixed_len_video: bool = False,
        **kwargs,
    ):
        visual_transformer = visual_transformer or {}

        self.pe_encoder = pe_encoder
        self.visual_transformer = TransformerConfig(**visual_transformer)
        self.fixed_len_video = fixed_len_video


class PEAudioVisualEncoderConfig:
    def __init__(
        self,
        audio_visual_transformer: Optional[dict] = None,
        visual_model: Optional[dict] = None,
        audio_model: Optional[dict] = None,
        **kwargs,
    ):
        visual_model = visual_model or {}
        audio_visual_transformer = audio_visual_transformer or {}
        audio_model = audio_model or {}

        self.visual_model = VisualEncoderConfig(**visual_model)
        self.audio_model = AudioEncoderConfig(**audio_model)
        self.audio_visual_transformer = TransformerConfig(**audio_visual_transformer)


class PEAudioVisualConfig:
    def __init__(
        self,
        audio_visual_model: Optional[dict] = None,
        text_model: Optional[dict] = None,
        output_dim: int = 1024,
        nth_text_layer: Optional[int] = 22,
        **kwargs,
    ):
        text_model = text_model or DFLT_TEXT_ENCODER_CONFIG
        audio_visual_model = audio_visual_model or {}
        self.text_model = ModernBertConfig(**text_model)
        self.audio_visual_model = PEAudioVisualEncoderConfig(**audio_visual_model)
        self.output_dim = output_dim
        self.nth_text_layer = nth_text_layer


class PEAudioFrameConfig:
    def __init__(
        self,
        audio_model: Optional[dict] = None,
        text_model: Optional[dict] = None,
        output_dim: int = 1024,
        nth_text_layer: Optional[int] = 22,
        **kwargs,
    ):
        text_model = text_model or DFLT_TEXT_ENCODER_CONFIG
        audio_model = audio_model or {}
        self.text_model = ModernBertConfig(**text_model)
        self.audio_model = AudioEncoderConfig(**audio_model)
        self.output_dim = output_dim
        self.nth_text_layer = nth_text_layer
