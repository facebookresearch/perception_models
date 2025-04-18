# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE.PE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# OpenCLIP: https://github.com/mlfoundations/open_clip
# --------------------------------------------------------

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from huggingface_hub import PyTorchModelHubMixin

from core.vision_encoder.pev1 import TextTransformer, VisionTransformer


@dataclass
class CLIPVisionCfg:
    layers: int = 50
    width: int = 1536
    head_width: int = 96
    heads: int = 16
    mlp_ratio: float = 5.833333333333333
    patch_size: int = 14
    image_size: Union[Tuple[int, int], int] = 224
    embed_cls_token: bool = False

    image_size: int = 448
    output_dim: int = 1280
    vision_select_feature: str = "pooled"

    use_rope2d: bool = True
    abs_pos_embed: bool = True
    pool_type: str = "attn"

    relative_pos_embed_type: str = (
        "rope_2d"  # Relative posemb in attn: ["", "rope_1d", "rope_2d"]
    )
    pos_embed_type: str = (
        "learnable"  # Absolute position embedding: ["", "learnable", "sin_cos_2d"]
    )
    global_layers: int = (
        -1
    )  # (-1 for all) If windowing, number of global layers (evenly dispersed starting from the end)


@dataclass
class CLIPTextCfg:
    context_length: int = 72
    vocab_size: int = 49408
    hf_tokenizer_name: Optional[str] = None
    tokenizer_kwargs: Optional[dict] = None

    width: int = 1280
    heads: int = 20
    layers: int = 24
    mlp_ratio: float = 4.0
    output_dim: int = 1280

    ls_init_value: Optional[float] = None  # layer scale initial value
    embed_cls: bool = False
    pad_id: int = 0
    no_causal_mask: bool = False  # disable causal masking
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = "argmax"
    proj_bias: bool = False

    output_tokens: bool = False
    act_kwargs: dict = None
    norm_kwargs: dict = None
    use_ln_post: bool = True

    norm_type: str = "layernorm"  # "layernorm" or "rmsnorm"


class CLIP(nn.Module,
           PyTorchModelHubMixin,
           repo_url="https://github.com/facebookresearch/perception_models",
           pipeline_tag="image-feature-extraction",
           license="apache-2.0"):
    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,  # change to dict for fast develop
        text_cfg: CLIPTextCfg,  # change to dict for fast develop
        quick_gelu: bool = False,
    ):
        super(CLIP, self).__init__()

        if isinstance(vision_cfg, dict):
            vision_cfg = CLIPVisionCfg(**vision_cfg)

        self.visual = VisionTransformer(
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_cfg.heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            embed_cls_token=vision_cfg.embed_cls_token,
            use_rope2d=vision_cfg.use_rope2d,
            abs_pos_embed=vision_cfg.abs_pos_embed,
            image_size=vision_cfg.image_size,
            output_dim=vision_cfg.output_dim,
            vision_select_feature=vision_cfg.vision_select_feature,
            pool_type=vision_cfg.pool_type,
        )

        if isinstance(text_cfg, dict):
            text_cfg = CLIPTextCfg(**text_cfg)

        text = TextTransformer(
            width=text_cfg.width,
            layers=text_cfg.layers,
            heads=text_cfg.heads,
            mlp_ratio=text_cfg.mlp_ratio,
            output_dim=text_cfg.output_dim,
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            pool_type=text_cfg.pool_type,
            use_ln_post=text_cfg.use_ln_post,
            ls_init_value=text_cfg.ls_init_value,
            embed_cls=text_cfg.embed_cls,
            no_causal_mask=text_cfg.no_causal_mask,
            proj_bias=text_cfg.proj_bias,
        )
        self.transformer = text.transformer

        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.text_pool_type = text.pool_type
        self.register_buffer("attn_mask", text.attn_mask, persistent=False)

        if self.text_pool_type == "attn" or self.text_pool_type == "attn_eos":
            self.text_attn_pool = text.attn_pool
        self.text_global_pool = text.text_global_pool

    def encode_image(self, image, normalize: bool = False, features=None):
        if features is None:
            features = self.visual(image)
        if hasattr(features, "pooled"):
            out = features.pooled
        else:
            out = features
        out = F.normalize(out, dim=-1) if normalize else out

        if hasattr(features, "target") and features.target is not None:
            return out, features.latent, features.target
        else:
            return out

    def encode_video(self, video, num_frames, normalize: bool = False):
        n = len(video)
        b, c, h, w = video[0].shape
        assert num_frames == n
        frms = torch.stack(video).permute(1, 0, 2, 3, 4).reshape(b * n, c, h, w)
        frm_feats = self.encode_image(frms, normalize=normalize)
        video_feats = frm_feats.reshape(b, num_frames, -1)
        video_feats = video_feats.mean(dim=1)

        return video_feats

    def encode_text(self, text, normalize: bool = False):
        # cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(
            text
        )  # .to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding  # .to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]

        if self.text_pool_type == "attn":
            x = self.text_attn_pool(x).squeeze(1)
        elif self.text_pool_type == "attn_eos":
            # Only use the eos token for the clip token, but emulate going through a self-attn layer
            # so that the text distribution can more closely match the vision distribution
            tokens = self.ln_final(x)
            pool_x, _ = text_global_pool(tokens, text, pool_type="argmax")
            x = self.text_attn_pool(pool_x.unsqueeze(1)).squeeze(1)
        else:
            x, _ = self.text_global_pool(x, text, self.text_pool_type)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return F.normalize(x, dim=-1) if normalize else x

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
    ):
        image_features = (
            self.encode_image(image, normalize=True, features=image_features)
            if image is not None
            else None
        )
        text_features = (
            self.encode_text(text, normalize=True) if text is not None else None
        )
        return image_features, text_features  # , self.logit_scale.exp()


def set_model_preprocess_cfg(model, preprocess_cfg: Dict[str, Any]):
    module = getattr(model, "visual", model)
    module.image_mean = preprocess_cfg[
        "mean"
    ]  # legacy attribute, keeping for bwd compat
    module.image_std = preprocess_cfg["std"]  # legacy attribute, keeping for bwd compat
    module.preprocess_cfg = copy.deepcopy(
        preprocess_cfg
    )  # new attr, package all pp cfg as dict


def resize_pos_embed(
    state_dict, model, interpolation: str = "bicubic", antialias: bool = True
):
    # Absolute position embedding uses a sampling strategy now and doesn't need to be resized
    return

    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get("visual.positional_embedding", None)
    if old_pos_embed is None or not hasattr(model.visual, "grid_size"):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = (
        1  # FIXME detect different token configs (ie no class token, or more)
    )
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = (
            old_pos_embed[:extra_tokens],
            old_pos_embed[extra_tokens:],
        )
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info(
        "Resizing position embedding grid-size from %s to %s", old_grid_size, grid_size
    )
    pos_emb_img = pos_emb_img.reshape(
        1, old_grid_size[0], old_grid_size[1], -1
    ).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(
        1, grid_size[0] * grid_size[1], -1
    )[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict["visual.positional_embedding"] = new_pos_embed


def resize_text_pos_embed(
    state_dict, model, interpolation: str = "linear", antialias: bool = False
):
    pos_key = "positional_embedding"
    old_pos_embed = state_dict.get(pos_key, None)
    if old_pos_embed is None:
        pos_key = "text.positional_embedding"
        old_pos_embed = state_dict.get(pos_key, None)
    if old_pos_embed is None:
        return
    # FIXME add support for text cls_token
    model_pos_embed = getattr(model, "positional_embedding", None)
    if model_pos_embed is None:
        model_pos_embed = getattr(model.text, "positional_embedding", None)

    old_num_pos = old_pos_embed.shape[0]
    old_width = old_pos_embed.shape[1]
    num_pos = model_pos_embed.shape[0]
    width = model_pos_embed.shape[1]
    assert old_width == width, "text pos_embed width changed!"
    if old_num_pos == num_pos:
        return

    logging.info(
        "Resizing text position embedding num_pos from %s to %s", old_num_pos, num_pos
    )
    old_pos_embed = old_pos_embed.reshape(1, old_num_pos, old_width).permute(0, 2, 1)
    old_pos_embed = F.interpolate(
        old_pos_embed,
        size=num_pos,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    old_pos_embed = old_pos_embed.permute(0, 2, 1)[0]
    new_pos_embed = old_pos_embed

    state_dict[pos_key] = new_pos_embed


if __name__ == "__main__":
    vision_cfg = CLIPVisionCfg
    text_cfg = CLIPTextCfg

    model = CLIP(
        embed_dim=1280,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=False,
    )

    ckpt_path = "/checkpoint/vision_encoder/berniehuang/share/pev1_gs14_448_rc2.pt"

    _sd = torch.load(ckpt_path)
    if "state_dict" in _sd:
        _sd = _sd["state_dict"]
    _sd_new = {}
    for key, value in _sd.items():
        if key.startswith(
            "module."
        ):  # extract the visual tower and remove 'module.' prefix
            new_key = key.replace("module.", "")
            _sd_new[new_key] = value
    _sd = _sd_new

    m, u = model.load_state_dict(_sd, strict=False)
    print(f"Missing keys for loading CLIP: {m}")
    print(f"Unexpected keys for loading CLIP: {u}")

    model = model.cuda()
    x = torch.randn(8, 3, 224, 224).cuda()
    t = torch.randint(0, 10001, (8, 72)).cuda()
    print("image shape:", x.shape)
    print("text shape:", t.shape)

    x_feat = model.encode_image(x)
    t_feat = model.encode_text(t)
    print("image feature shape:", x_feat.shape)
    print("text feature shape:", t_feat.shape)
