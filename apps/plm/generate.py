# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

logging.basicConfig(level=logging.INFO)

import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask
from tqdm import tqdm

from apps.plm.tokenizer import PLMTokenizer, Tokenizer, build_tokenizer
from apps.plm.transformer import LMTransformer, LMTransformerArgs
from core.args import dataclass_from_dict
from core.checkpoint import load_consolidated_checkpoint
from core.transformer import (Attention, causal_mask, generate_doc_mask_mod,
                              lengths_to_local_ids, lengths_to_start_ids)
from core.transforms.image_transform import get_image_transform
from core.transforms.video_transform import get_video_transform


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def sample_top_k(probs, k):
    topk_value, _ = torch.topk(probs, k)  # batch_sz x topk
    min_value_top_k = topk_value[:, [-1]]
    probs[probs < min_value_top_k] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None):
    shape = logits.shape
    logits = logits.flatten(end_dim=-2)
    if temperature > 0.0:
        probs = torch.softmax(logits / temperature, dim=-1)

        if top_p is not None:
            next_token = sample_top_p(probs, top_p)
        elif top_k is not None:
            next_token = sample_top_k(probs, top_k)
        else:
            next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(logits, dim=-1)
    return next_token.view(shape[:-1])


def pack_prompts(prompts: List[int]):
    res = []
    lengths = []
    for i, p in enumerate(prompts):
        p = torch.tensor(p, dtype=torch.long)
        l = p.size(0)
        res.append(p)
        lengths.append(l)
    lengths = torch.tensor(lengths, dtype=torch.long)
    res = torch.cat(res)
    return res, lengths


def batch_prompts(prompts, max_elements, lengths=None):
    batches = []
    current_batch = []
    current_count = 0

    for i in range(len(prompts)):
        prt = prompts[i]
        prompt_size = len(prt) if lengths is None else lengths[i]
        if current_count + prompt_size <= max_elements:
            current_batch.append(prt)
            current_count += prompt_size
        else:
            if current_batch:  # Add the current batch to batches
                batches.append(current_batch)
            # Start a new batch with the current prompt
            current_batch = [prt]
            current_count = prompt_size

    # Add the last batch if it contains any prompts
    if current_batch:
        batches.append(current_batch)

    return batches


class KVCache(nn.Module):
    def __init__(self, bsz, seqlen, n_heads, head_dim, dtype, device):
        super().__init__()
        shape = (bsz, seqlen, n_heads, head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=dtype, device=device))
        self.register_buffer("v_cache", torch.zeros(shape, dtype=dtype, device=device))
        self.offset = 0

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.offset = 0

    def update(self, k_val, v_val, tok_idx):
        # input_pos: [B], k_val: [B, S, H, D]
        self.k_cache.index_copy_(1, self.offset + tok_idx, k_val)
        self.v_cache.index_copy_(1, self.offset + tok_idx, v_val)
        return self.k_cache, self.v_cache


@dataclass
class PackedCausalTransformerGeneratorArgs:
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    max_gen_len: int = 256
    max_tokens: int = 11264
    until: List[str] = field(default_factory=list)
    compile_prefilling: bool = False
    reduce_generation_overhead: bool = False
    show_progress: bool = False
    dtype: Optional[str] = "bf16"
    device: Optional[str] = "cuda"


class PackedCausalTransformerGenerator:
    def __init__(
        self,
        cfg: PackedCausalTransformerGeneratorArgs,
        model: nn.Module,
        tokenizer: Tokenizer,
    ):
        """
        This class wraps a causal transformer model with its corresponding tokenizer
        and provides an efficient way to pack prompts together and do generation on
        the packed sequence.

        For example, if we had the prompts "Hello, I am a " and "Initiating calibration "
        Then this class will concatenate those sequence (pack them together)
        "Hello, I am a Initiating calibration"
        And make the necessary attention masks such that a sequence only attends to itself
        during prefilling and generation.

        This class creates a fixed size cache of size max_tokens or sum of prompt sizes
        + the max number of generated tokens per sequence.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.top_k = cfg.top_k

        self.max_gen_len = cfg.max_gen_len
        self.max_tokens = cfg.max_tokens
        self.until = cfg.until
        self.max_until_size = max([len(e) for e in self.until]) if self.until else 1
        self.device = cfg.device

        # Compile if necessary
        self.prefill = torch.compile(self.prefill, disable=not cfg.compile_prefilling)
        self.generate_next_token = torch.compile(
            self.generate_next_token,
            backend="inductor",
            fullgraph=True,
            mode="reduce-overhead",  # Other availabel mode is "max-autotune"
            disable=not cfg.reduce_generation_overhead,
        )

        self.show_progress = cfg.show_progress
        self.dtype = dict(fp32=torch.float32, bf16=torch.bfloat16)[cfg.dtype]

        self.prefill_doc_id, self.prefill_tok_id = None, None
        self.padded_doc_id, self.padded_tok_id = None, None
        self.current_doc_id, self.current_tok_id = None, None
        self.padded_doc_start = None
        self.prefill_mask = None

    def clear_cache(self, offset):
        for module in self.model.modules():
            if isinstance(module, Attention):
                if not hasattr(module, "kv_cache"):
                    module.kv_cache = KVCache(
                        1,
                        self.max_tokens,
                        module.n_kv_heads,
                        module.head_dim,
                        self.dtype,
                        self.device,
                    )
                module.kv_cache.offset = offset

    @torch.compiler.disable
    def setup_prefilling(self, lengths: torch.Tensor):
        # The KV cache is a fixed size tensor of size max_tokens that we need
        # to update in order to do correct autoregressive generation.

        # Here we will generate token by token but on multiple sequences
        # at once. To do so, we need to have an attention mask that makes
        # each sequence independent.

        # Each sequence will write to its allocated space in the KV Cache.
        # We allocate len(seq) + max_gen_len to each sequence in the cache.

        # We will generate max_gen_len for each document
        padded_lengths = lengths + self.max_gen_len
        max_tokens = self.max_tokens or padded_lengths.sum().item()
        # The last document might have more padding to fill up to max_tokens
        padded_lengths[-1] += max_tokens - padded_lengths.sum()

        # This is the start index in the cache for each document
        self.padded_doc_start = lengths_to_start_ids(padded_lengths)
        # For example with ab--123--cdef--
        # this would be 0, 4, 9 if max_gen_len is 2

        # We repeat interleave to align with tokens for prefilling
        # Ex: ab--123--cdef--
        #     000044444999999
        prefill_offset = torch.repeat_interleave(self.padded_doc_start, lengths)
        # This offset will make sure the tokens are written to the
        # correct positions in the cache during prefilling

        # We either init the cache or clear it by resetting the offset to prefill_offset
        self.clear_cache(prefill_offset)

        # The prefilling mask looks like the following for
        # the two packed sequences ab and 123 : ab123
        # Where spaces are empty cache positions
        #                 keys
        #                ab---123---
        #   queries    a 10000000000
        #              b 11000000000
        #              1 00000100000
        #              2 00000110000
        #              3 00000111000
        # We make sure to skip the empty cache positions
        # and only attend to positions within the same sequence
        doc_mask_mod = generate_doc_mask_mod(causal_mask, lengths, padded_lengths)
        self.prefill_mask = create_block_mask(
            doc_mask_mod, 1, None, lengths.sum(), max_tokens
        )

        # This creates the prefilling token ids which look like
        # the following for the packed sequence abcdefg1234
        # abcdefg1234
        # 01234560123
        # The token id gives us the position within each sequence
        # This is used to compute ROPE and to update the cache
        # At each forward pass the current tokens are written to
        # offset + tok_id
        self.prefill_doc_id, self.prefill_tok_id = lengths_to_local_ids(lengths)

        # This creates the padded token and document ids
        # which look like the following for the packed sequence ab123
        #               ab---123---               ab---123---
        # padded_doc_id 00000111111 padded_tok_id 01234012345
        # This will later be useful for the attention mask at generation
        self.padded_doc_id, self.padded_tok_id = lengths_to_local_ids(padded_lengths)

    @torch.compiler.disable
    def setup_generation(self, lengths):
        # KV Cache offset is set to the start of the padded documents
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.kv_cache.offset = self.padded_doc_start
        # The token ids during generations correspond to the lengths of each doc
        # current_tok_id will be incremented during generation
        self.current_tok_id = lengths.clone()
        # Since we're generating one token per document
        # the document id is just an arange
        self.current_doc_id = torch.arange(lengths.size(0), device=lengths.device)

    @torch.compiler.disable
    def prepare_media_inputs(
        self, tokens, lengths, images, image_patch_text_ids, num_image_chunks
    ):
        image_pos_index = None
        num_chunks = []
        if images is not None and len(images) > 0:
            assert image_patch_text_ids is not None and len(
                image_patch_text_ids
            ) == len(images)
            assert num_image_chunks is not None and len(num_image_chunks) == len(images)
            image_pos_index = torch.full(tokens.shape, -1, dtype=torch.int).to(
                self.device
            )
            assert tokens.shape[0] == 1
            offsets = torch.roll(lengths.cpu(), shifts=1, dims=-1).numpy()
            offsets[0] = 0
            num_chunks_seq = 0
            image_id_offset = 0
            for image_id, offset in enumerate(offsets):
                num_image_tokens = len(image_patch_text_ids[image_id])
                image_indices = (
                    torch.arange(num_image_tokens, dtype=torch.int).to(self.device)
                    + image_id_offset
                )
                text_indices = [i + offset for i in image_patch_text_ids[image_id]]
                image_pos_index[0, text_indices] = image_indices
                image_id_offset += num_image_tokens
                num_chunks_seq += num_image_chunks[image_id]
            num_chunks.append(num_chunks_seq)
            # Move images to the same device and dtype as model parameters
            model_param = next(self.model.parameters())
            images = torch.cat(images).to(model_param)
        else:
            images = None
        return images, image_pos_index, num_chunks

    # From here on some methods for generation
    def prefill(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        images: Optional[List[torch.Tensor]] = None,
        image_patch_text_ids: Optional[List[List[int]]] = None,
        num_image_chunks: Optional[List[int]] = None,
    ):
        # Prefilling is done by taking multiple packed sequences and
        # doing block diagonal attention on them so they remain independent
        self.setup_prefilling(lengths=lengths)
        images, image_pos_index, num_chunks = self.prepare_media_inputs(
            tokens, lengths, images, image_patch_text_ids, num_image_chunks
        )
        prefill_out = self.model.forward(
            tokens,
            tok_idx=self.prefill_tok_id,
            mask="causal",
            images=images,
            image_pos_index=image_pos_index,
            num_chunks=num_chunks,
            attn_impl="sdpa",
        )
        self.setup_generation(lengths=lengths)
        return prefill_out

    def generate_next_token(self, current_token):
        # Since we're doing generation with multiple sequences at once
        # we need to ignore tokens and cache entries from other sequences
        # or in the future.
        # Example mask :
        #                  keys
        #                abc--1234--
        #   queries    c 11100000000
        #              4 00000111100

        # mask shape : (n_seqs, cache_size)
        doc_mask = self.current_doc_id.unsqueeze(1) == self.padded_doc_id.unsqueeze(0)
        caus_mask = self.current_tok_id.unsqueeze(1) >= self.padded_tok_id.unsqueeze(0)
        mask = doc_mask & caus_mask
        out = self.model.forward(
            current_token,
            tok_idx=self.current_tok_id,  # n_seqs
            mask=mask,
            attn_impl="sdpa",
        )
        self.current_tok_id += 1
        return out

    @torch.inference_mode()
    def generate(self, prompts):
        images = []
        image_patch_text_ids = []
        num_image_chunks = []
        # Tokenize
        if isinstance(self.tokenizer, PLMTokenizer):
            encoded_prompts = []
            for p in prompts:
                assert isinstance(p, tuple)
                assert len(p) == 2
                question, image = p

                images.append(image)
                text_ids, image_pos = self.tokenizer._tokenize_for_generation(
                    question, image
                )
                num_chunks = image.size(0)

                encoded_prompts.append(text_ids)
                image_patch_text_ids.append(image_pos)
                num_image_chunks.append(num_chunks)
            prompts = encoded_prompts
        else:
            prompts = [
                self.tokenizer.encode(p, add_bos=False, add_eos=False) for p in prompts
            ]

        # Account for the generation in lengths
        padded_lengths = [len(p) + self.max_gen_len for p in prompts]
        generation = []
        loglikelihood = []
        greedy = []
        it = batch_prompts(prompts, self.max_tokens, lengths=padded_lengths)
        if self.show_progress:
            it = tqdm(it)
        for batch in it:
            n_seqs = len(batch)
            generated_tokens = [[] for _ in range(n_seqs)]
            is_done = [False for _ in range(n_seqs)]
            packed_batch, lengths = pack_prompts(batch)
            packed_batch, lengths = packed_batch.cuda(), lengths.cuda()
            n_seqs = lengths.size(0)
            current_images = images[:n_seqs]
            current_image_patch_text_ids = image_patch_text_ids[:n_seqs]
            current_num_image_chunks = num_image_chunks[:n_seqs]
            images = images[n_seqs:]
            image_patch_text_ids = image_patch_text_ids[n_seqs:]
            num_image_chunks = num_image_chunks[n_seqs:]

            # Prefilling cache
            prompt_logits = self.prefill(
                packed_batch.unsqueeze(0),
                lengths,
                images=current_images,
                image_patch_text_ids=current_image_patch_text_ids,
                num_image_chunks=current_num_image_chunks,
            )
            # Selecting last token in each prompt
            all_tokens = sample_tokens(
                prompt_logits, self.temperature, self.top_p, self.top_k
            )
            start_token = all_tokens[:, lengths.cumsum(0) - 1]

            for seq_id, tok in enumerate(start_token.squeeze(0).tolist()):
                generated_tokens[seq_id].append(tok)

            current_token = start_token
            for i in range(1, self.max_gen_len):

                next_logits = self.generate_next_token(current_token)
                next_token = sample_tokens(
                    next_logits.clone(), self.temperature, self.top_p, self.top_k
                )

                for seq_id, tok in enumerate(next_token.squeeze(0).tolist()):
                    if not is_done[seq_id]:
                        generated_tokens[seq_id].append(tok)
                        current_end_str = self.tokenizer.decode(
                            generated_tokens[seq_id][-self.max_until_size :]
                        )
                        contains_end_string = any(
                            [e in current_end_str for e in self.until]
                        )
                        is_done[seq_id] = (
                            contains_end_string
                            or tok == self.tokenizer.eot_id
                            or tok == self.tokenizer.eos_id
                        )
                if all(is_done):
                    break

                current_token = next_token

            generation.extend([self.tokenizer.decode(g) for g in generated_tokens])

            for p, logit in zip(
                batch, prompt_logits.squeeze(0).split(lengths.tolist())
            ):
                x = logit[:-1]
                y = torch.tensor(p[1:], device=x.device)
                loglikelihood.append(-F.cross_entropy(x, y, reduction="none").cpu())
                greedy.append((x.argmax(dim=-1) == y).cpu())

        generation = [
            response.replace("<|eot_id|>", "").replace("<|end_of_text|>", "")
            for response in generation
        ]
        return generation, loglikelihood, greedy


def load_consolidated_model_and_tokenizer(ckpt):
    # Download the model from Hugging Face if not available locally.
    if os.path.exists(ckpt):
        ckpt_path = ckpt  # It's a local path
    else:
        try:
            print(f"Downloading {ckpt} from Hugging Face Hub...")
            ckpt_path = snapshot_download(ckpt)
            print(f"Downloaded to: {ckpt_path}")
        except Exception as e:
            # Handle exceptions, such as model not found on HF Hub
            print(f"An error occurred while downloading {ckpt}: {e}")
            return

    # Load params from model config
    config = os.path.join(ckpt_path, "params.json")
    config = OmegaConf.load(config)

    # Build tokenizer
    tokenizer = build_tokenizer(
        config.data.tokenizer_name,
        (
            config.data.tokenizer_path
            if os.path.exists(config.data.tokenizer_path)
            else os.path.join(ckpt_path, config.data.tokenizer_path)
        ),
        pooling_ratio=config.model.pooling_ratio,
        patch_size=config.model.vision_model.patch_size,
    )

    # Build model and load the consolidated checkpoints
    model_args = dataclass_from_dict(LMTransformerArgs, config.model, strict=False)
    model = LMTransformer(model_args)
    load_consolidated_checkpoint(model, ckpt_path)
    param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
        config.distributed.model_dtype
    ]
    model = model.cuda().eval()
    for param in model.parameters():
        param.data = param.data.to(dtype=param_dtype)

    return model, tokenizer, config


def main(args):
    # Load model and tokenizer
    model, tokenizer, config = load_consolidated_model_and_tokenizer(args.ckpt)
    media_type = args.media_type
    media_path = args.media_path
    question = args.question

    prompts = []
    if media_type == "image":
        transform = get_image_transform(
            vision_input_type=config.data.vision_input_type,
            image_res=model.vision_model.image_size,
            max_num_tiles=config.data.max_num_tiles,
        )
        image = Image.open(media_path).convert("RGB")
        image, _ = transform(image)
        prompts.append((question, image))
    elif media_type == "video":
        transform = get_video_transform(
            image_res=model.vision_model.image_size,
        )
        video_info = (media_path, config.data.max_video_frames, None, None, None)
        frames, _ = transform(video_info)
        prompts.append((question, frames))
    else:
        raise NotImplementedError(
            f"The provided generate function only supports image and video."
        )

    # Create generator
    gen_cfg = dataclass_from_dict(
        PackedCausalTransformerGeneratorArgs, {}, strict=False
    )
    generator = PackedCausalTransformerGenerator(gen_cfg, model, tokenizer)

    # Run generation
    start_time = time.time()
    generation, loglikelihood, greedy = generator.generate(prompts)
    end_time = time.time()

    for i, gen in enumerate(generation):
        # Calculate tokens per second
        total_tokens = sum(
            len(tokenizer.encode(gen, False, False)) for gen in generation
        )
        tokens_per_second = total_tokens / (end_time - start_time)

        print("==============================================")
        print(f"\nPrompt {i+1}: {prompts[i][0]}")
        print(f"Generated Text: {gen}")
        print(f"Tokens per second: {tokens_per_second:.2f}")
        print("==============================================")


def parse_args():
    parser = argparse.ArgumentParser(description="Run a multimodal language model")

    # Define CLI arguments that will be used to override YAML config
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to the checkpoint directory."
    )
    parser.add_argument(
        "--media_type",
        type=str,
        choices=["image", "video"],
        required=True,
        help="Type of media (image or video)",
    )
    parser.add_argument(
        "--media_path", type=str, required=True, help="Path to the media file"
    )
    parser.add_argument(
        "--question", type=str, required=True, help="Question or prompt for the model."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
