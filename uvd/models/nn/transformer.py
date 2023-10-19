"""Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch.nn import functional as F

from uvd.models.nn.net_base import NetBase

__all__ = ["GPTConfig", "GPT"]


@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = None
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_llama_impl: bool = True
    position_embed: Literal["relative", "rotary", "absolute"] | None = "relative"


KVCache = tuple[torch.Tensor, torch.Tensor]
MaskCache = torch.Tensor
RoPECache = torch.Tensor


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, position):
        """
        Args:
            position: Tensor, shape [batch_size, seq_len]
        """
        B, L = position.shape
        position = position.unsqueeze(-1)  # BxLx1
        pe = torch.zeros([B, L, self.d_model], device=position.device)
        pe[:, :, 0::2] = torch.sin(position * self.div_term)
        pe[:, :, 1::2] = torch.cos(position * self.div_term)
        return pe


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived
    from:er_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License: https://github.com/labml
    ai/annotated_deep_learning_paper_implementations/blob/master/license
    .
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (
        base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem)
    )

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).float()

    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        cache = cache.half()
    return cache


def apply_rope(x: torch.Tensor, rope_cache: RoPECache) -> torch.Tensor:
    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]

    # cast because the reference does
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias.

    PyTorch doesn't support simply bias=False
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from
    https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py.
    BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed

    def extra_repr(self) -> str:
        return f"{(self.scale.data.shape[0],)}, dim={self.dim}, eps={self.eps}"


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        self.pos_embed_type = config.position_embed

    def forward(
        self,
        x: torch.Tensor,
        *,
        pos_emb: torch.Tensor | RoPECache | None = None,
        mask: MaskCache,
        max_seq_length: int | None = None,
        input_pos: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size)
        q = q.view(B, T, self.n_head, head_size)
        v = v.view(B, T, self.n_head, head_size)

        if pos_emb is not None and self.pos_embed_type == "rotary":
            q = apply_rope(q, pos_emb)
            k = apply_rope(k, pos_emb)

        k = k.transpose(1, 2)  # (B, nh, T, hs)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor([max_seq_length - 1], device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=2)
                cache_v = torch.roll(cache_v, -1, dims=2)
            k = cache_k.index_copy(2, input_pos, k)
            v = cache_v.index_copy(2, input_pos, v)
            kv_cache = k, v

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout * int(self.training)
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, kv_cache


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert not config.use_llama_impl
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


class LLaMA_MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.use_llama_impl
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)

        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, bias=config.bias)
        self.silu = nn.SiLU()
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, bias=config.bias)
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.use_llama_impl = config.use_llama_impl
        if self.use_llama_impl:
            self.ln_1 = RMSNorm(config.n_embd)
            self.ln_2 = RMSNorm(config.n_embd)
            self.mlp = LLaMA_MLP(config)
        else:
            self.ln_1 = LayerNorm(config.n_embd, config.bias)
            self.ln_2 = LayerNorm(config.n_embd, config.bias)
            self.mlp = MLP(config)
        self.attn = CausalSelfAttention(config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        pos_emb: torch.Tensor | RoPECache | None = None,
        mask: MaskCache,
        max_seq_length: int,
        input_pos: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        h, new_kv_cache = self.attn(
            self.ln_1(x),
            pos_emb=pos_emb,
            mask=mask,
            max_seq_length=max_seq_length,
            input_pos=input_pos,
            kv_cache=kv_cache,
        )
        x = x + h
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv_cache


class GPT(NetBase):
    def __init__(
        self,
        gpt_config: GPTConfig,
        input_shape: int | None = None,
        use_wte: bool = False,
    ):
        super().__init__()
        if input_shape is None:
            input_shape = gpt_config.vocab_size
        assert input_shape is not None
        gpt_config.vocab_size = input_shape

        assert gpt_config.block_size is not None
        self.config = gpt_config
        if not use_wte:
            assert gpt_config.vocab_size == gpt_config.n_embd

        position_embed = gpt_config.position_embed
        assert position_embed in ["relative", "absolute", "rotary", None]
        self.position_embed_type = position_embed
        if position_embed == "rotary":
            self.pos_emb: RoPECache | None = None
        elif position_embed == "relative":
            self.pos_emb = nn.Parameter(
                torch.zeros(1, gpt_config.block_size, gpt_config.n_embd)
            )
        elif position_embed == "absolute":
            self.pos_emb = PositionalEncoder(gpt_config.n_embd)
        elif position_embed is None:
            self.pos_emb = None
        else:
            raise NotImplementedError

        self.transformer = nn.ModuleDict(
            dict(
                # rep -> emb dim
                wte=nn.Linear(gpt_config.vocab_size, gpt_config.n_embd)
                if use_wte
                else nn.Identity(),
                # wpe=nn.Embedding(config.block_size, config.n_embd),
                # drop=nn.Dropout(gpt_config.dropout),
                h=nn.ModuleList([Block(gpt_config) for _ in range(gpt_config.n_layer)]),
                ln_f=RMSNorm(gpt_config.n_embd)
                if gpt_config.use_llama_impl
                else LayerNorm(gpt_config.n_embd, gpt_config.bias),
            )
        )

        self.kv_caches: list[KVCache] = []
        self.mask_cache: MaskCache | None = None

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * gpt_config.n_layer)
                )

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def build_mask_cache(self, x: torch.Tensor) -> MaskCache:
        ones = torch.ones(
            (self.config.block_size, self.config.block_size),
            device=x.device,
            dtype=torch.bool,
        )
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        rep: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        max_seq_length: int | None = None,
        input_pos: int | None = None,
    ):
        if rep.ndim == 2:
            assert self.config.block_size == 1, self.config.block_size
            rep = rep[:, None, :]
        b, t, _ = rep.size()

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size

        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        assert (
            max_seq_length <= block_size
        ), f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert (
            t <= block_size
        ), f"Cannot forward sequence of length {t}, block size is only {block_size}"

        if self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(rep)
        use_rotary = self.position_embed_type == "rotary"
        if use_rotary and self.pos_emb is None:
            self.pos_emb = build_rope_cache(
                seq_len=self.config.block_size,
                n_elem=self.config.n_embd // self.config.n_head,
                dtype=torch.int64,
                device=rep.device,
            )

        pos_emb = None
        if input_pos is not None:
            if use_rotary:
                pos_emb = self.pos_emb.index_select(0, input_pos)
            elif self.position_embed_type == "relative":
                pos_emb = self.pos_emb[:, input_pos, :]
            elif self.position_embed_type == "absolute":
                pos_emb = self.pos_emb(timesteps)
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            if use_rotary:
                pos_emb = self.pos_emb[:t]
            elif self.position_embed_type == "relative":
                pos_emb = self.pos_emb[:, :t, :]
            elif self.position_embed_type == "absolute":
                pos_emb = self.pos_emb(timesteps)
            mask = self.mask_cache[:, :, :t, :t]

        # forward the GPT model itself
        x = self.transformer.wte(rep)  # token embeddings of shape (b, t, n_embd)

        if self.position_embed_type in ["relative", "absolute"]:
            x = x + pos_emb
        # x = self.transformer.drop(x)

        if input_pos is None:  # proxy for use_cache=False
            for block in self.transformer.h:
                x, _ = block(
                    x, pos_emb=pos_emb, max_seq_length=max_seq_length, mask=mask
                )
        else:
            if not self.kv_caches:
                head_size = self.config.n_embd // self.config.n_head
                cache_shape = (b, self.config.n_head, max_seq_length, head_size)
                self.kv_caches = [
                    (
                        torch.zeros(cache_shape, device=x.device, dtype=x.dtype),
                        torch.zeros(cache_shape, device=x.device, dtype=x.dtype),
                    )
                    for _ in range(self.config.n_layer)
                ]
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(
                    x,
                    pos_emb=pos_emb,
                    max_seq_length=max_seq_length,
                    mask=mask,
                    input_pos=input_pos,
                    kv_cache=self.kv_caches[i],
                )

        x = self.transformer.ln_f(x)  # (b, t, n_embd)
        return x

    def reset_cache(self) -> None:
        self.kv_caches.clear()
        if self.mask_cache is not None and self.mask_cache.device.type == "xla":
            # https://github.com/Lightning-AI/lit-parrot/pull/83#issuecomment-1558150179
            self.mask_cache = None
            if self.position_embed_type == "rotary":
                self.pos_emb = None

    @property
    def output_dim(self) -> tuple | int:
        return self.config.n_embd

    def _pos_emb_repr(self) -> str | None:
        if self.position_embed_type == "relative":
            return f"Parameter({tuple(self.pos_emb.data.shape)}, requires_grad={self.pos_emb.requires_grad})"
        elif self.position_embed_type == "rotary":
            return '"Rotary"'
        elif self.position_embed_type == "absolute":  # already nn.module
            return None
        elif self.position_embed_type is None:
            return "None"
        else:
            raise NotImplementedError

    def extra_repr(self) -> str:
        return (
            super().extra_repr() + f"\n(block_size): {self.config.block_size}"
            f"\n(pos_emb): {self._pos_emb_repr()}" * (self._pos_emb_repr() is not None)
        )


if __name__ == "__main__":
    """Test kv cache."""
    import time
    import os

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    block_size = 500
    bs = 2
    n_emb = 768
    position_embed: Literal["relative", "rotary", "absolute"] | None = "absolute"

    max_seq_length = 500

    config = GPTConfig(
        n_embd=n_emb,
        block_size=block_size,
        n_layer=8,
        n_head=8,
        position_embed=position_embed,
    )

    model = GPT(gpt_config=config, input_shape=n_emb, use_wte=True).eval()
    model = model.to("cuda")
    print(model)

    xs = torch.randn(bs, block_size, n_emb, device="cuda")
    input_pos = torch.arange(block_size, device=xs.device)
    timesteps = (
        torch.arange(0, block_size)
        .repeat(bs)
        .reshape(bs, block_size)
        .to(input_pos.device)
    )

    out = None
    t1 = time.time()
    for i in range(block_size):
        with torch.no_grad():
            out = model(
                xs[:, [i], :],
                timesteps=timesteps[:, [i]],
                max_seq_length=max_seq_length,
                input_pos=input_pos[i][None] % max_seq_length,
            )
    print(time.time() - t1)
    print("use cache", out)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = GPT(gpt_config=config, input_shape=n_emb, use_wte=True).eval()
    model = model.to("cuda")

    t1 = time.time()
    out2 = None
    for i in range(block_size):
        with torch.no_grad():
            out2 = model(
                xs[:, max(0, i + 1 - max_seq_length) : i + 1, :],
                timesteps=timesteps[:, max(0, i + 1 - max_seq_length) : i + 1],
                input_pos=None,
                max_seq_length=max_seq_length,
            )
    print(time.time() - t1)
    print("not use cache", out2[:, [-1], :])

    torch.testing.assert_close(out, out2[:, [-1], :])
