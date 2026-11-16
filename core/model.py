"""Small decoder-only causal Transformer initialized from random weights."""
import math

import torch
from torch import nn

from .config import ModelConfig


class CausalSelfAttentionBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        mask = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1
        )
        normalized = self.norm1(x)
        attended, _ = self.attention(
            normalized, normalized, normalized, attn_mask=mask, need_weights=False
        )
        x = x + self.dropout(attended)
        return x + self.feed_forward(self.norm2(x))


class DevelopmentAIModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.blocks = nn.ModuleList(
            [CausalSelfAttentionBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.language_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize_weights)
        self.language_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None):
        if (
            not isinstance(token_ids, torch.Tensor)
            or token_ids.ndim != 2
            or token_ids.size(0) == 0
            or token_ids.size(1) == 0
        ):
            raise ValueError("token_ids must have shape [non-empty batch, non-empty sequence].")
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must use torch.long dtype.")
        _, length = token_ids.shape
        if length > self.config.block_size:
            raise ValueError("Input exceeds configured block_size.")
        positions = torch.arange(length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.language_head(self.final_norm(x))
        loss = None
        if targets is not None:
            if (
                not isinstance(targets, torch.Tensor)
                or targets.shape != token_ids.shape
                or targets.dtype != torch.long
            ):
                raise ValueError("targets must be torch.long and match token_ids shape.")
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.config.vocab_size), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        *,
        do_sample: bool = True,
    ):
        """Append tokens with sampling by default or deterministic greedy decoding."""
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError("generation prompt must be a torch.Tensor.")
        if (
            token_ids.ndim != 2
            or token_ids.size(0) == 0
            or token_ids.size(1) == 0
        ):
            raise ValueError("generation prompt must have shape [non-empty batch, non-empty sequence].")
        if token_ids.dtype != torch.long:
            raise TypeError("generation prompt must use torch.long dtype.")
        if token_ids.device != self.token_embedding.weight.device:
            raise ValueError("generation prompt and model must be on the same device.")
        if bool(((token_ids.min() < 0) | (token_ids.max() >= self.config.vocab_size)).item()):
            raise ValueError(
                f"generation token IDs must be in [0, {self.config.vocab_size - 1}]."
            )
        if type(max_new_tokens) is not int:
            raise TypeError("max_new_tokens must be an integer.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative.")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise TypeError("temperature must be a finite number.")
        try:
            temperature_is_finite = math.isfinite(temperature)
        except OverflowError:
            temperature_is_finite = False
        if not temperature_is_finite or temperature <= 0:
            raise ValueError("temperature must be finite and greater than zero.")
        if not isinstance(do_sample, bool):
            raise TypeError("do_sample must be a boolean.")

        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                # Long prompts are accepted; only the most recent context window is used.
                context = token_ids[:, -self.config.block_size :]
                logits, _ = self(context)
                next_logits = logits[:, -1, :] / temperature
                if do_sample:
                    next_token = torch.multinomial(
                        torch.softmax(next_logits, dim=-1), num_samples=1
                    )
                else:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                token_ids = torch.cat((token_ids, next_token), dim=1)
            return token_ids
        finally:
            self.train(was_training)
