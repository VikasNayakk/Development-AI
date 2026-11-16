"""Small decoder-only causal Transformer, initialized from random weights."""
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
        causal_mask = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1
        )
        normalized = self.norm1(x)
        attended, _ = self.attention(
            normalized, normalized, normalized, attn_mask=causal_mask, need_weights=False
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
        batch, length = token_ids.shape
        if length > self.config.block_size:
            raise ValueError("Input exceeds configured block_size.")
        positions = torch.arange(length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.language_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.config.vocab_size), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")
        self.eval()
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.block_size :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(next_logits, dim=-1), num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids
