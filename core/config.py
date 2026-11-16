from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 256
    block_size: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1

    def validate(self) -> None:
        if self.vocab_size != 256:
            raise ValueError("The initial ByteTokenizer requires vocab_size=256.")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if min(self.block_size, self.d_model, self.n_heads, self.n_layers) <= 0:
            raise ValueError("Model dimensions must be positive.")

    def to_dict(self) -> dict:
        return asdict(self)
