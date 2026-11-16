"""A deterministic UTF-8 byte tokenizer; vocabulary IDs are 0..255."""


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        return bytes(int(token) for token in token_ids).decode("utf-8", errors="replace")
