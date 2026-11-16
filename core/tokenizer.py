"""Stable UTF-8 byte tokenizer; IDs are exactly the byte values 0..255."""

from collections.abc import Sequence


class ByteTokenizer:
    vocab_size = 256
    name = "utf8-byte-v1"

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        return list(text.encode("utf-8"))

    def decode(self, token_ids: Sequence[int]) -> str:
        if not isinstance(token_ids, Sequence) or isinstance(token_ids, str):
            raise TypeError("token_ids must be a sequence of integers")
        for token in token_ids:
            if type(token) is not int:
                raise TypeError("Token IDs must be integers; booleans are not valid token IDs.")
            if not 0 <= token < self.vocab_size:
                raise ValueError("Token IDs must be in [0, 255].")
        return bytes(token_ids).decode("utf-8", errors="replace")
