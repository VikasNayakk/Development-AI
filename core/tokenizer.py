"""Stable UTF-8 byte tokenizer; IDs are exactly the byte values 0..255."""


class ByteTokenizer:
    vocab_size = 256
    name = "utf8-byte-v1"

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        if any(not isinstance(token, int) or token < 0 or token >= self.vocab_size
               for token in token_ids):
            raise ValueError("Token IDs must be integers in [0, 255].")
        return bytes(token_ids).decode("utf-8", errors="replace")
