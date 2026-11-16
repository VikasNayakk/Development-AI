import unittest

import torch

from core.config import ModelConfig
from core.model import DevelopmentAIModel
from core.tokenizer import ByteTokenizer


class TokenizerTests(unittest.TestCase):
    def test_round_trip_ascii_and_unicode(self):
        tokenizer = ByteTokenizer()
        text = "Development-AI नमस्ते"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_vocab_is_byte_sized(self):
        self.assertEqual(ByteTokenizer.vocab_size, 256)


class ModelTests(unittest.TestCase):
    def test_forward_shape_and_loss(self):
        config = ModelConfig(block_size=16, d_model=32, n_heads=4, n_layers=2)
        model = DevelopmentAIModel(config)
        tokens = torch.randint(0, 256, (2, 12))
        logits, loss = model(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (2, 12, 256))
        self.assertTrue(torch.isfinite(loss).item())

    def test_rejects_context_overflow(self):
        config = ModelConfig(block_size=8, d_model=32, n_heads=4, n_layers=1)
        model = DevelopmentAIModel(config)
        with self.assertRaises(ValueError):
            model(torch.randint(0, 256, (1, 9)))


if __name__ == "__main__":
    unittest.main()
