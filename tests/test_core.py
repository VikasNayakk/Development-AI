import tempfile
import unittest
from pathlib import Path

import torch

from core.config import ModelConfig
from core.model import DevelopmentAIModel
from core.tokenizer import ByteTokenizer
from core.train import atomic_save_checkpoint, estimate_loss, get_batch, split_corpus


class TokenizerTests(unittest.TestCase):
    def test_round_trip_ascii_and_unicode(self):
        tokenizer = ByteTokenizer()
        text = "Development-AI नमस्ते"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_vocab_is_byte_sized(self):
        self.assertEqual(ByteTokenizer.vocab_size, 256)


class ConfigTests(unittest.TestCase):
    def test_rejects_invalid_dimensions_and_dropout(self):
        with self.assertRaises(ValueError):
            ModelConfig(d_model=30, n_heads=4).validate()
        with self.assertRaises(ValueError):
            ModelConfig(dropout=1.0).validate()


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(block_size=16, d_model=32, n_heads=4, n_layers=2)
        self.model = DevelopmentAIModel(self.config)

    def test_forward_shape_and_loss(self):
        tokens = torch.randint(0, 256, (2, 12))
        logits, loss = self.model(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (2, 12, 256))
        self.assertTrue(torch.isfinite(loss).item())

    def test_rejects_context_overflow(self):
        with self.assertRaises(ValueError):
            self.model(torch.randint(0, 256, (1, 17)))

    def test_generation_preserves_prompt_and_appends_tokens(self):
        self.model.eval()
        prompt = torch.tensor([[65, 66]])
        result = self.model.generate(prompt.clone(), 3, temperature=1.0)
        self.assertEqual(result.shape, (1, 5))
        self.assertTrue(torch.equal(result[:, :2], prompt))

    def test_rejects_invalid_generation_temperature(self):
        with self.assertRaises(ValueError):
            self.model.generate(torch.tensor([[65]]), 1, temperature=0)


class TrainingTests(unittest.TestCase):
    def test_split_is_disjoint_and_large_enough(self):
        tokens = torch.arange(100)
        train, validation = split_corpus(tokens, 0.2, 10)
        self.assertEqual(train[-1].item(), 79)
        self.assertEqual(validation[0].item(), 80)
        self.assertEqual(len(train) + len(validation), len(tokens))

    def test_rejects_too_small_corpus(self):
        with self.assertRaises(ValueError):
            split_corpus(torch.arange(20), 0.2, 16)

    def test_batch_shapes_and_next_token_alignment(self):
        data = torch.arange(30)
        x, y = get_batch(data, 4, 5, torch.device("cpu"))
        self.assertEqual(tuple(x.shape), (4, 5))
        self.assertTrue(torch.equal(y, x + 1))

    def test_evaluation_restores_training_mode(self):
        model = DevelopmentAIModel(
            ModelConfig(block_size=4, d_model=16, n_heads=2, n_layers=1)
        )
        model.train()
        data = torch.randint(0, 256, (40,))
        metrics = estimate_loss(
            model, data[:20], data[20:], batch_size=2, block_size=4,
            eval_iters=2, device=torch.device("cpu")
        )
        self.assertIn("train", metrics)
        self.assertIn("validation", metrics)
        self.assertTrue(model.training)

    def test_atomic_checkpoint_write(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.pt"
            atomic_save_checkpoint({"value": torch.tensor([1])}, target)
            loaded = torch.load(target, weights_only=True)
            self.assertEqual(loaded["value"].item(), 1)


if __name__ == "__main__":
    unittest.main()
