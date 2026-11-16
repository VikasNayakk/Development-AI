import copy
import unittest
from unittest.mock import patch

import torch

from core.config import ModelConfig
from core.model import DevelopmentAIModel
from core.train import optimizer_update


class GradientAccumulationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.config = ModelConfig(
            vocab_size=256,
            block_size=4,
            d_model=16,
            n_heads=2,
            n_layers=1,
            dropout=0.0,
        )
        self.model = DevelopmentAIModel(self.config)
        self.model.train()
        self.micro_batches = [
            (
                torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long),
                torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]], dtype=torch.long),
            ),
            (
                torch.tensor([[10, 11, 12, 13], [14, 15, 16, 17]], dtype=torch.long),
                torch.tensor([[11, 12, 13, 14], [15, 16, 17, 18]], dtype=torch.long),
            ),
        ]

    def test_accumulated_update_matches_effective_batch_gradient_and_clipping(self):
        reference_model = copy.deepcopy(self.model)
        accumulated_model = copy.deepcopy(self.model)
        reference_optimizer = torch.optim.AdamW(
            reference_model.parameters(), lr=1e-3, weight_decay=0.01
        )
        accumulated_optimizer = torch.optim.AdamW(
            accumulated_model.parameters(), lr=1e-3, weight_decay=0.01
        )
        inputs = torch.cat([batch[0] for batch in self.micro_batches], dim=0)
        targets = torch.cat([batch[1] for batch in self.micro_batches], dim=0)

        reference_optimizer.zero_grad(set_to_none=True)
        _, reference_loss = reference_model(inputs, targets)
        reference_loss.backward()
        reference_grad_norm = torch.nn.utils.clip_grad_norm_(
            reference_model.parameters(), max_norm=0.05, error_if_nonfinite=True
        )
        reference_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in reference_model.named_parameters()
            if parameter.grad is not None
        }
        reference_optimizer.step()

        captured_gradients = {}
        original_step = accumulated_optimizer.step

        def capture_gradients_then_step(*args, **kwargs):
            captured_gradients.update({
                name: parameter.grad.detach().clone()
                for name, parameter in accumulated_model.named_parameters()
                if parameter.grad is not None
            })
            return original_step(*args, **kwargs)

        with (
            patch("core.train.get_batch", side_effect=self.micro_batches) as get_batch,
            patch.object(
                accumulated_optimizer,
                "step",
                side_effect=capture_gradients_then_step,
            ) as optimizer_step,
        ):
            result = optimizer_update(
                accumulated_model,
                accumulated_optimizer,
                torch.arange(100, dtype=torch.long),
                batch_size=2,
                block_size=4,
                gradient_accumulation_steps=2,
                device=torch.device("cpu"),
                max_grad_norm=0.05,
            )

        self.assertEqual(get_batch.call_count, 2)
        self.assertEqual(optimizer_step.call_count, 1)
        self.assertEqual(result["micro_batches"], 2)
        self.assertEqual(result["effective_batch_size"], 4)
        self.assertAlmostEqual(result["loss"], float(reference_loss.item()), places=6)
        self.assertAlmostEqual(
            result["gradient_norm"], float(reference_grad_norm.item()), places=5
        )
        self.assertEqual(set(captured_gradients), set(reference_gradients))
        for name, expected in reference_gradients.items():
            actual = captured_gradients[name]
            self.assertTrue(
                torch.allclose(expected, actual, rtol=1e-4, atol=2e-7),
                f"Accumulated pre-step gradient differs from effective-batch gradient for {name}; "
                f"max_abs_error={(expected - actual).abs().max().item():.3e}",
            )
        self.assertTrue(all(
            state["step"].item() == 1
            for state in accumulated_optimizer.state.values()
        ))

    def test_one_micro_batch_matches_a_standard_optimizer_update(self):
        reference_model = copy.deepcopy(self.model)
        accumulated_model = copy.deepcopy(self.model)
        reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=1e-3)
        accumulated_optimizer = torch.optim.AdamW(accumulated_model.parameters(), lr=1e-3)
        inputs, targets = self.micro_batches[0]

        reference_optimizer.zero_grad(set_to_none=True)
        _, loss = reference_model(inputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference_model.parameters(), 1.0, error_if_nonfinite=True)
        reference_optimizer.step()

        with patch("core.train.get_batch", return_value=(inputs, targets)):
            result = optimizer_update(
                accumulated_model,
                accumulated_optimizer,
                torch.arange(100, dtype=torch.long),
                batch_size=2,
                block_size=4,
                gradient_accumulation_steps=1,
                device=torch.device("cpu"),
            )

        self.assertEqual(result["effective_batch_size"], 2)
        self.assertAlmostEqual(result["loss"], float(loss.item()), places=6)
        for (name, expected), (_, actual) in zip(
            reference_model.named_parameters(),
            accumulated_model.named_parameters(),
        ):
            self.assertTrue(
                torch.allclose(expected, actual, rtol=1e-6, atol=1e-8),
                f"Single-batch update differs for {name}",
            )

    def test_rejects_zero_or_non_integer_accumulation_steps(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    optimizer_update(
                        self.model,
                        optimizer,
                        torch.arange(100, dtype=torch.long),
                        batch_size=2,
                        block_size=4,
                        gradient_accumulation_steps=value,
                        device=torch.device("cpu"),
                    )

    def test_non_finite_micro_batch_loss_cancels_optimizer_update_and_clears_gradients(self):
        class NonFiniteLossModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, inputs, targets):
                return None, self.weight * torch.tensor(float("nan"))

        model = NonFiniteLossModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        original = model.weight.detach().clone()
        batch = (torch.ones((2, 4), dtype=torch.long), torch.ones((2, 4), dtype=torch.long))

        with patch("core.train.get_batch", return_value=batch):
            with self.assertRaises(FloatingPointError):
                optimizer_update(
                    model,
                    optimizer,
                    torch.arange(100, dtype=torch.long),
                    batch_size=2,
                    block_size=4,
                    gradient_accumulation_steps=2,
                    device=torch.device("cpu"),
                )

        self.assertTrue(torch.equal(model.weight.detach(), original))
        self.assertIsNone(model.weight.grad)
        self.assertEqual(optimizer.state, {})

    def test_validation_failure_does_not_apply_an_optimizer_update(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        originals = [p.detach().clone() for p in self.model.parameters()]
        with self.assertRaises(ValueError):
            optimizer_update(
                self.model,
                optimizer,
                torch.arange(100, dtype=torch.long),
                batch_size=2,
                block_size=4,
                gradient_accumulation_steps=0,
                device=torch.device("cpu"),
            )
        for before, after in zip(originals, self.model.parameters()):
            self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
