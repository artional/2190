import unittest

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from ack_defense.model import Reader


class Batch(dict):
    def to(self, device):
        return Batch({key: value.to(device) for key, value in self.items()})


class Tokenizer:
    def pad(self, data, **kwargs):
        sequences = data["input_ids"]
        length = max(map(len, sequences))
        return Batch(input_ids=torch.tensor([s + [0] * (length - len(s)) for s in sequences]),
                     attention_mask=torch.tensor([[1] * len(s) + [0] * (length - len(s)) for s in sequences]))


class ReaderTests(unittest.TestCase):
    def test_last_valid_block_output_and_hook_removal(self):
        torch.manual_seed(3)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                           num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2)).eval()
        reader = Reader.__new__(Reader)
        reader.model, reader.body = model, model.model
        reader.layers, reader.device, reader.tokenizer = [1, 2], torch.device("cpu"), Tokenizer()
        sequences = [[1, 4, 7], [1, 2, 3, 6, 8]]
        actual = reader.raw(sequences)
        batch = reader.tokenizer.pad(dict(input_ids=sequences))
        with torch.inference_mode():
            full = model.model(**batch, output_hidden_states=True, use_cache=False)
        positions = batch["attention_mask"].sum(1) - 1
        expected = torch.stack([full.hidden_states[i][torch.arange(2), positions] for i in (1, 2)], 1).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        self.assertTrue(all(len(layer._forward_hooks) == 0 for layer in model.model.layers))
        reader.close()


if __name__ == "__main__":
    unittest.main()
