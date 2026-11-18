import gc

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import add_probe, format_prompt


class StopReading(Exception):
    pass


class Reader:
    def __init__(self, specification, layers, device="cuda:0", batch_size=4, token_budget=1024, max_length=4096):
        self.layers = list(layers)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.token_budget = token_budget
        self.max_length = max_length
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.tokenizer = AutoTokenizer.from_pretrained(specification["base"], use_fast=False, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define a pad or EOS token")
        self.tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
            specification["base"], torch_dtype=torch.float32,
            device_map={"": str(self.device)}, low_cpu_mem_usage=True,
            local_files_only=True, attn_implementation="sdpa",
        )
        if specification.get("adapter"):
            model = PeftModel.from_pretrained(model, specification["adapter"], is_trainable=False, local_files_only=True)
        self.model = model.eval().requires_grad_(False)
        base = self.model.get_base_model() if isinstance(self.model, PeftModel) else self.model
        self.body = base.model
        if not self.layers or min(self.layers) < 1 or max(self.layers) > len(self.body.layers):
            raise ValueError("Layer indices are one-based transformer block indices")
        self.hidden_size = self.model.config.hidden_size

    def encode(self, rows):
        result = [self.tokenizer.encode(format_prompt(row), add_special_tokens=True) for row in rows]
        if any(len(ids) > self.max_length for ids in result):
            raise ValueError("Prompt exceeds max_length; no silent truncation is performed")
        return result

    @torch.inference_mode()
    def raw(self, sequences):
        batch = self.tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt").to(self.device)
        positions = batch["attention_mask"].sum(1) - 1
        captured, handles = {}, []

        def make_hook(layer):
            def hook(module, arguments, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer] = hidden[torch.arange(len(sequences), device=hidden.device), positions].detach().clone()
                if layer == max(self.layers):
                    raise StopReading()
            return hook

        try:
            for layer in self.layers:
                handles.append(self.body.layers[layer - 1].register_forward_hook(make_hook(layer)))
            try:
                self.body(**batch, use_cache=False, output_hidden_states=False, return_dict=True)
            except StopReading:
                pass
            result = torch.stack([captured[layer] for layer in self.layers], dim=1).float().cpu().numpy()
        finally:
            for handle in handles:
                handle.remove()
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite hidden states")
        return result

    def extract(self, rows):
        sequences = self.encode(rows)
        result = np.empty((len(rows), len(self.layers), self.hidden_size), dtype=np.float32)
        order = sorted(range(len(rows)), key=lambda i: (len(sequences[i]), i))
        cursor = 0
        while cursor < len(order):
            size = min(self.batch_size, len(order) - cursor)
            while size > 1 and size * len(sequences[order[cursor + size - 1]]) > self.token_budget:
                size -= 1
            indices = order[cursor:cursor + size]
            result[indices] = self.raw([sequences[i] for i in indices])
            cursor += size
        return result

    def responses(self, rows):
        baseline = self.extract(rows)
        probed = self.extract([add_probe(row) for row in rows])
        return probed - baseline

    def close(self):
        del self.body, self.model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
