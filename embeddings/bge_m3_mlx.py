"""
BGE-M3 dense embeddings on Apple Silicon with MLX.

The same XLM-RoBERTa forward pass as transformers (post-LN, exact GELU, CLS of the last layer, L2 norm), written
directly in MLX with the fused scaled_dot_product_attention. On an M5 Max it is ~1.5x faster than PyTorch on MPS.
On first start the weights are taken from the transformers model and cached as fp16 safetensors.
"""

import os
from typing import List, Optional

import numpy as np

import time

from .bge_m3 import EMBEDDING_DIMENSION, MAX_LENGTH, MODEL_NAME, DutyCycle

CACHE_DIR = os.path.expanduser(os.getenv("MLX_CACHE_DIR", "~/.cache/embedding-agent"))


def mlx_available() -> bool:
    try:
        import mlx.core as mx

        return mx.metal.is_available()
    except ImportError:
        return False


def _build_model(config: dict):
    import mlx.core as mx
    import mlx.nn as nn

    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    eps = config["layer_norm_eps"]
    pad = config["pad_token_id"]

    class SelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(hidden, hidden)
            self.key = nn.Linear(hidden, hidden)
            self.value = nn.Linear(hidden, hidden)

        def __call__(self, x, mask):
            b, n, _ = x.shape
            q, k, v = (proj(x).reshape(b, n, heads, -1).transpose(0, 2, 1, 3)
                       for proj in (self.query, self.key, self.value))
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=(hidden // heads) ** -0.5, mask=mask)
            return out.transpose(0, 2, 1, 3).reshape(b, n, hidden)

    class DenseNorm(nn.Module):
        def __init__(self, dims_in):
            super().__init__()
            self.dense = nn.Linear(dims_in, hidden)
            self.LayerNorm = nn.LayerNorm(hidden, eps=eps)

        def __call__(self, x, residual):
            return self.LayerNorm(self.dense(x) + residual)

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            setattr(self, "self", SelfAttention())
            self.output = DenseNorm(hidden)

        def __call__(self, x, mask):
            return self.output(getattr(self, "self")(x, mask), x)

    class Intermediate(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(hidden, config["intermediate_size"])

        def __call__(self, x):
            return nn.gelu(self.dense(x))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()
            self.intermediate = Intermediate()
            self.output = DenseNorm(config["intermediate_size"])

        def __call__(self, x, mask):
            x = self.attention(x, mask)
            return self.output(self.intermediate(x), x)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = [Layer() for _ in range(config["num_hidden_layers"])]

    class Embeddings(nn.Module):
        def __init__(self):
            super().__init__()
            self.word_embeddings = nn.Embedding(config["vocab_size"], hidden)
            self.position_embeddings = nn.Embedding(config["max_position_embeddings"], hidden)
            self.token_type_embeddings = nn.Embedding(config["type_vocab_size"], hidden)
            self.LayerNorm = nn.LayerNorm(hidden, eps=eps)

        def __call__(self, input_ids):
            not_pad = (input_ids != pad).astype(mx.int32)
            positions = mx.cumsum(not_pad, axis=1) * not_pad + pad
            x = (self.word_embeddings(input_ids) + self.position_embeddings(positions)
                 + self.token_type_embeddings(mx.zeros_like(input_ids)))
            return self.LayerNorm(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = Embeddings()
            self.encoder = Encoder()

        def __call__(self, input_ids, attention_mask):
            x = self.embeddings(input_ids)
            mask = attention_mask.astype(mx.bool_)[:, None, None, :]
            for layer in self.encoder.layer:
                x = layer(x, mask)
            cls = x[:, 0].astype(mx.float32)
            return cls / mx.linalg.norm(cls, axis=-1, keepdims=True)

    return Model()


def _weights_file(dtype: str) -> str:
    path = os.path.join(CACHE_DIR, f"bge-m3-mlx-{dtype}.safetensors")
    if os.path.exists(path):
        return path
    import mlx.core as mx
    from transformers import AutoModel

    print(f"Converting {MODEL_NAME} weights for MLX ({dtype}), once...", flush=True)
    state = AutoModel.from_pretrained(MODEL_NAME).state_dict()
    weights = {k: mx.array(v.float().numpy()).astype(getattr(mx, dtype))
               for k, v in state.items() if not k.startswith("pooler.") and "position_ids" not in k}
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = path.replace(".safetensors", ".tmp.safetensors")
    mx.save_safetensors(tmp, weights)
    os.replace(tmp, path)
    return path


class MlxEmbedder:
    """BGE-M3 dense embeddings with MLX on the Apple GPU."""

    backend = "mlx"
    device = "mlx"

    def __init__(self, batch_size: Optional[int] = None, dtype: str = "float16", max_length: int = MAX_LENGTH):
        import mlx.core as mx
        from transformers import AutoConfig, AutoTokenizer

        self.mx = mx
        self.dtype = dtype
        self.batch_size = batch_size or 16
        self.duty_cycle = DutyCycle.from_env()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = _build_model(AutoConfig.from_pretrained(MODEL_NAME).to_dict())
        self.model.load_weights(_weights_file(dtype))
        self.model.eval()
        mx.eval(self.model.parameters())

    @property
    def device_label(self) -> str:
        return f"Apple GPU / MLX ({'fp16' if self.dtype == 'float16' else self.dtype})"

    def embed(self, texts: List[str]) -> np.ndarray:
        """L2-normalized dense vectors, float32, shape (len(texts), 1024)."""
        mx = self.mx
        out = np.zeros((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        for pos in range(0, len(order), self.batch_size):
            idx = order[pos:pos + self.batch_size]
            encoded = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="np")
            started = time.time()
            vectors = self.model(mx.array(encoded["input_ids"]), mx.array(encoded["attention_mask"]))
            out[idx] = np.array(vectors)
            self.duty_cycle.pause(time.time() - started)
        return out
