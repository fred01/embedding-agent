"""
BGE-M3 dense embeddings: the only vectors the indexer accepts.

The dense BGE-M3 vector is the hidden state of the first (CLS) token of the last layer, L2-normalized.
This is exactly what FlagEmbedding's BGEM3FlagModel returns as `dense_vecs`; here it is computed with plain
transformers, which works on CUDA, Apple Silicon (MPS) and CPU and does not pin old library versions.
Vectors must stay compatible with the ~47M already stored in Qdrant: `check_reference()` compares against a
vector computed with FlagEmbedding and refuses to run if they drift apart.
"""

import base64
import json
import os
import time
from typing import List, Optional

import numpy as np

MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024
MAX_LENGTH = 8192

REFERENCE_FILE = os.path.join(os.path.dirname(__file__), "reference.json")


def detect_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_batch_size(device: str) -> int:
    if device.startswith("cuda"):
        return 32
    if device == "mps":
        return 16
    return 4


class TorchEmbedder:
    """BGE-M3 dense embeddings with PyTorch (cuda / mps / cpu)."""

    backend = "torch"

    def __init__(self, device: Optional[str] = None, fp16: Optional[bool] = None,
                 batch_size: Optional[int] = None, max_length: int = MAX_LENGTH):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or detect_device()
        self.fp16 = (self.device != "cpu") if fp16 is None else fp16
        self.batch_size = batch_size or default_batch_size(self.device)
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        try:
            model = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="sdpa")
        except (ValueError, TypeError, ImportError):
            # older transformers / model class without SDPA support
            model = AutoModel.from_pretrained(MODEL_NAME)
        if self.fp16:
            model = model.half()
        self.model = model.to(self.device).eval()

        if self.device == "cpu":
            torch.set_num_threads(int(os.getenv("CPU_THREADS", "0")) or os.cpu_count() or 1)

    @property
    def device_label(self) -> str:
        if self.device.startswith("cuda"):
            idx = int(self.device.split(":")[1]) if ":" in self.device else 0
            return f"{self.torch.cuda.get_device_name(idx)} ({'fp16' if self.fp16 else 'fp32'})"
        if self.device == "mps":
            return f"Apple GPU / MPS ({'fp16' if self.fp16 else 'fp32'})"
        return f"CPU x{self.torch.get_num_threads()} ({'fp16' if self.fp16 else 'fp32'})"

    def embed(self, texts: List[str]) -> np.ndarray:
        """L2-normalized dense vectors, float32, shape (len(texts), 1024)."""
        if not texts:
            return np.zeros((0, EMBEDDING_DIMENSION), dtype=np.float32)
        torch = self.torch
        out = np.zeros((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)
        # longest first: batches of similar length waste less on padding
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        pos = 0
        batch_size = self.batch_size
        while pos < len(order):
            idx = order[pos:pos + batch_size]
            try:
                out[idx] = self._embed_batch([texts[i] for i in idx])
            except RuntimeError as e:
                if "out of memory" not in str(e).lower() or batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                self.batch_size = batch_size
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                elif self.device == "mps":
                    torch.mps.empty_cache()
                print(f"Out of memory, batch size lowered to {batch_size}", flush=True)
                continue
            pos += len(idx)
        return out

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        torch = self.torch
        encoded = self.tokenizer(texts, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            hidden = self.model(**encoded).last_hidden_state[:, 0]
            vectors = torch.nn.functional.normalize(hidden.float(), dim=-1)
        return vectors.cpu().numpy()


class HttpEmbedder:
    """
    Any OpenAI-compatible /embeddings endpoint serving BGE-M3 dense vectors: HuggingFace
    text-embeddings-inference, infinity, LiteLLM... Fast GPU servers batch requests themselves.
    """

    backend = "http"

    def __init__(self, url: str, model: str = MODEL_NAME, api_key: Optional[str] = None,
                 batch_size: int = 32, timeout: int = 600):
        import requests

        self.url = url.rstrip("/")
        if not self.url.endswith("/embeddings"):
            self.url += "/embeddings"
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"
        self.device = "http"

    @property
    def device_label(self) -> str:
        return f"{self.url} ({self.model})"

    def embed(self, texts: List[str]) -> np.ndarray:
        out = []
        for pos in range(0, len(texts), self.batch_size):
            batch = texts[pos:pos + self.batch_size]
            response = self.session.post(self.url, json={"model": self.model, "input": batch}, timeout=self.timeout)
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda d: d["index"])
            out.extend(d["embedding"] for d in data)
        vectors = np.asarray(out, dtype=np.float32).reshape(len(texts), -1)
        if vectors.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError(f"{self.url} returned {vectors.shape[1]}-dim vectors, expected {EMBEDDING_DIMENSION}")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)


def encode_vector(vector: np.ndarray) -> str:
    """Wire format of the indexer: base64 of little-endian float32."""
    return base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode("ascii")


def decode_vector(data: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(data), dtype="<f4")


def load_reference() -> dict:
    with open(REFERENCE_FILE, encoding="utf-8") as f:
        return json.load(f)


def check_reference(embedder, min_cosine: float = 0.995) -> float:
    """
    Embed the reference text and compare with the vector FlagEmbedding (fp32, CPU) produced for it.
    fp16 on GPU/MPS gives ~0.9999; anything below min_cosine means a different model or a broken setup,
    and its vectors would not be comparable with those already in Qdrant.
    """
    reference = load_reference()
    expected = decode_vector(reference["vector"])
    actual = embedder.embed([reference["text"]])[0]
    cosine = float(np.dot(expected, actual) / (np.linalg.norm(expected) * np.linalg.norm(actual)))
    if cosine < min_cosine:
        raise RuntimeError(
            f"Reference check failed: cosine {cosine:.5f} < {min_cosine} "
            f"({embedder.device_label}). Vectors would not match the ones stored in Qdrant."
        )
    return cosine


def measure_rate(embedder, chunks: int) -> float:
    """Chunks per second on full-size (600-word) chunks; also warms the model up."""
    text = load_reference()["text"]
    words = text.split()
    sample = " ".join((words * (600 // max(1, len(words)) + 1))[:600])
    embedder.embed([sample])  # warm-up
    started = time.time()
    embedder.embed([sample] * chunks)
    return chunks / max(time.time() - started, 1e-6)
