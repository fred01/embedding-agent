"""
BGE-M3 dense embeddings for the book search system (fred01/search-lib).

Model: BAAI/bge-m3, dense vector (CLS, L2-normalized), dimension 1024.
The indexer accepts only these vectors: they must match the ones already stored in Qdrant.
"""

from .bge_m3 import (
    EMBEDDING_DIMENSION,
    MAX_LENGTH,
    MODEL_NAME,
    HttpEmbedder,
    TorchEmbedder,
    check_reference,
    decode_vector,
    detect_device,
    encode_vector,
    measure_rate,
)
from .bge_m3_mlx import MlxEmbedder, mlx_available

__all__ = [
    "EMBEDDING_DIMENSION",
    "MAX_LENGTH",
    "MODEL_NAME",
    "HttpEmbedder",
    "TorchEmbedder",
    "check_reference",
    "decode_vector",
    "detect_device",
    "encode_vector",
    "measure_rate",
    "MlxEmbedder",
    "mlx_available",
]
