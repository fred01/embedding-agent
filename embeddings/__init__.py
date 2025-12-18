"""
Shared embedding utilities for the book search system.

This module provides the canonical implementation for computing text embeddings
using BGE-M3 model. All services that need to compute embeddings MUST use this
module to ensure consistency across the system.

Model: BAAI/bge-m3
Dimension: 1024
Max Length: 8192
"""

from .bge_embeddings import (
    MODEL_NAME,
    EMBEDDING_DIMENSION,
    MAX_LENGTH,
    load_model,
    compute_embedding,
    compute_embeddings_batch
)

__all__ = [
    'MODEL_NAME',
    'EMBEDDING_DIMENSION',
    'MAX_LENGTH',
    'load_model',
    'compute_embedding',
    'compute_embeddings_batch'
]
