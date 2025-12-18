"""
BGE-M3 Embedding Model Utilities

This is the CANONICAL implementation for computing embeddings in the book search system.
All embedding computations MUST use these functions to ensure consistency.

Configuration:
- Model: BAAI/bge-m3 (BGE-M3 Flag Model)
- Embedding Dimension: 1024
- Max Text Length: 8192 tokens
- Batch Size: Configurable (default: 1 for single texts)

Usage:
    from embeddings import load_model, compute_embedding
    
    model = load_model(device='cuda', use_fp16=True)
    embedding = compute_embedding(model, "Your text here")
"""

import warnings
import logging
from typing import List, Optional
import torch
from FlagEmbedding import BGEM3FlagModel

# Suppress tokenizer warnings
warnings.filterwarnings('ignore', message='.*XLMRobertaTokenizerFast.*')
logging.getLogger('transformers').setLevel(logging.ERROR)

# Model configuration - these are the canonical values
MODEL_NAME = 'BAAI/bge-m3'
EMBEDDING_DIMENSION = 1024
MAX_LENGTH = 8192
DEFAULT_BATCH_SIZE = 1


def detect_device() -> str:
    """
    Detect the best available device for ML computations.
    
    Returns:
        str: Device name - 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def load_model(device: Optional[str] = None, use_fp16: Optional[bool] = None) -> BGEM3FlagModel:
    """
    Load the BGE-M3 embedding model.
    
    This is the canonical way to load the model. All services should use this function.
    
    Args:
        device: Device to use ('cuda', 'mps', 'cpu'). If None, auto-detects.
        use_fp16: Whether to use FP16 precision. If None, uses FP16 for GPU, FP32 for CPU.
    
    Returns:
        BGEM3FlagModel: Loaded model ready for inference
    """
    if device is None:
        device = detect_device()
    
    if use_fp16 is None:
        use_fp16 = (device != 'cpu')
    
    model = BGEM3FlagModel(
        MODEL_NAME,
        use_fp16=use_fp16
    )
    
    return model


def compute_embedding(
    model: BGEM3FlagModel,
    text: str,
    max_length: int = MAX_LENGTH
) -> List[float]:
    """
    Compute embedding for a single text.
    
    This is the canonical implementation. All services MUST use this function
    to ensure embedding consistency across the system.
    
    Args:
        model: Pre-loaded BGE-M3 model
        text: Input text to embed
        max_length: Maximum token length (default: 8192)
    
    Returns:
        List[float]: Embedding vector of dimension 1024
    """
    result = model.encode(
        [text],
        batch_size=DEFAULT_BATCH_SIZE,
        max_length=max_length
    )
    
    # Extract dense vector embedding
    embedding = result['dense_vecs'][0].tolist()
    
    # Validate dimension
    if len(embedding) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Unexpected embedding dimension: {len(embedding)}, expected {EMBEDDING_DIMENSION}"
        )
    
    return embedding


def compute_embeddings_batch(
    model: BGEM3FlagModel,
    texts: List[str],
    batch_size: int = 32,
    max_length: int = MAX_LENGTH
) -> List[List[float]]:
    """
    Compute embeddings for multiple texts in batch.
    
    Args:
        model: Pre-loaded BGE-M3 model
        texts: List of input texts to embed
        batch_size: Batch size for processing (default: 32)
        max_length: Maximum token length (default: 8192)
    
    Returns:
        List[List[float]]: List of embedding vectors, each of dimension 1024
    """
    result = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length
    )
    
    # Extract dense vector embeddings
    embeddings = [vec.tolist() for vec in result['dense_vecs']]
    
    # Validate dimensions
    for i, embedding in enumerate(embeddings):
        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Unexpected embedding dimension at index {i}: {len(embedding)}, expected {EMBEDDING_DIMENSION}"
            )
    
    return embeddings
