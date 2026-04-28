"""
LLM-Agent-Core 工具模块

提供统一的 LLM 和 Embedding 后端接口
"""

from .llm_backend import (
    get_llm_backend,
    auto_detect_backend,
    BaseLLMBackend,
    LLMConfig,
    OpenAIBackend,
    OllamaBackend,
    HuggingFaceBackend,
    VLLMBackend,
)

from .embedding_backend import (
    get_embedding_backend,
    BaseEmbeddingBackend,
    EmbeddingConfig,
    SentenceTransformersBackend,
    OpenAIEmbeddingBackend,
    SimpleVectorStore,
    TFIDFEmbeddingBackend,
)

__all__ = [
    # LLM
    "get_llm_backend",
    "auto_detect_backend",
    "BaseLLMBackend",
    "LLMConfig",
    "OpenAIBackend",
    "OllamaBackend",
    "HuggingFaceBackend",
    "VLLMBackend",
    # Embedding
    "get_embedding_backend",
    "BaseEmbeddingBackend",
    "EmbeddingConfig",
    "SentenceTransformersBackend",
    "OpenAIEmbeddingBackend",
    "SimpleVectorStore",
    "TFIDFEmbeddingBackend",
]
