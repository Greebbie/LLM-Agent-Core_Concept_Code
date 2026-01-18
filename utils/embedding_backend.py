"""
统一的 Embedding 后端接口

支持多种后端:
1. SentenceTransformers (推荐，本地运行)
2. OpenAI Embeddings API
3. HuggingFace Transformers

使用示例:
    # SentenceTransformers (推荐)
    embedder = get_embedding_backend("sentence-transformers")
    vectors = embedder.embed(["Hello world", "How are you?"])

    # OpenAI
    embedder = get_embedding_backend("openai")
    vectors = embedder.embed(["Hello world"])
"""

import os
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Optional
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Embedding 配置"""
    model: str
    dimension: int = 384
    api_key: Optional[str] = None
    normalize: bool = True


class BaseEmbeddingBackend(ABC):
    """Embedding 后端基类"""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        文本向量化

        Args:
            texts: 单个文本或文本列表

        Returns:
            np.ndarray: 向量矩阵 [N, dim]
        """
        pass

    @property
    def dimension(self) -> int:
        """向量维度"""
        return self.config.dimension


class SentenceTransformersBackend(BaseEmbeddingBackend):
    """SentenceTransformers 后端 (推荐)"""

    # 推荐模型
    RECOMMENDED_MODELS = {
        "en": "all-MiniLM-L6-v2",           # 英文，快速，384维
        "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # 多语言
        "chinese": "shibing624/text2vec-base-chinese",  # 中文
        "large": "all-mpnet-base-v2",       # 大模型，768维
    }

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("请安装: pip install sentence-transformers")

        print(f"正在加载 Embedding 模型: {config.model}...")
        self.model = SentenceTransformer(config.model)
        self.config.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✓ 模型加载完成! 维度: {self.config.dimension}")

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=len(texts) > 100
        )
        return np.array(embeddings)


class OpenAIEmbeddingBackend(BaseEmbeddingBackend):
    """OpenAI Embeddings API 后端"""

    MODELS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装: pip install openai")

        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请设置 OPENAI_API_KEY")

        self.client = OpenAI(api_key=api_key)
        self.config.dimension = self.MODELS.get(config.model, 1536)

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        response = self.client.embeddings.create(
            model=self.config.model,
            input=texts
        )

        embeddings = [item.embedding for item in response.data]
        result = np.array(embeddings)

        if self.config.normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / np.maximum(norms, 1e-10)

        return result


class HuggingFaceEmbeddingBackend(BaseEmbeddingBackend):
    """HuggingFace Transformers 后端"""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("请安装: pip install torch transformers")

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"正在加载 Embedding 模型: {config.model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.model = AutoModel.from_pretrained(config.model).to(self.device)
        self.model.eval()

        # 获取维度
        with torch.no_grad():
            test = self.tokenizer("test", return_tensors="pt").to(self.device)
            output = self.model(**test)
            self.config.dimension = output.last_hidden_state.shape[-1]

        print(f"✓ 模型加载完成! 维度: {self.config.dimension}")

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling"""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return self.torch.sum(token_embeddings * input_mask_expanded, 1) / self.torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with self.torch.no_grad():
            output = self.model(**encoded)
            embeddings = self._mean_pooling(output, encoded["attention_mask"])

            if self.config.normalize:
                embeddings = self.torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()


class OllamaEmbeddingBackend(BaseEmbeddingBackend):
    """Ollama Embeddings 后端"""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        try:
            import requests
        except ImportError:
            raise ImportError("请安装: pip install requests")

        self.requests = requests
        self.base_url = "http://localhost:11434"

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            response = self.requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.config.model, "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])

        result = np.array(embeddings)

        if self.config.normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / np.maximum(norms, 1e-10)

        return result


# ==================== 便捷工厂函数 ====================

def get_embedding_backend(
    backend: str = "sentence-transformers",
    model: str = None,
    **kwargs
) -> BaseEmbeddingBackend:
    """
    获取 Embedding 后端实例

    Args:
        backend: 后端类型
        model: 模型名称
        **kwargs: 其他配置

    Returns:
        BaseEmbeddingBackend 实例

    示例:
        # SentenceTransformers (推荐)
        embedder = get_embedding_backend("sentence-transformers")

        # 中文模型
        embedder = get_embedding_backend(
            "sentence-transformers",
            model="shibing624/text2vec-base-chinese"
        )

        # OpenAI
        embedder = get_embedding_backend("openai", model="text-embedding-3-small")
    """
    default_models = {
        "sentence-transformers": "all-MiniLM-L6-v2",
        "st": "all-MiniLM-L6-v2",
        "openai": "text-embedding-3-small",
        "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
        "hf": "sentence-transformers/all-MiniLM-L6-v2",
        "ollama": "nomic-embed-text",
    }

    model = model or default_models.get(backend, "all-MiniLM-L6-v2")
    config = EmbeddingConfig(model=model, **kwargs)

    backends = {
        "sentence-transformers": SentenceTransformersBackend,
        "st": SentenceTransformersBackend,
        "openai": OpenAIEmbeddingBackend,
        "huggingface": HuggingFaceEmbeddingBackend,
        "hf": HuggingFaceEmbeddingBackend,
        "ollama": OllamaEmbeddingBackend,
    }

    if backend not in backends:
        raise ValueError(f"不支持的后端: {backend}. 可选: {list(backends.keys())}")

    return backends[backend](config)


# ==================== 简易向量数据库 ====================

class SimpleVectorStore:
    """
    简易向量数据库

    支持:
    - 添加文档
    - 相似度搜索
    - 持久化存储
    """

    def __init__(self, embedding_backend: BaseEmbeddingBackend = None):
        """
        初始化向量数据库

        Args:
            embedding_backend: Embedding 后端，如果为 None 则自动创建
        """
        if embedding_backend is None:
            try:
                embedding_backend = get_embedding_backend("sentence-transformers")
            except ImportError:
                print("⚠️ sentence-transformers 不可用，使用简单的 TF-IDF")
                embedding_backend = TFIDFEmbeddingBackend()

        self.embedder = embedding_backend
        self.documents: List[str] = []
        self.vectors: np.ndarray = np.array([])
        self.metadata: List[dict] = []

    def add_documents(
        self,
        documents: List[str],
        metadata: List[dict] = None,
        batch_size: int = 32
    ):
        """
        添加文档

        Args:
            documents: 文档列表
            metadata: 元数据列表
            batch_size: 批处理大小
        """
        if metadata is None:
            metadata = [{"id": i} for i in range(len(documents))]

        # 批量计算向量
        all_vectors = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vectors = self.embedder.embed(batch)
            all_vectors.append(vectors)

        new_vectors = np.vstack(all_vectors)

        # 更新存储
        self.documents.extend(documents)
        self.metadata.extend(metadata)

        if len(self.vectors) == 0:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])

        print(f"✓ 已添加 {len(documents)} 个文档，总计 {len(self.documents)} 个")

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[dict]:
        """
        相似度搜索

        Args:
            query: 查询文本
            top_k: 返回数量
            threshold: 相似度阈值

        Returns:
            List[dict]: 搜索结果，包含 document, score, metadata
        """
        if len(self.documents) == 0:
            return []

        # 计算查询向量
        query_vector = self.embedder.embed(query)[0]

        # 计算余弦相似度
        similarities = np.dot(self.vectors, query_vector)

        # 排序
        indices = np.argsort(similarities)[::-1]

        # 筛选结果
        results = []
        for idx in indices[:top_k]:
            score = float(similarities[idx])
            if score >= threshold:
                results.append({
                    "document": self.documents[idx],
                    "score": score,
                    "metadata": self.metadata[idx]
                })

        return results

    def save(self, path: str):
        """保存到文件"""
        import json
        np.savez(
            f"{path}.npz",
            vectors=self.vectors
        )
        with open(f"{path}.json", "w", encoding="utf-8") as f:
            json.dump({
                "documents": self.documents,
                "metadata": self.metadata
            }, f, ensure_ascii=False, indent=2)
        print(f"✓ 已保存到 {path}")

    def load(self, path: str):
        """从文件加载"""
        import json
        data = np.load(f"{path}.npz")
        self.vectors = data["vectors"]

        with open(f"{path}.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
            self.documents = meta["documents"]
            self.metadata = meta["metadata"]

        print(f"✓ 已加载 {len(self.documents)} 个文档")


class TFIDFEmbeddingBackend(BaseEmbeddingBackend):
    """TF-IDF Embedding 后端 (备选方案，无需额外依赖)"""

    def __init__(self, config: EmbeddingConfig = None):
        if config is None:
            config = EmbeddingConfig(model="tfidf", dimension=1000)
        super().__init__(config)

        self.vocabulary = {}
        self.idf = {}
        self.fitted = False

    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        import re
        text = text.lower()
        # 英文按空格，中文按字
        words = re.findall(r'[a-zA-Z]+|[\u4e00-\u9fff]', text)
        return words

    def fit(self, documents: List[str]):
        """学习词汇表"""
        doc_count = len(documents)
        word_doc_count = {}

        for doc in documents:
            words = set(self._tokenize(doc))
            for word in words:
                word_doc_count[word] = word_doc_count.get(word, 0) + 1

        # 取最常见的词构建词汇表
        sorted_words = sorted(word_doc_count.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_words[:self.config.dimension]

        self.vocabulary = {word: idx for idx, (word, _) in enumerate(top_words)}

        # 计算 IDF
        for word, count in word_doc_count.items():
            if word in self.vocabulary:
                self.idf[word] = np.log(doc_count / (count + 1)) + 1

        self.fitted = True
        self.config.dimension = len(self.vocabulary)

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        if not self.fitted:
            self.fit(texts)

        vectors = []
        for text in texts:
            vector = np.zeros(len(self.vocabulary))
            words = self._tokenize(text)
            word_count = {}

            for word in words:
                word_count[word] = word_count.get(word, 0) + 1

            for word, count in word_count.items():
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    tf = count / len(words) if words else 0
                    vector[idx] = tf * self.idf.get(word, 1)

            # 归一化
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

            vectors.append(vector)

        return np.array(vectors)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Embedding Backend 测试")
    print("=" * 60)

    # 测试 SentenceTransformers
    try:
        embedder = get_embedding_backend("sentence-transformers")
        vectors = embedder.embed(["Hello world", "How are you?"])
        print(f"\n✓ SentenceTransformers: 维度 {vectors.shape}")
    except Exception as e:
        print(f"\n⚠️ SentenceTransformers 不可用: {e}")
        print("使用 TF-IDF 备选方案...")
        embedder = TFIDFEmbeddingBackend()
        embedder.fit(["Hello world", "How are you?", "This is a test"])
        vectors = embedder.embed(["Hello world", "How are you?"])
        print(f"✓ TF-IDF: 维度 {vectors.shape}")

    # 测试向量数据库
    print("\n测试向量数据库...")
    store = SimpleVectorStore(embedder)
    store.add_documents([
        "Python is a programming language",
        "Machine learning uses data to learn patterns",
        "Transformers revolutionized NLP",
    ])

    results = store.search("What is ML?", top_k=2)
    print("\n搜索 'What is ML?' 结果:")
    for r in results:
        print(f"  - {r['document'][:50]}... (score: {r['score']:.3f})")

    print("\n✓ 所有测试通过!")
