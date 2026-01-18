"""
统一的 LLM 后端接口

支持多种后端:
1. OpenAI API (GPT-3.5/GPT-4)
2. Ollama (本地模型如 Llama, Qwen, Mistral)
3. HuggingFace Transformers (本地运行)
4. vLLM (高性能推理服务)

使用示例:
    # OpenAI
    llm = get_llm_backend("openai", model="gpt-3.5-turbo")

    # Ollama (本地)
    llm = get_llm_backend("ollama", model="qwen2.5:7b")

    # HuggingFace (本地)
    llm = get_llm_backend("huggingface", model="Qwen/Qwen2.5-1.5B-Instruct")

    # 统一调用
    response = llm.chat([{"role": "user", "content": "Hello!"}])
"""

import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass

@dataclass
class LLMConfig:
    """LLM 配置"""
    model: str
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class BaseLLMBackend(ABC):
    """LLM 后端基类"""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        聊天接口

        Args:
            messages: [{"role": "user/assistant/system", "content": "..."}]

        Returns:
            str: 模型回复
        """
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        文本生成接口

        Args:
            prompt: 输入提示

        Returns:
            str: 生成的文本
        """
        pass

    def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """流式聊天 (可选实现)"""
        yield self.chat(messages, **kwargs)


class OpenAIBackend(BaseLLMBackend):
    """OpenAI API 后端"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量或传入 api_key")

        self.client = OpenAI(
            api_key=api_key,
            base_url=config.base_url
        )

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            top_p=kwargs.get("top_p", self.config.top_p),
        )
        return response.choices[0].message.content

    def generate(self, prompt: str, **kwargs) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        stream = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OllamaBackend(BaseLLMBackend):
    """Ollama 本地模型后端"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import requests
        except ImportError:
            raise ImportError("请安装 requests: pip install requests")

        self.base_url = config.base_url or "http://localhost:11434"
        self.requests = requests

        # 检查 Ollama 是否可用
        try:
            resp = self.requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                print(f"⚠️ Ollama 服务可能未启动，请运行: ollama serve")
        except:
            print(f"⚠️ 无法连接 Ollama ({self.base_url})，请确保 Ollama 已启动")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = self.requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.config.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                    "top_p": kwargs.get("top_p", self.config.top_p),
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]

    def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        response = self.requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.config.model,
                "messages": messages,
                "stream": True,
            },
            stream=True,
            timeout=120
        )
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    yield data["message"]["content"]


class HuggingFaceBackend(BaseLLMBackend):
    """HuggingFace Transformers 本地后端"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("请安装: pip install torch transformers")

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"正在加载模型 {config.model} 到 {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            trust_remote_code=True
        )

        # 自动选择精度
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        # 设置 pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"✓ 模型加载完成!")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # 使用 chat template (如果模型支持)
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback: 手动构建
            prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            prompt += "Assistant: "

        return self.generate(prompt, **kwargs)

    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=kwargs.get("top_p", self.config.top_p),
                do_sample=kwargs.get("temperature", self.config.temperature) > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # 只返回新生成的部分
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


class VLLMBackend(BaseLLMBackend):
    """vLLM 高性能推理后端 (OpenAI 兼容接口)"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

        # vLLM 默认运行在 8000 端口
        base_url = config.base_url or "http://localhost:8000/v1"

        self.client = OpenAI(
            api_key="EMPTY",  # vLLM 不需要真实 key
            base_url=base_url
        )

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        return response.choices[0].message.content

    def generate(self, prompt: str, **kwargs) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)


# ==================== 便捷工厂函数 ====================

def get_llm_backend(
    backend: str = "ollama",
    model: str = None,
    **kwargs
) -> BaseLLMBackend:
    """
    获取 LLM 后端实例

    Args:
        backend: 后端类型 ("openai", "ollama", "huggingface", "vllm")
        model: 模型名称
        **kwargs: 其他配置参数

    Returns:
        BaseLLMBackend: LLM 后端实例

    示例:
        # OpenAI
        llm = get_llm_backend("openai", model="gpt-3.5-turbo")

        # Ollama (推荐用于本地)
        llm = get_llm_backend("ollama", model="qwen2.5:7b")

        # HuggingFace
        llm = get_llm_backend("huggingface", model="Qwen/Qwen2.5-1.5B-Instruct")
    """
    # 默认模型
    default_models = {
        "openai": "gpt-3.5-turbo",
        "ollama": "qwen2.5:7b",
        "huggingface": "Qwen/Qwen2.5-1.5B-Instruct",
        "vllm": "Qwen/Qwen2.5-7B-Instruct",
    }

    model = model or default_models.get(backend, "gpt-3.5-turbo")
    config = LLMConfig(model=model, **kwargs)

    backends = {
        "openai": OpenAIBackend,
        "ollama": OllamaBackend,
        "huggingface": HuggingFaceBackend,
        "hf": HuggingFaceBackend,
        "vllm": VLLMBackend,
    }

    if backend not in backends:
        raise ValueError(f"不支持的后端: {backend}. 可选: {list(backends.keys())}")

    return backends[backend](config)


def auto_detect_backend() -> BaseLLMBackend:
    """
    自动检测可用的后端

    优先级: Ollama > OpenAI > HuggingFace
    """
    import requests

    # 1. 尝试 Ollama
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            if models:
                model_name = models[0]["name"]
                print(f"✓ 检测到 Ollama，使用模型: {model_name}")
                return get_llm_backend("ollama", model=model_name)
    except:
        pass

    # 2. 尝试 OpenAI
    if os.getenv("OPENAI_API_KEY"):
        print("✓ 检测到 OPENAI_API_KEY，使用 OpenAI")
        return get_llm_backend("openai")

    # 3. 使用 HuggingFace (总是可用，但需要下载模型)
    print("⚠️ 使用 HuggingFace 本地模型 (首次运行需要下载)")
    return get_llm_backend("huggingface", model="Qwen/Qwen2.5-0.5B-Instruct")


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("LLM Backend 测试")
    print("=" * 60)

    # 测试 Ollama (如果可用)
    try:
        llm = get_llm_backend("ollama", model="qwen2.5:7b")
        response = llm.chat([{"role": "user", "content": "Say 'Hello' in one word."}])
        print(f"\nOllama 测试: {response}")
    except Exception as e:
        print(f"\nOllama 不可用: {e}")

    print("\n✓ LLM Backend 模块加载成功!")
