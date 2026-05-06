"""
统一的 LLM 后端接口

支持多种后端:
1. OpenAI API (GPT-3.5/GPT-4)
2. Ollama (本地模型如 Llama, Qwen, Mistral)
3. DashScope / 通义千问 API (阿里云，OpenAI 兼容)
4. HuggingFace Transformers (本地运行)
5. vLLM (高性能推理服务)
6. Mock (课堂离线/额度不足时的确定性教学后端)

使用示例:
    # Ollama (本地)
    llm = get_llm_backend("ollama", model="qwen2.5:7b")

    # DashScope / 通义千问 (云端 API，推荐)
    llm = get_llm_backend("dashscope", model=os.getenv("LLM_MODEL", "qwen-plus-2025-01-25"))

    # OpenAI
    llm = get_llm_backend("openai", model="gpt-3.5-turbo")

    # HuggingFace (本地)
    llm = get_llm_backend("huggingface", model="Qwen/Qwen2.5-1.5B-Instruct")

    # 统一调用
    response = llm.chat([{"role": "user", "content": "Hello!"}])
"""

import os
import json
import time
import random
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator, Callable
from dataclasses import dataclass


_PROJECT_ENV_LOADED = False


def load_project_env() -> None:
    """Load simple KEY=VALUE pairs from the project .env without overriding env vars."""
    global _PROJECT_ENV_LOADED
    if _PROJECT_ENV_LOADED:
        return

    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]
    seen: set[Path] = set()
    for env_path in candidates:
        env_path = env_path.resolve()
        if env_path in seen or not env_path.exists():
            continue
        seen.add(env_path)
        for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

    _PROJECT_ENV_LOADED = True


load_project_env()


def _retry_on_429(func: Callable, *, max_retries: int = 6, base_delay: float = 8.0):
    """Call func() and retry on RateLimitError / 429 with exponential backoff (jittered).

    DashScope free-tier limit is per-minute, so we wait ~10s, 20s, 40s, 60s capped, 60s, 60s.
    """
    try:
        from openai import RateLimitError
    except ImportError:
        RateLimitError = None  # type: ignore

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            is_rate = (RateLimitError is not None and isinstance(exc, RateLimitError)) or "429" in str(exc) or "rate" in str(exc).lower()
            if not is_rate or attempt == max_retries:
                raise
            delay = min(60.0, base_delay * (2 ** attempt)) + random.uniform(0, 2.0)
            time.sleep(delay)
    return None  # unreachable

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
        def _call():
            return self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                top_p=kwargs.get("top_p", self.config.top_p),
            )
        response = _retry_on_429(_call)
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


class DashScopeBackend(OpenAIBackend):
    """
    DashScope / 通义千问 API 后端（阿里云）

    OpenAI 兼容接口，无需本地 GPU。
    申请 API Key: https://dashscope.console.aliyun.com/

    模型列表:
      - qwen-plus      : 性价比最高（推荐）
      - qwen-turbo      : 最快、最便宜
      - qwen-max        : 最强
      - qwen-long       : 长上下文
    """

    def __init__(self, config: LLMConfig):
        config.base_url = config.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        config.api_key = config.api_key or os.getenv("DASHSCOPE_API_KEY")
        if not config.api_key:
            raise ValueError(
                "请设置 DASHSCOPE_API_KEY 环境变量，或传入 api_key 参数。\n"
                "申请地址: https://dashscope.console.aliyun.com/"
            )
        super().__init__(config)


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
                print(f"WARN Ollama 服务可能未启动，请运行: ollama serve")
        except:
            print(f"WARN 无法连接 Ollama ({self.base_url})，请确保 Ollama 已启动")

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

        print(f"OK 模型加载完成!")

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


class MockLLMBackend(BaseLLMBackend):
    """Deterministic backend for classroom demos when live LLM APIs are unavailable."""

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        system = messages[0].get("content", "") if messages else ""
        prompt = "\n".join(str(message.get("content", "")) for message in messages)
        return self._respond(system, prompt)

    def generate(self, prompt: str, **kwargs) -> str:
        return self._respond("", prompt)

    def _respond(self, system: str, prompt: str) -> str:
        text = f"{system}\n{prompt}".lower()
        system_text = system.lower()

        if "compile a final summary" in text:
            return (
                "SUMMARY:\n"
                "- Built a factorial implementation with input validation.\n"
                "- Reviewed the code and approved it.\n"
                "- Generated and executed tests for normal, edge, and error cases.\n\n"
                "FINAL CODE:\n"
                "```python\n"
                f"{self._factorial_function_code()}\n"
                "```\n"
            )

        if "planner agent" in system_text:
            return (
                "ANALYSIS: Implement a small deterministic Python function and verify edge cases.\n\n"
                "SUBTASKS:\n"
                "1. Define factorial(n) for n >= 0.\n"
                "2. Raise ValueError for negative input.\n"
                "3. Add a short example printing factorial(5).\n"
                "4. Review the implementation for correctness and maintainability.\n"
                "5. Test base, normal, and error cases.\n\n"
                "DEPENDENCIES: Coder -> Reviewer -> Tester.\n\n"
                "IMPLEMENTATION_ORDER: Write code, review it, then run tests."
            )

        if "coder agent" in system_text:
            return "```python\n" + self._factorial_function_code() + "\n```"

        if "reviewer agent" in system_text:
            return (
                "CORRECTNESS: PASS - factorial handles 0, positive integers, and negative input.\n"
                "ERROR_HANDLING: PASS - negative values raise ValueError.\n"
                "CODE_QUALITY: PASS - iterative code is readable and avoids recursion depth issues.\n"
                "SECURITY: PASS - no file, network, or shell access.\n\n"
                "ISSUES:\n"
                "- None.\n\n"
                "SUGGESTIONS:\n"
                "- Add tests for 0, 5, and negative input.\n\n"
                "VERDICT: APPROVED"
            )

        if "tester agent" in system_text:
            return (
                "TEST_CASES:\n"
                "1. factorial(0) == 1\n"
                "2. factorial(5) == 120\n"
                "3. factorial(-1) raises ValueError\n\n"
                "TEST_CODE:\n"
                "```python\n"
                "assert factorial(0) == 1\n"
                "assert factorial(5) == 120\n"
                "try:\n"
                "    factorial(-1)\n"
                "    raise AssertionError('expected ValueError')\n"
                "except ValueError:\n"
                "    pass\n"
                "print('All tests passed')\n"
                "```\n\n"
                "RESULTS:\n"
                "- Test 1: PASS\n"
                "- Test 2: PASS\n"
                "- Test 3: PASS\n\n"
                "SUMMARY: 3/3 tests passed"
            )

        if "debate moderator" in system_text or "moderator agent" in system_text:
            return (
                "ANALYSIS: Both sides made practical engineering arguments. "
                "For an AI-heavy startup, Python wins on ecosystem depth and iteration speed; "
                "Go wins on simple deployment and concurrency.\n\n"
                "WINNER: Python (Django/FastAPI)\n\n"
                "REASON: The stated startup context benefits more from AI/ML library access, "
                "rapid prototyping, and hiring availability."
            )

        if "advocate" in system_text:
            if "python" in system_text:
                return (
                    "Python is the stronger startup default when the product depends on AI, data, "
                    "or rapid backend iteration. FastAPI gives clean APIs, Django gives batteries-included "
                    "product scaffolding, and the ML ecosystem avoids cross-language glue early on."
                )
            return (
                "Go is a strong startup choice when deployment simplicity, predictable concurrency, "
                "and low operational overhead matter most. It produces small binaries and keeps runtime "
                "behavior easy to reason about."
            )

        if "python code generator" in system_text or "return only executable python code" in text:
            return "```python\n" + self._code_for_prompt(text) + "\n```"

        if "够不够回答原始问题" in prompt:
            if "医疗保险" in prompt:
                return "INSUFFICIENT\n公司 医疗保险 员工福利 政策"
            return "SUFFICIENT\n根据当前检索结果，可以回答问题；答案应严格基于知识库内容。"

        if "改写一个更具体的 query" in prompt:
            return "StarLink 套餐 价格 API 限流 员工福利 报销"

        if "基于下面信息答" in prompt or "基于:" in prompt:
            if "医疗保险" in prompt:
                return "知识库没有检索到公司医疗保险政策的明确条款，建议标记为未收录并转人工确认。"
            return "根据检索到的资料，可以给出一个基于知识库的简明回答；未覆盖的信息不应编造。"

        if "observation:" in text:
            if "15 * 7" in text:
                return "Thought: I have the calculation result.\nFinal Answer: 15 days from Monday is Tuesday, and 15 * 7 = 105."
            if "sqrt(144)" in text:
                return "Thought: I have the calculation result.\nFinal Answer: sqrt(144) + 25 = 37."
            if "current date and time" in text:
                return "Thought: I have the current datetime observation.\nFinal Answer: Use the weekday shown in the observation as today's weekday."
            if "weather" in text:
                return "Thought: I have the weather observation.\nFinal Answer: The requested weather information is available in the observation."
            if "python" in text:
                return "Thought: I have enough search context.\nFinal Answer: Python is a readable, general-purpose programming language used for web development, automation, data science, and AI."
            return "Thought: I have enough information from the observation.\nFinal Answer: Based on the observation, here is the answer."

        if "15 * 7" in text:
            return 'Thought: I need the arithmetic result.\nAction: calculator\nAction Input: {"expression": "15 * 7"}'

        if "square root" in text or "sqrt" in text:
            return 'Thought: I need a calculator for this expression.\nAction: calculator\nAction Input: {"expression": "sqrt(144) + 25"}'

        if "weather" in text or "天气" in prompt:
            return 'Thought: I need current weather data.\nAction: get_weather\nAction Input: {"city": "Tokyo"}'

        if "day of the week" in text or "date" in text or "time" in text or "日期" in prompt or "时间" in prompt:
            return "Thought: I need the current date and time.\nAction: get_datetime\nAction Input: {}"

        if "python programming language" in text or "web search" in text or "搜索" in prompt:
            return 'Thought: I should search for a concise definition.\nAction: web_search\nAction Input: {"query": "Python programming language"}'

        return (
            "Thought: This is a deterministic teaching mock response.\n"
            "Final Answer: Mock LLM backend is active; use a real backend for open-ended generation."
        )

    def _factorial_function_code(self) -> str:
        return '''def factorial(n: int) -> int:
    """Return n! for n >= 0."""
    if n < 0:
        raise ValueError("n must be non-negative")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


print(factorial(5))'''

    def _code_for_prompt(self, text: str) -> str:
        if "factorial of 10" in text:
            return """result = 1
for i in range(2, 11):
    result *= i
print(result)"""

        if "fibonacci" in text:
            return """numbers = [0, 1]
while len(numbers) < 20:
    numbers.append(numbers[-1] + numbers[-2])
print(numbers[:20])"""

        if "prime" in text and "50" in text:
            return """def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

print([n for n in range(1, 51) if is_prime(n)])"""

        if "standard deviation" in text or "statistics" in text or "median" in text:
            return """import statistics

numbers = [23, 45, 67, 89, 12, 34, 56, 78, 90, 11]
print("Mean:", statistics.mean(numbers))
print("Median:", statistics.median(numbers))
print("Standard deviation:", statistics.stdev(numbers))
print("Min:", min(numbers))
print("Max:", max(numbers))"""

        if "palindrome" in text:
            return """import re

def is_palindrome(s):
    cleaned = re.sub(r"[^a-z0-9]", "", s.lower())
    return cleaned == cleaned[::-1]

for value in ["racecar", "hello", "A man a plan a canal Panama"]:
    print(f"{value} -> {is_palindrome(value)}")"""

        if "bubble sort" in text:
            return """arr = [64, 34, 25, 12, 22, 11, 90]
for i in range(len(arr)):
    for j in range(0, len(arr) - i - 1):
        if arr[j] > arr[j + 1]:
            arr[j], arr[j + 1] = arr[j + 1], arr[j]
print(arr)"""

        if "print('test')" in text or 'print("test")' in text:
            return "print('test')"

        return "print('Mock code generation response')"


# ==================== 便捷工厂函数 ====================

def get_llm_backend(
    backend: str = "dashscope",
    model: str = None,
    **kwargs
) -> BaseLLMBackend:
    """
    获取 LLM 后端实例

    Args:
        backend: 后端类型 ("ollama", "dashscope", "openai", "huggingface", "vllm", "mock")
        model: 模型名称
        **kwargs: 其他配置参数

    Returns:
        BaseLLMBackend: LLM 后端实例

    示例:
        # Ollama (本地免费)
        llm = get_llm_backend("ollama", model="qwen2.5:7b")

        # DashScope / 通义千问 (云端 API，推荐)
        llm = get_llm_backend("dashscope", model=os.getenv("LLM_MODEL", "qwen-plus-2025-01-25"))

        # OpenAI
        llm = get_llm_backend("openai", model="gpt-3.5-turbo")
    """
    # 默认模型
    default_models = {
        "openai": "gpt-3.5-turbo",
        "ollama": "qwen2.5:7b",
        "dashscope": os.getenv("LLM_MODEL", "qwen-plus-2025-01-25"),
        "huggingface": "Qwen/Qwen2.5-1.5B-Instruct",
        "vllm": "Qwen/Qwen2.5-7B-Instruct",
        "mock": "deterministic-teaching-mock",
    }

    model = model or default_models.get(backend, "gpt-3.5-turbo")
    config = LLMConfig(model=model, **kwargs)

    backends = {
        "openai": OpenAIBackend,
        "ollama": OllamaBackend,
        "dashscope": DashScopeBackend,
        "huggingface": HuggingFaceBackend,
        "hf": HuggingFaceBackend,
        "vllm": VLLMBackend,
        "mock": MockLLMBackend,
    }

    if backend not in backends:
        raise ValueError(f"不支持的后端: {backend}. 可选: {list(backends.keys())}")

    return backends[backend](config)


def auto_detect_backend() -> BaseLLMBackend:
    """
    自动检测可用的后端

    优先级: DashScope > Ollama > OpenAI > HuggingFace
    """
    import requests

    # 1. 尝试 DashScope / 通义千问（推荐，阿里百炼）
    if os.getenv("DASHSCOPE_API_KEY"):
        model_name = os.getenv("LLM_MODEL", "qwen-plus-2025-01-25")
        print(f"OK 检测到 DASHSCOPE_API_KEY，使用通义千问 ({model_name})")
        return get_llm_backend("dashscope", model=model_name)

    # 2. 尝试 Ollama（本地）
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            if models:
                model_name = models[0]["name"]
                print(f"OK 检测到 Ollama，使用模型: {model_name}")
                return get_llm_backend("ollama", model=model_name)
    except:
        pass

    # 3. 尝试 OpenAI
    if os.getenv("OPENAI_API_KEY"):
        print("OK 检测到 OPENAI_API_KEY，使用 OpenAI")
        return get_llm_backend("openai")

    # 4. 使用 HuggingFace (总是可用，但需要下载模型)
    print("WARN 未检测到 API Key，使用 HuggingFace 本地模型 (首次运行需要下载)")
    print("提示: 推荐设置 DASHSCOPE_API_KEY 使用阿里百炼: https://dashscope.console.aliyun.com/")
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

    # 测试 DashScope (如果配置了 API Key)
    if os.getenv("DASHSCOPE_API_KEY"):
        try:
            llm = get_llm_backend(
                "dashscope",
                model=os.getenv("LLM_MODEL", "qwen-plus-2025-01-25"),
            )
            response = llm.chat([{"role": "user", "content": "Say 'Hello' in one word."}])
            print(f"\nDashScope 测试: {response}")
        except Exception as e:
            print(f"\nDashScope 不可用: {e}")

    print("\nOK LLM Backend 模块加载成功!")
