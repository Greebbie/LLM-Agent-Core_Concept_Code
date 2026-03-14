"""
统一配置文件 — 所有 notebook 共用

使用方式（在任意 notebook 中）：
    from config import setup
    env = setup()           # 加载 API Key + 返回配置
    llm = env.get_llm()     # 获取 LLM 后端
    embedder = env.get_embedder()  # 获取 Embedding 后端
"""

import os
import sys

# ============================================================
# 默认配置（可在 Day0 notebook 中修改并保存）
# ============================================================
DEFAULTS = {
    # LLM 后端
    "llm_backend": "dashscope",
    "llm_model": "qwen-plus",

    # Embedding 后端
    "embedding_backend": "dashscope",
    "embedding_model": "text-embedding-v3",
}

# 配置文件路径（与 notebook 同级目录）
_CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
# fallback: 当前工作目录
if not os.path.exists(os.path.dirname(_CONFIG_FILE)):
    _CONFIG_FILE = os.path.join(os.getcwd(), ".env")


def _load_env_file(path=None):
    """从 .env 文件加载环境变量"""
    path = path or _CONFIG_FILE
    # 也尝试当前目录
    candidates = [path, os.path.join(os.getcwd(), ".env"), ".env"]
    for p in candidates:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and value:
                            os.environ.setdefault(key, value)
            return p
    return None


def save_config(api_key=None, llm_backend=None, llm_model=None,
                embedding_backend=None, embedding_model=None, path=None):
    """
    保存配置到 .env 文件（由 Day0 调用）

    Args:
        api_key: DashScope API Key
        llm_backend: LLM 后端名称
        llm_model: LLM 模型名称
        embedding_backend: Embedding 后端名称
        embedding_model: Embedding 模型名称
        path: .env 文件路径
    """
    path = path or _CONFIG_FILE
    # 也尝试写到当前工作目录
    if not os.path.exists(os.path.dirname(path)):
        path = os.path.join(os.getcwd(), ".env")

    lines = []
    lines.append("# 课程配置文件（由 Day0 自动生成，请勿提交到 Git）")
    lines.append("")

    if api_key:
        lines.append(f'DASHSCOPE_API_KEY={api_key}')
        os.environ["DASHSCOPE_API_KEY"] = api_key

    lines.append("")
    lines.append("# 后端配置")
    lines.append(f'LLM_BACKEND={llm_backend or DEFAULTS["llm_backend"]}')
    lines.append(f'LLM_MODEL={llm_model or DEFAULTS["llm_model"]}')
    lines.append(f'EMBEDDING_BACKEND={embedding_backend or DEFAULTS["embedding_backend"]}')
    lines.append(f'EMBEDDING_MODEL={embedding_model or DEFAULTS["embedding_model"]}')

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"✅ 配置已保存到: {os.path.abspath(path)}")


class Environment:
    """课程环境配置，所有 notebook 统一使用"""

    def __init__(self):
        self.llm_backend = os.getenv("LLM_BACKEND", DEFAULTS["llm_backend"])
        self.llm_model = os.getenv("LLM_MODEL", DEFAULTS["llm_model"])
        self.embedding_backend = os.getenv("EMBEDDING_BACKEND", DEFAULTS["embedding_backend"])
        self.embedding_model = os.getenv("EMBEDDING_MODEL", DEFAULTS["embedding_model"])
        self._llm = None
        self._embedder = None

    @staticmethod
    def _import_sibling(module_name, attr_name):
        """从同目录下导入模块，避免 sys.path 中其他同名文件干扰"""
        import importlib.util
        module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{module_name}.py")
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, attr_name)

    def get_llm(self):
        """获取 LLM 后端实例（懒加载，只创建一次）"""
        if self._llm is None:
            get_llm_backend = self._import_sibling("llm_backend", "get_llm_backend")
            self._llm = get_llm_backend(self.llm_backend, model=self.llm_model)
            print(f"[LLM] {self.llm_backend} / {self.llm_model}")
        return self._llm

    def get_embedder(self):
        """获取 Embedding 后端实例（懒加载，只创建一次）"""
        if self._embedder is None:
            get_embedding_backend = self._import_sibling("embedding_backend", "get_embedding_backend")
            self._embedder = get_embedding_backend(self.embedding_backend, model=self.embedding_model)
            print(f"[Embedding] {self.embedding_backend} / {self.embedding_model} (dim={self._embedder.dimension})")
        return self._embedder

    def __repr__(self):
        key = os.getenv("DASHSCOPE_API_KEY", "")
        key_status = f"sk-...{key[-4:]}" if key.startswith("sk-") else "未配置"
        return (
            f"课程环境配置:\n"
            f"  API Key:   {key_status}\n"
            f"  LLM:       {self.llm_backend} / {self.llm_model}\n"
            f"  Embedding: {self.embedding_backend} / {self.embedding_model}"
        )


def setup():
    """
    一键加载配置，返回 Environment 对象

    在任意 notebook 开头调用：
        from config import setup
        env = setup()
    """
    # 确保 utils 在 sys.path 中
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)

    # 加载 .env
    env_path = _load_env_file()
    if env_path:
        print(f"[OK] 已加载配置: {env_path}")
    else:
        # 没有 .env，检查系统环境变量
        if os.getenv("DASHSCOPE_API_KEY"):
            print("[OK] 使用系统环境变量中的 DASHSCOPE_API_KEY")
        else:
            print("⚠️ 未找到 .env 配置文件，也没有 DASHSCOPE_API_KEY 环境变量")
            print("   请先运行 Day0_环境配置与测试.ipynb 完成配置")

    env = Environment()
    print(env)
    return env
