import os
from pathlib import Path

# 必须在导入 Hugging Face 相关库前设置；可通过环境变量改回官网或其他镜像。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 使用独立的 NLTK 数据目录，避免依赖安装目录的共享写权限导致资源被拒读。
os.environ.setdefault("NLTK_DATA", str(Path(__file__).resolve().parents[2] / ".cache" / "nltk_data"))
LLM_DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
LLM_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
LLM_MODEL_NAME = "deepseek-flash"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 恢复 LlamaIndex 默认的 NLTK 分句方式，再按长度组合成文本块。
Settings.node_parser = SentenceSplitter()

# 配置 DeepSeek API
Settings.llm = OpenAILike(
    model=LLM_MODEL_NAME,
    api_base=LLM_DEEPSEEK_BASE_URL,
    api_key=LLM_DEEPSEEK_API_KEY,
    is_chat_model=True,
    is_function_calling_model=False,
    context_window=65536,
)

# 配置 BGE 向量模型，下载和缓存由库自动管理。
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)

# 第一步：加载 data 目录中的 PDF
documents = SimpleDirectoryReader(
    "/home/huyuanhai/project/datas/data",
    required_exts=[".pdf"]
).load_data()

# 第二步：构建索引
index = VectorStoreIndex.from_documents(documents)

# 第三步：创建查询引擎
query_engine = index.as_query_engine()

# 第四步：提问
response = query_engine.query("世运电路2023年同比增长")
print(response)
