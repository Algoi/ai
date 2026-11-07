import faiss

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.memory import ChatMemoryBuffer

from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike


# ============================================================
# 1. 数据加载（Data Connectors）
#    从网页读取内容，转换为 LlamaIndex Document。
# ============================================================

local_doc = SimpleDirectoryReader(
    PROJECT_ROOT + "/datas",
    recursive=True,
    file_extractor={
        ".pdf": PyMuPDFReader()
    },
).load_data()

web_doc = BeautifulSoupWebReader().load_data(
    urls=["https://cloud.tencent.com/developer/article/2499999?fromSource=gwzcw.9358214.9358214.9358214&utm_medium=cpc&utm_id=gwzcw.9358214.9358214.9358214"]
)

documents = local_doc + web_doc


# ============================================================
# 2. 文档切分（Chunking）
#    将 Document 切分为多个 TextNode，作为后续索引和检索的基本单位。
#    Chunk 是文本内容，Node 是包含文本、元数据及节点关系的对象。
# ============================================================

splitter = SentenceSplitter(
    chunk_size=512,               # 每个 Chunk 的最大 Token 数
    chunk_overlap=50,             # 相邻 Chunk 的重叠 Token 数
    paragraph_separator="\n\n",   # 段落分隔符
    secondary_chunking_regex="<table>(.+?)</table>", # 二次切分正则表达式，用于在段落内进一步切分内容，比如识别表格标签。
    tokenizer=None,
)

nodes = splitter.get_nodes_from_documents(documents)
print(f"总共切分为 {len(nodes)} 个 Node")

# 查看第一个 Node 的完整结构
# print(nodes[0].model_dump_json(indent=2))


# ============================================================
# 3. 模型配置（Models）
#    LLM 负责生成回答，Embedding 模型负责将文本转换为向量。
# ============================================================

Settings.llm = OpenAILike(
    model=LLM_MODEL_NAME,
    api_base=LLM_DEEPSEEK_BASE_URL,
    api_key=LLM_DEEPSEEK_API_KEY,
    is_chat_model=True,
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL_NAME,
)


# ============================================================
# 4. 向量索引构建（Indexing）
#    Node → Embedding → FAISS，并将索引持久化到本地磁盘。
# ============================================================

# 创建 FAISS 索引，512 是 BGE-small-zh-v1.5 的向量维度
faiss_index = faiss.IndexFlatL2(512)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# 创建存储上下文，统一管理向量存储和 Node 等数据
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 将 Node 向量化并写入 FAISS
vector_index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
)

# 持久化 FAISS 索引及 LlamaIndex 相关数据
vector_index.storage_context.persist(
    persist_dir=PROJECT_ROOT + "/.cache/faiss_index"
)


# ============================================================
# 5. 检索与问答（Retrieval & Query Engine）
#    Query → Embedding → FAISS Top-K 检索 → LLM 生成回答。
# ============================================================

# 检索与问题最相关的 3 个 Node
vector_retriever = VectorIndexRetriever(
    index=vector_index,
    similarity_top_k=3,
)

# 查看实际检索到的 Node，便于分析检索质量
# retrieved_nodes = vector_retriever.retrieve("福利一是什么？")
# print(f"检索到 {len(retrieved_nodes)} 个相关 Node：")
# for i, node in enumerate(retrieved_nodes):
#     print(f"\n节点 {i+1}:")
#     print(f"相似度得分: {node.score}")
#     print(f"内容: {node.node.text}")
#     print("-" * 50)

# 创建查询引擎
query_engine = RetrieverQueryEngine(retriever=vector_retriever)

# 执行单轮 RAG 问答
response = query_engine.query("福利一是什么？")
print("回答：", response)


# ============================================================
# 6. 多轮对话（Chat Engine）实现记忆聊天
#    在 RAG 检索基础上加入 Memory，支持结合历史对话回答问题。
# ============================================================

# 保存对话历史，token_limit 控制可使用的历史上下文长度
memory = ChatMemoryBuffer.from_defaults(token_limit=2048)

chat_engine = vector_index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=(
        "你是一个专业的问答助手，基于检索到的内容回答用户的问题。"
        "如果检索不到相关内容，根据你自身的能力回答，不要不回答。"
    ),
)

response = chat_engine.chat("詹姆斯是谁？")
print("回答：", response)

response = chat_engine.chat("福利一是什么？")
print("回答：", response)


# ============================================================
# 7. 流式输出（Streaming）
#    逐步输出 LLM 生成的内容，而不是等待完整回答。
# ============================================================

response = chat_engine.stream_chat("福利二是什么？")

for chunk in response.response_gen:
    print(chunk, end="", flush=True)