# RAG Demo -- 分片策略 & 检索模式 对比实验

基于 LangChain 搭建的本地 RAG（Retrieval-Augmented Generation）Demo，研究不同文本分片策略和检索模式对回答质量的影响。

## 核心模块

| 模块 | 说明 |
|------|------|
| `document_loader.py` | 文档加载器，读取 `documents/` 目录下的 `.txt` 文件 |
| `chunking.py` | 四种分片策略实现 |
| `vector_store.py` | ChromaDB 向量数据库管理 |
| `retriever.py` | 三种检索模式：向量检索 / BM25 关键词 / 混合检索 (RRF) |
| `reranker.py` | 重排序模块：Cross-Encoder 本地模型 / LLM 重排序 |
| `rag_chain.py` | LangChain RAG 链构建 |
| `main.py` | 主入口，运行对比实验 |
| `config.py` | 全局配置 |

## 四种分片策略

| 策略 | 方法 | 优点 | 缺点 |
|------|------|------|------|
| **Character** | 按固定字符数切分 | 简单、确定性强 | 可能在句子中间截断，破坏语义 |
| **Recursive** | 递归按段落>换行>句子>单词切分 | 在多个层次寻找自然分割点 | 需要调参 chunk_size |
| **Semantic** | 基于 embedding 相似度分组相邻句子 | 语义连贯性最好 | 计算成本高、片段大小不可控 |
| **Paragraph** | 按空行（自然段落）切分 | 保留作者原始结构 | 片段大小差异大 |

## 三种检索模式

| 模式 | 方法 | 特点 |
|------|------|------|
| **Vector** | 纯向量相似度检索 | 语义匹配能力强，但可能遗漏关键词精确匹配 |
| **BM25** | 纯关键词检索 (BM25Okapi) | 精确关键词匹配，但缺乏语义理解 |
| **Hybrid** | BM25 + Vector + RRF 融合 | 兼顾语义和关键词，综合效果最好 |

## Re-ranking 重排序

检索后使用 Cross-Encoder 模型 (`ms-marco-MiniLM-L-6-v2`) 对候选文档重新打分排序，过滤掉低相关性结果，显著提升最终检索精度。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 LLM API
cp .env.example .env
# 编辑 .env 填入 API Key

# 3. 默认运行（recursive 分片 + hybrid 检索 + rerank）
python main.py

# 4. 对比三种检索模式
python main.py --compare-retrieval

# 5. 对比四种分片策略
python main.py --compare-chunking

# 6. 禁用重排序
python main.py --no-rerank

# 7. 交互式问答
python main.py --interactive

# 8. 指定单个策略和检索模式
python main.py --strategy semantic --retrieval bm25

# 9. 重置向量数据库
python main.py --reset
```

## 项目结构

```
rag/
├── documents/                  # 示例文档
├── chroma_db/                  # ChromaDB 持久化目录（自动生成）
├── config.py                   # 配置
├── document_loader.py          # 文档加载
├── chunking.py                 # 分片策略
├── vector_store.py             # 向量数据库
├── retriever.py                # 检索模式（vector / bm25 / hybrid）
├── reranker.py                 # 重排序（Cross-Encoder / LLM）
├── rag_chain.py                # RAG 链
├── main.py                     # 主入口
├── requirements.txt            # 依赖
└── .env.example                # 环境变量模板
```

## 技术栈

- **LangChain** -- RAG 编排框架
- **ChromaDB** -- 本地向量数据库
- **Sentence-Transformers** (`all-MiniLM-L6-v2`) -- 本地 Embedding 模型
- **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`) -- 本地 Reranker 模型
- **BM25Okapi** -- 关键词检索算法
- **通义千问** (qwen-turbo) -- LLM 生成回答
- **Rich** -- 终端美化输出
