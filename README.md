# RAG Demo — 分片策略对比实验

基于 LangChain 搭建的本地 RAG（Retrieval-Augmented Generation）Demo，重点研究不同文本分片策略对检索和回答质量的影响。

## 核心模块

| 模块 | 说明 |
|------|------|
| `document_loader.py` | 文档加载器，读取 `documents/` 目录下的 `.txt` 文件 |
| `chunking.py` | 四种分片策略实现 |
| `vector_store.py` | ChromaDB 向量数据库管理 |
| `rag_chain.py` | LangChain RAG 链构建 |
| `main.py` | 主入口，运行对比实验 |
| `config.py` | 全局配置 |

## 四种分片策略

| 策略 | 方法 | 优点 | 缺点 |
|------|------|------|------|
| **Character** | 按固定字符数切分 | 简单、确定性强 | 可能在句子中间截断，破坏语义 |
| **Recursive** | 递归按段落→换行→句子→单词切分 | 在多个层次寻找自然分割点 | 需要调参 chunk_size |
| **Semantic** | 基于 embedding 相似度分组相邻句子 | 语义连贯性最好 | 计算成本高、片段大小不可控 |
| **Paragraph** | 按空行（自然段落）切分 | 保留作者原始结构 | 片段大小差异大 |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. (可选) 配置 LLM API
cp .env.example .env
# 编辑 .env 填入 API Key；不配置则使用纯检索模式（无 LLM 生成）

# 3. 运行对比实验
python main.py

# 4. 仅运行单个策略
python main.py --strategy semantic

# 5. 交互式问答
python main.py --interactive

# 6. 重置向量数据库后重新构建
python main.py --reset
```

## 项目结构

```
rag/
├── documents/                  # 示例文档
│   ├── artificial_intelligence.txt
│   ├── python_programming.txt
│   └── cloud_computing.txt
├── chroma_db/                  # ChromaDB 持久化目录（自动生成）
├── config.py                   # 配置
├── document_loader.py          # 文档加载
├── chunking.py                 # 分片策略
├── vector_store.py             # 向量数据库
├── rag_chain.py                # RAG 链
├── main.py                     # 主入口
├── requirements.txt            # 依赖
└── .env.example                # 环境变量模板
```

## 技术栈

- **LangChain** — RAG 编排框架
- **ChromaDB** — 本地向量数据库
- **Sentence-Transformers** — 本地 Embedding 模型 (`all-MiniLM-L6-v2`)
- **Rich** — 终端美化输出
