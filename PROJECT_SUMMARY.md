# KB Builder Project Summary

## Project Status: ✅ COMPLETED

kb-builder 是项目分解的第3步，已成功创建为独立的可运行项目。

## 项目结构

```
kb-builder/
├── kb_builder/              # 核心包
│   ├── __init__.py         # 公共API导出
│   ├── builder.py          # 知识库构建逻辑 (590行)
│   ├── loader.py           # 知识库加载功能 (70行)
│   ├── schemas.py          # Pydantic数据模型 (30行)
│   └── utils.py            # 文本处理工具 (180行)
├── tests/                   # 测试套件
│   ├── test_schemas.py     # 模型测试 (3个测试)
│   └── test_utils.py       # 工具函数测试 (8个测试)
├── example_build.py         # 构建示例脚本
├── example_load.py          # 加载查询示例脚本
├── requirements.txt         # 依赖清单
├── setup.py                 # 包安装配置
├── Makefile                 # 构建命令
├── pytest.ini               # 测试配置
├── .gitignore              # Git忽略规则
└── README.md               # 项目文档
```

## 核心功能

### 1. 文档扫描与转换 (builder.py)
- 扫描目录中的 .doc/.docx 文件
- 多种转换工具链：
  - unstructured库
  - LibreOffice/soffice
  - macOS textutil
  - antiword (降级方案)
- 过滤锁定文件和空文件

### 2. 智能分块 (builder.py)
- 段落分块：
  - 检测列表项并拆分
  - 文本重叠切分
  - 过滤重复短段落（页眉/页脚）
- 表格分块：
  - 按行数拆分大表
  - 生成Markdown格式
  - 线性化文本格式

### 3. 向量嵌入 (builder.py)
- 批量嵌入处理
- 指数退避重试机制
- 失败降级到单条嵌入
- 支持LangChain嵌入接口

### 4. FAISS索引 (builder.py)
- L2归一化处理
- IndexFlatIP (余弦相似度)
- 版本化输出
- 原子化符号链接切换

### 5. 知识库加载 (loader.py)
- 加载FAISS索引
- 读取chunk元数据
- 解析manifest信息
- 返回KnowledgeBase对象

### 6. 工具函数 (utils.py)
- 文本规范化
- SHA1哈希计算
- 表格转换 (Markdown/线性文本)
- 列表项检测与拆分
- 批次迭代器

## 测试结果

```
11 passed, 3 warnings in 0.21s
```

**测试覆盖**:
- ✅ ChunkRecord模型创建
- ✅ Manifest模型创建
- ✅ 默认值处理
- ✅ 文本规范化
- ✅ 文本分块（短文本/长文本/重叠）
- ✅ 表格Markdown转换
- ✅ 表格线性化文本
- ✅ 列表项拆分
- ✅ SHA1哈希生成

## 技术栈

| 依赖 | 版本 | 用途 |
|------|------|------|
| faiss-cpu | >=1.8.0 | 向量索引 |
| numpy | >=1.24.0 | 数值计算 |
| tqdm | >=4.65.0 | 进度条 |
| pydantic | >=2.0.0 | 数据验证 |
| docx-parser | >=0.1.0 | 文档解析 (Step 1) |
| langchain | >=0.3.0 | LLM抽象 |
| langchain-community | >=0.3.0 | 社区集成 |
| pytest | >=7.0.0 | 测试框架 (dev) |

## 输出格式

### 目录结构
```
kb/
├── versions/
│   ├── 20240101-120000/
│   │   ├── index.faiss       # FAISS向量索引
│   │   ├── chunks.jsonl      # chunk元数据
│   │   ├── manifest.json     # 构建清单
│   │   └── build_log.json    # 详细日志
│   └── 20240102-093000/
│       └── ...
└── current -> versions/20240102-093000  # 当前版本符号链接
```

### manifest.json 示例
```json
{
  "kb_version": "20240102-093000",
  "source_dir": "./docs",
  "build_time": "2024-01-02T09:30:00Z",
  "embedding_model": "mxbai-embed-large",
  "llm_provider_default": "ollama",
  "faiss_metric": "cosine",
  "doc_count": 50,
  "chunk_count": 1234,
  "failed_files": []
}
```

### chunks.jsonl 格式
每行一个JSON对象：
```json
{
  "vector_id": 0,
  "chunk_id": "abc123...",
  "file_path": "docs/manual.docx",
  "title_path": ["Chapter 1", "Section 1.1"],
  "chunk_type": "paragraph",
  "chunk_index": 0,
  "doc_hash": "def456...",
  "text_for_embedding": "Chapter 1 > Section 1.1\nContent text...",
  "excerpt_markdown": "Content text..."
}
```

## 使用示例

### 1. 安装
```bash
cd kb-builder
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. 构建知识库
```bash
# 设置环境变量
export SOURCE_DIR=./docs
export OUT_DIR=./kb
export EMBED_MODEL=mxbai-embed-large
export OLLAMA_BASE_URL=http://localhost:11434

# 运行构建脚本
python example_build.py
```

### 3. 查询知识库
```bash
# 加载并查询
python example_load.py

# 交互式查询
Query: 如何安装软件？
Top 5 results:
[1] Score: 0.8567
    File:  docs/install_guide.docx
    Title: 安装指南 > 系统要求
    Text:  系统要求: Windows 10或更高版本...
```

### 4. 编程接口
```python
from langchain_community.embeddings import OllamaEmbeddings
from kb_builder import build_kb, load_kb

# 构建
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://localhost:11434"
)

manifest = build_kb(
    source_dir="./docs",
    out_dir="./kb",
    embeddings_client=embeddings,
    embed_model="mxbai-embed-large",
    provider="ollama"
)

# 加载
kb = load_kb("./kb/current")
print(f"Loaded {kb.manifest.chunk_count} chunks")

# 搜索
import numpy as np
import faiss

query_vector = embeddings.embed_query("安装步骤")
query_array = np.array([query_vector], dtype="float32")
faiss.normalize_L2(query_array)

scores, indices = kb.index.search(query_array, k=5)
for idx, score in zip(indices[0], scores[0]):
    chunk = kb.chunks[int(idx)]
    print(f"Score: {score:.4f}, File: {chunk.file_path}")
```

## Makefile命令

| 命令 | 说明 |
|------|------|
| `make venv` | 创建虚拟环境 |
| `make install` | 安装依赖 |
| `make test` | 运行测试 |
| `make example-build` | 运行构建示例 |
| `make example-query` | 运行查询示例 |
| `make clean` | 清理生成文件 |

## 与其他项目的关系

### 依赖项目
- **docx-parser** (Step 1): 提供 parse_docx(), blocks_from_text() 等函数
- **embedding-service** (Step 2): kb-builder调用其API进行向量嵌入

### 被依赖项目
- **rag-service** (Step 4): 将使用kb-builder构建的知识库进行RAG查询
- **customer-service-api** (Step 5): 后端API集成知识库功能
- **customer-service-web** (Step 6): 前端展示知识库查询结果

## 核心设计模式

1. **工厂模式**: embeddings_client作为参数传入，支持不同提供商
2. **批处理**: 批量嵌入，批量FAISS索引
3. **重试机制**: 指数退避，失败降级
4. **版本管理**: 时间戳版本，原子切换
5. **渐进式解析**: 增量处理，进度条反馈
6. **降级策略**: 多种转换工具链，antiword降级

## 已知限制

1. docx-parser依赖路径硬编码 (需要在 ../../docx-parser)
2. 仅支持 .doc/.docx 格式
3. 大文件内存占用较高 (需要加载所有向量到内存)
4. 不支持增量更新 (每次全量重建)

## 下一步

继续第4个项目：**rag-service**
- 集成kb-builder和embedding-service
- 实现相似度搜索
- LLM上下文增强
- 生成带引用的回答
