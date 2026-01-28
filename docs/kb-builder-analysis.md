# kb-builder 详细技术分析

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                     kb-builder                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: .doc/.docx files → Processing → Output: FAISS KB   │
│         (docs/)              (stages)      (kb/current)     │
│                                                              │
│  ┌──────────┐  ┌───────────┐  ┌─────────┐  ┌────────────┐  │
│  │ Document │→ │  Parse &  │→ │Embedding│→ │   Index    │  │
│  │  Scan    │  │  Chunk    │  │ & Batch │  │  Creation  │  │
│  └──────────┘  └───────────┘  └─────────┘  └────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 2. 核心模块分析

### builder.py (754行) - 构建引擎

**主要职责:**
- 文档扫描和转换
- 文本分块（智能分割）
- 向量嵌入
- FAISS索引构建
- 版本管理

**关键函数流程:**

```python
build_kb() 
  ├── scan_documents()        # 扫描源目录
  │   └── 过滤: 锁定文件、空文件、非Word文档
  │
  ├── convert_doc_to_docx()   # .doc → .docx 转换
  │   ├── unstructured库
  │   ├── LibreOffice/soffice
  │   ├── macOS textutil
  │   └── antiword (降级)
  │
  ├── parse_docx()            # 解析文档结构
  │   └── 调用 docx-parser 获取blocks
  │
  ├── _collect_chunks()       # 智能分块
  │   ├── 段落处理
  │   │   ├── 列表项检测
  │   │   ├── 文本分割 (with overlap)
  │   │   └── 重复短段落过滤
  │   │
  │   └── 表格处理
  │       ├── 行数拆分
  │       └── Markdown + 线性化转换
  │
  ├── _embed_documents_with_retry()  # 嵌入 (带重试)
  │   ├── 批处理 (batch_size=32)
  │   ├── 指数退避重试
  │   └── 降级: 单条嵌入
  │
  ├── faiss.normalize_L2()    # L2规范化
  ├── IndexFlatIP()           # 构建索引
  │
  └── _activate_version()     # 原子切换
      └── 符号链接 (或fallback: 复制)
```

### loader.py (73行) - 加载器

**结构:**
```
KnowledgeBase (dataclass)
├── index: faiss.Index        # FAISS向量索引
├── chunks: Dict[vector_id → ChunkRecord]
└── manifest: Manifest        # 元数据

load_kb(kb_path) → KnowledgeBase
├── 读取 index.faiss
├── 读取 chunks.jsonl (JSONL格式)
└── 读取 manifest.json
```

### schemas.py (35行) - 数据模型

```python
ChunkRecord:
  ├── vector_id: int          # FAISS向量ID
  ├── chunk_id: str           # SHA1(file:hash:type:index)
  ├── file_path: str          # 相对路径
  ├── title_path: List[str]   # 层级路径 ["Ch1", "Sec1.1"]
  ├── chunk_type: str         # "paragraph" | "table"
  ├── chunk_index: int        # 文件内序号
  ├── doc_hash: str           # SHA1(file content)
  ├── text_for_embedding: str # 用于嵌入的文本
  └── excerpt_markdown: str   # 显示用MD格式

Manifest:
  ├── kb_version: str         # 时间戳版本 "20240128-164300"
  ├── source_dir: str         # 源目录路径
  ├── build_time: str         # ISO 8601时间
  ├── embedding_model: str    # "mxbai-embed-large"
  ├── llm_provider_default: str # "ollama"
  ├── faiss_metric: str       # "cosine" (L2归一化后)
  ├── doc_count: int          # 处理文档数
  ├── chunk_count: int        # 总chunk数
  └── failed_files: List[str] # 失败文件列表
```

### utils.py (153行) - 工具函数

| 函数 | 功能 | 说明 |
|------|------|------|
| `normalize_text()` | 文本规范化 | 去除多余空格、特殊空格符 |
| `split_text()` | 文本分割 | 带重叠分割 (overlap参数) |
| `split_list_items()` | 列表项检测 | 正则识别: 1. 2. ① ② 等 |
| `sha1_file()` | 文件哈希 | SHA1(file content) |
| `sha1_text()` | 文本哈希 | SHA1(text) |
| `table_to_markdown()` | 表格转MD | 生成\|格式表格 |
| `table_to_linearized_text()` | 表格线性化 | "key1: val1; key2: val2" |
| `safe_relpath()` | 安全相对路径 | fallback绝对路径 |
| `iter_batches()` | 批处理迭代 | 分批yield |

## 3. 工作流详解

### 阶段1: 文档扫描 (scan_documents)

```
源目录
├── document1.docx  ✓ included
├── document2.doc   ✓ included (需转换)
├── ~$document3.docx ✗ skipped (Word lock)
├── empty.docx      ✗ skipped (0字节)
├── image.jpg       ✗ skipped (非Word)
└── readme.txt      ✗ skipped (非Word)

输出:
  included = ["document1.docx", "document2.doc"]
  skipped = [
    {path: "~$document3.docx", reason: "word_lock"},
    {path: "empty.docx", reason: "empty_file"},
    ...
  ]
```

### 阶段2: 文档解析 (parse_docx)

```
document.docx
  ↓
parse_docx() → Block[]
  ├── Block {
  │     block_type: "paragraph"
  │     text: "Installation Steps"
  │     title_path: ["Chapter 1", "Section 1.1"]
  │   }
  │
  ├── Block {
  │     block_type: "table"
  │     table_rows: [["Name", "Value"], ...]
  │     table_markdown: "| Name | Value |\n| --- | --- |"
  │   }
  │
  └── Block { ... }
```

### 阶段3: 智能分块 (_collect_chunks)

**段落处理:**
```
原文本:
"1. First item
 2. Second item  
 3. Third item"

split_list_items() 检测:
  ↓
["1. First item", "2. Second item", "3. Third item"]

split_text() 分割 (max_len=800):
  ↓
如果每项 < 800字符，保持原样
如果 > 800字符，分割为多个chunks (with 100字符overlap)
```

**表格处理:**
```
表格 (1000行)
  ↓
_split_table_rows() 按max_len分割
  ↓
表格块1 (500行) → table_to_markdown() → "| ... |"
表格块2 (500行) → table_to_linearized_text() → "col1: val1; col2: val2"
  ↓
生成ChunkRecord[]
```

**标题路径:**
```
文档结构:
  Chapter 1
    ├── Section 1.1
    │   └── "某段落文本"
    └── Section 1.2
        └── "另一段落文本"

当处理"某段落文本"时:
  title_path = ["Chapter 1", "Section 1.1"]
  text_for_embedding = "Chapter 1 > Section 1.1\n某段落文本"
  
这样可以在搜索结果中显示上下文。
```

### 阶段4: 向量嵌入 (with retry)

```
chunks: ChunkRecord[]
  ↓
分批处理 (batch_size=32)
  ↓
batch_texts = [text1, text2, ..., text32]
  ↓
embeddings.embed_documents(batch_texts)
  ↓ (失败处理)
  ├─ 如果批处理失败:
  │   └─ 降级为逐条嵌入
  │       └─ 如果单条也失败: 记录失败，跳过
  │
  └─ 指数退避重试 (0.5s → 1s → 2s)

输出:
  vectors: np.array([
    [0.1, 0.2, ..., 0.9],  # 1024维
    [0.2, 0.3, ..., 0.8],
    ...
  ])
```

### 阶段5: FAISS索引 (with L2 normalization)

```
vectors: float32 array (N × 1024)
  ↓
faiss.normalize_L2(vectors)  # 每向量 L2范数 = 1
  ↓
index = faiss.IndexFlatIP(1024)  # Inner Product
  ↓
index.add(vectors)  # 添加向量
  ↓
faiss.write_index(index, "index.faiss")

搜索时:
  query_vector (1024维) → normalize_L2
    ↓
  scores, indices = index.search(query, k=5)
    ↓ (IndexFlatIP使用IP相似度)
  score ∈ [0, 1] (余弦相似度等价)
```

### 阶段6: 版本管理 (atomic switch)

```
kb/
├── versions/
│   ├── 20240128-120000/  (old)
│   │   ├── index.faiss
│   │   ├── chunks.jsonl
│   │   └── manifest.json
│   │
│   └── 20240128-164300/  (new, just built)
│       ├── index.faiss
│       ├── chunks.jsonl
│       ├── manifest.json
│       └── build_log.json
│
└── current → versions/20240128-164300 (symlink)
```

**原子切换策略:**
```
try:
  os.symlink("versions/20240128-164300", "current_tmp")
  os.replace("current_tmp", "current")  # 原子操作
except OSError:
  # fallback: 复制整个目录 (不支持symlink的系统)
  shutil.copytree(version_dir, current_path)
```

## 4. 输出结构详解

### index.faiss (FAISS二进制)
```
Header: FAISS格式标识
Data: N × 1024 float32向量
  (8字节*1024*N = 8MB per 1000 chunks)
```

### chunks.jsonl (逐行JSON)
```json
{"vector_id":0,"chunk_id":"abc123...","file_path":"docs/sample.docx",...}
{"vector_id":1,"chunk_id":"def456...","file_path":"docs/sample.docx",...}
...
```

### manifest.json (元数据)
```json
{
  "kb_version": "20240128-164300",
  "doc_count": 3,
  "chunk_count": 156,
  "embedding_model": "mxbai-embed-large",
  "faiss_metric": "cosine",
  "failed_files": []
}
```

### build_log.json (详细日志)
```json
{
  "kb_version": "20240128-164300",
  "included_files": ["docs/sample.docx"],
  "skipped_files": [],
  "failed_files": [],
  "degraded_files": [],
  "embedding_failed_chunks": []
}
```

## 5. 性能特性

| 指标 | 值 | 说明 |
|------|------|------|
| **嵌入维度** | 1024 | mxbai-embed-large |
| **批大小** | 32 | 可配置 |
| **最大chunk长** | 800字符 | 可配置 |
| **重叠大小** | 100字符 | 可配置 |
| **重试次数** | 3 | 指数退避 |
| **内存占用** | 8MB/1000chunks | FAISS float32 |
| **构建速度** | ~50-100 chunks/sec | 取决于Ollama |

## 6. 错误处理

```
build_kb():
  ├── 文档层面
  │   ├── 转换失败 → antiword降级 → blocks_from_text()
  │   ├── 解析失败 → 记录到failed_files
  │   └── 转换超时 → 跳过
  │
  ├── 块层面
  │   ├── 嵌入失败 → 逐条重试
  │   └── 嵌入仍失败 → 记录到embedding_failed_chunks
  │
  └── 索引层面
      ├── 向量数=0 → RuntimeError
      └── 向量维度不匹配 → RuntimeError
```

## 7. 与其他项目关系

```
依赖关系:
  kb-builder
  ├── docx-parser (Step 1)  ← parse_docx()
  └── embedding-service (Step 2) ← embeddings.embed_documents()

被依赖:
  kb-builder
  └── rag-service (Step 4) ← load_kb() + vector search
```

## 8. 代码特点

| 特点 | 说明 |
|------|------|
| **类型注解** | 完整，支持IDE提示 |
| **错误处理** | 详细，支持部分失败继续 |
| **进度显示** | tqdm进度条 |
| **日志记录** | 详细的build_log.json |
| **可配置性** | max_len, overlap, batch_size等可参数化 |
| **测试覆盖** | 11个单元测试 |
| **模块解耦** | builder/loader/schemas分离 |

## 总结

**kb-builder 是一个生产级别的知识库构建工具:**
- ✅ 鲁棒的文档处理（多种转换工具链）
- ✅ 智能的内容分块（列表+表格识别）
- ✅ 高效的向量索引（FAISS + L2归一化）
- ✅ 原子版本管理（无缝切换）
- ✅ 详细的错误记录（部分失败不影响整体）
- ✅ 完整的元数据（可追溯性）
