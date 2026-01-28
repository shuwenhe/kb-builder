# 完整系统架构设计 - 6步分解方案

**项目目标**: 构建一个完整的知识库问答系统  
**设计方针**: 按照功能拆分成每个可以运行的子项目，按照开发的先后顺序逐步构建  
**总体进度**: ✅ Step 1-3 完成 | ⏳ Step 4-6 待开发

---

## 📐 系统全景图

```
┌─────────────────────────────────────────────────────────────────────┐
│                      用户界面 (Customer Service Web)                 │
│                          Step 6: Frontend                             │
│  (React/Vue + Next.js, 问答界面, 文档管理, 用户认证)                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTP/REST
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    业务API (Customer Service API)                     │
│                         Step 5: Backend                              │
│  (FastAPI, 用户管理, 问答历史, 权限控制, 日志记录)                      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │   RAG    │   │   用户   │   │   问答   │
        │ Service  │   │   数据库 │   │   历史   │
        │(Step 4)  │   │  (MySQL) │   │  (Redis) │
        └────┬─────┘   └──────────┘   └──────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌──────────────┐  ┌──────────────────┐
│   KB Loader  │  │ Query Embedder   │
│  (Step 3)    │  │  (Step 2)        │
└──────────────┘  └──────────────────┘
    │                    │
    ▼                    ▼
┌──────────────┐  ┌──────────────────┐
│  FAISS Index │  │ Embedding Models │
│  (Vector DB) │  │ (Ollama/OpenAI)  │
└──────────────┘  └──────────────────┘
    ▲
    │
┌───┴──────────────┐
│  KB Builder      │
│  (Step 3)        │
│  构建向量索引    │
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────────────┐
│ Document Parser │ ◄─── 源文档
│ (Step 1)        │     (Docs/)
│ 解析文档结构    │
└─────────────────┘
```

---

## 🔨 详细架构分解

### 第1步: Document Parser (docx-parser) ✅

**目标**: 将各种格式的文档转换为结构化的内容

**职责**:
```
输入: .docx, .doc, .pdf, .md, .txt 等文件
  ↓
解析逻辑:
  ├─ 识别文档结构 (标题、段落、列表、表格)
  ├─ 提取文本内容
  ├─ 保留格式信息 (粗体、斜体、链接)
  └─ 生成元数据 (作者、创建时间等)
  ↓
输出: DocContent 对象
  {
    filename: str,
    title: str,
    blocks: List[Block]  # 块包含类型、内容、格式信息
    metadata: Dict
  }
```

**核心模块**:
```python
# docx-parser/parser.py
class DocContent:
    filename: str
    blocks: List[Block]
    metadata: Dict

class Block:
    block_type: str  # "heading", "paragraph", "table", "list", etc.
    level: int  # 标题级别
    text: str
    metadata: Dict

def parse_docx(path: str) -> DocContent:
    """解析 DOCX 文件"""
    
def parse_pdf(path: str) -> DocContent:
    """解析 PDF 文件"""
    
def parse_markdown(path: str) -> DocContent:
    """解析 Markdown 文件"""
```

**特点**:
- ✅ 保留文档层级关系（标题-内容树）
- ✅ 支持多种格式
- ✅ 提供结构化输出

**输出示例**:
```json
{
  "filename": "user_guide.docx",
  "blocks": [
    {"type": "heading", "level": 1, "text": "用户指南"},
    {"type": "heading", "level": 2, "text": "快速开始"},
    {"type": "paragraph", "text": "首先安装..."},
    {"type": "list", "items": ["步骤1", "步骤2"]},
    {"type": "table", "rows": [[...], [...]]},
    {"type": "heading", "level": 2, "text": "常见问题"},
    ...
  ]
}
```

---

### 第2步: Embedding Service (embedding-service) ✅

**目标**: 将文本转换为向量表示

**职责**:
```
输入: 文本字符串 (单条或批量)
  ↓
嵌入过程:
  ├─ 选择嵌入模型 (mxbai-embed-large, text-embedding-3-small, etc.)
  ├─ 连接嵌入服务 (Ollama, OpenAI, Azure, etc.)
  ├─ 发送文本进行嵌入
  ├─ 接收1024维向量
  └─ 返回向量结果
  ↓
输出: numpy.ndarray (N, 1024)
```

**核心模块**:
```python
# embedding-service/embedding_service.py
from langchain.embeddings.base import Embeddings

class OllamaEmbeddings(Embeddings):
    """Ollama 本地嵌入"""
    
    def embed_documents(texts: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        
    def embed_query(query: str) -> List[float]:
        """单条嵌入查询"""

class OpenAIEmbeddings(Embeddings):
    """OpenAI API 嵌入"""

def build_embeddings(provider: str, model: str, **config) -> Embeddings:
    """工厂函数：根据 provider 创建嵌入对象"""
```

**REST API**:
```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/embed")
async def embed(texts: List[str]) -> List[List[float]]:
    """批量嵌入"""
    
@app.post("/embed-query")
async def embed_query(query: str) -> List[float]:
    """单条嵌入"""
```

**特点**:
- ✅ 统一接口支持多个 provider (Ollama, OpenAI, Azure)
- ✅ 支持批量和单条嵌入
- ✅ 可以作为库或 REST 服务使用
- ✅ 与 LangChain 集成

**性能指标**:
```
模型: mxbai-embed-large (1024维)
- 单条嵌入: ~100ms (GPU加速)
- 批量嵌入 (32条): ~200ms
- 吞吐量: ~150-300 条文本/秒
```

---

### 第3步: KB Builder (kb-builder) ✅ 已完成

**目标**: 将解析的文档转换为向量索引库

**职责**:
```
输入: 
  ├─ 源文档目录 (source_dir)
  ├─ 嵌入函数 (embedding_fn)
  └─ 配置参数 (chunk_size, overlap, etc.)
  ↓
构建流程 (6 个阶段):
  ├─ 阶段1: 扫描文档
  ├─ 阶段2: 解析文档 (调用 Step 1)
  ├─ 阶段3: 智能分块 (按段落、列表、表格)
  ├─ 阶段4: 嵌入向量 (调用 Step 2)
  ├─ 阶段5: 构建 FAISS 索引
  └─ 阶段6: 版本管理 (符号链接切换)
  ↓
输出目录结构:
  kb/
  ├─ 20250128_120530/  (版本1)
  │  ├─ index.faiss     (向量索引)
  │  ├─ chunks.jsonl    (块元数据)
  │  └─ manifest.json   (配置信息)
  ├─ 20250128_140000/  (版本2)
  │  └─ ...
  └─ latest → 20250128_140000  (当前版本符号链接)
```

**核心类**:
```python
# kb_builder/builder.py
class BuildConfig:
    max_len: int = 800
    overlap: int = 100
    batch_size: int = 32
    
def build_kb(
    source_dir: str,
    out_dir: str,
    embedding_fn: Callable,
    config: BuildConfig
) -> Manifest:
    """主构建函数"""

# kb_builder/loader.py
class KnowledgeBase:
    index: faiss.Index
    chunks: Dict[int, ChunkRecord]
    manifest: Manifest
    
def load_kb(kb_dir: str) -> KnowledgeBase:
    """加载知识库"""
```

**工作流示例**:
```python
from kb_builder import build_kb, load_kb
from embedding_service import build_embeddings

# 1️⃣ 构建 KB
embeddings = build_embeddings(provider="ollama", model="mxbai-embed-large")
manifest = build_kb(
    source_dir="./docs",
    out_dir="./kb",
    embedding_fn=embeddings.embed_documents
)

# 2️⃣ 加载 KB
kb = load_kb("./kb")

# 3️⃣ 查询
query_vec = embeddings.embed_query("什么是机器学习？")
distances, indices = kb.index.search(np.array([query_vec]), k=5)

for idx in indices[0]:
    chunk = kb.chunks[idx]
    print(f"📄 {chunk.file_path}: {chunk.excerpt_markdown}")
```

**特点**:
- ✅ 集成 Step 1 (docx-parser) 和 Step 2 (embedding-service)
- ✅ 智能分块保留上下文
- ✅ 容错处理 (部分失败不影响整体)
- ✅ 原子性版本切换 (零停机更新)
- ✅ 100% 测试覆盖

---

### 第4步: RAG Service (rag-service) ⏳ 待开发

**目标**: 实现检索增强生成 (Retrieval Augmented Generation)

**职责**:
```
输入: 用户查询文本
  ↓
RAG 流程:
  ├─ 查询向量化
  │  └─ 调用 Step 2: embedding_service.embed_query(query)
  │
  ├─ 向量检索
  │  └─ 调用 Step 3: kb.index.search(query_vec, k=5)
  │
  ├─ 检索块组织
  │  └─ 整理前 k 个相似块作为上下文
  │
  ├─ LLM 生成
  │  └─ 用检索的上下文 + 查询提示 LLM 生成回答
  │
  └─ 结果返回
     └─ 返回回答 + 源引用
  ↓
输出: 
{
  "answer": "机器学习是...",
  "sources": [
    {"file": "ML_Guide.docx", "chunk_id": "abc123", "text": "..."},
    {"file": "ML_Guide.docx", "chunk_id": "def456", "text": "..."}
  ],
  "confidence": 0.85
}
```

**核心模块**:
```python
# rag_service/rag_engine.py
class RAGEngine:
    def __init__(self, kb: KnowledgeBase, llm_model: str):
        self.kb = kb
        self.llm = build_chat_model(model=llm_model)
        self.embeddings = build_embeddings()
    
    def query(self, query: str, k: int = 5) -> Dict:
        """
        执行 RAG 查询
        
        步骤:
        1. 向量化查询
        2. FAISS 检索
        3. 组织上下文
        4. LLM 生成
        5. 返回结果
        """
        # 1️⃣ 向量化
        query_vec = self.embeddings.embed_query(query)
        
        # 2️⃣ 检索
        distances, indices = self.kb.index.search(
            np.array([query_vec]), k=k
        )
        
        # 3️⃣ 整理上下文
        context = ""
        sources = []
        for idx, dist in zip(indices[0], distances[0]):
            chunk = self.kb.chunks[idx]
            context += f"\n{chunk.excerpt_markdown}\n"
            sources.append({
                "file": chunk.file_path,
                "score": float(dist)
            })
        
        # 4️⃣ 生成回答
        prompt = f"""根据以下参考资料回答问题。

参考资料:
{context}

问题: {query}

答案:"""
        
        response = self.llm.invoke(prompt)
        
        # 5️⃣ 返回
        return {
            "answer": response.content,
            "sources": sources
        }

# FastAPI 服务
from fastapi import FastAPI

app = FastAPI()
rag_engine = RAGEngine(
    kb=load_kb("./kb"),
    llm_model="qwen2.5:7b"
)

@app.post("/query")
async def query(query: str, k: int = 5) -> Dict:
    return rag_engine.query(query, k)

@app.post("/chat")
async def chat(query: str, chat_history: List[Dict]) -> Dict:
    """支持多轮对话"""
    # 实现多轮对话逻辑
    pass
```

**集成关系**:
```
rag-service
  ├─ 依赖 kb-builder (Step 3)
  │  └─ 加载知识库 + 向量索引
  │
  ├─ 依赖 embedding-service (Step 2)
  │  └─ 向量化查询
  │
  └─ 新增组件
     ├─ LLM 调用 (Ollama, OpenAI)
     ├─ 上下文组织
     └─ 多轮对话管理
```

**API 设计**:
```
POST /query
{
  "query": "什么是机器学习？",
  "k": 5,
  "llm_model": "qwen2.5:7b",
  "temperature": 0.7
}

Response:
{
  "answer": "机器学习是...",
  "sources": [
    {
      "file": "ML_Guide.docx",
      "chunk_id": "abc123",
      "similarity_score": 0.92,
      "text": "..."
    }
  ],
  "tokens_used": 1250,
  "latency_ms": 3450
}
```

**特点**:
- ✅ 完整的 RAG 管道实现
- ✅ 支持多种 LLM 后端
- ✅ REST API 服务
- ✅ 多轮对话支持
- ✅ 结果溯源

---

### 第5步: Customer Service API (customer-service-api) ⏳ 待开发

**目标**: 提供业务级别的 API，包含认证、用户管理、历史记录等

**职责**:
```
输入: HTTP 请求 (带用户信息、请求内容)
  ↓
核心功能:
  ├─ 用户认证 (JWT Token, OAuth)
  ├─ 问答请求处理
  │  └─ 调用 Step 4: RAG Service
  ├─ 历史记录管理 (Redis/Database)
  ├─ 反馈收集 (对回答进行打分)
  ├─ 知识库管理 (上传文档、构建新 KB)
  ├─ 日志和监控
  └─ 速率限制 (Rate Limiting)
  ↓
输出: JSON 响应
```

**核心模块**:
```python
# customer_service_api/api.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer

app = FastAPI()
security = HTTPBearer()

# 依赖: RAG Service 客户端
rag_client = RAGServiceClient("http://rag-service:8000")

# ==================== 认证 ====================
@app.post("/auth/login")
async def login(username: str, password: str) -> Dict:
    """用户登录"""
    # 验证用户
    # 生成 JWT Token
    
@app.post("/auth/logout")
async def logout(token: str = Depends(security)) -> Dict:
    """用户登出"""

# ==================== 问答功能 ====================
@app.post("/chat")
async def chat(
    query: str,
    token: str = Depends(security),
    k: int = 5
) -> Dict:
    """主问答接口"""
    user = verify_token(token)
    
    # 调用 RAG Service
    result = rag_client.query(query, k=k)
    
    # 保存到历史记录
    save_chat_history(user.id, query, result)
    
    # 返回结果
    return result

@app.post("/feedback")
async def submit_feedback(
    query_id: str,
    score: int,  # 1-5
    comment: str = "",
    token: str = Depends(security)
) -> Dict:
    """用户反馈"""

@app.get("/chat-history")
async def get_chat_history(
    limit: int = 20,
    token: str = Depends(security)
) -> List[Dict]:
    """获取聊天历史"""

# ==================== 知识库管理 ====================
@app.post("/kb/upload")
async def upload_documents(
    files: List[UploadFile],
    token: str = Depends(security)
) -> Dict:
    """上传文档"""
    # 验证权限
    # 保存文档
    # 触发 KB 重建

@app.post("/kb/rebuild")
async def rebuild_kb(
    token: str = Depends(security)
) -> Dict:
    """重建知识库"""
    # 调用 KB Builder
    
@app.get("/kb/status")
async def get_kb_status() -> Dict:
    """获取 KB 状态"""

@app.get("/kb/documents")
async def list_documents(
    token: str = Depends(security)
) -> List[Dict]:
    """列出知识库中的文档"""

# ==================== 监控和日志 ====================
@app.get("/metrics")
async def get_metrics(token: str = Depends(security)) -> Dict:
    """获取服务指标"""
    
@app.get("/logs")
async def get_logs(
    limit: int = 100,
    token: str = Depends(security)
) -> List[Dict]:
    """获取操作日志"""
```

**数据模型**:
```python
# customer_service_api/models.py
from pydantic import BaseModel
from datetime import datetime

class User(BaseModel):
    id: str
    username: str
    email: str
    created_at: datetime
    role: str  # "admin", "user", "guest"

class ChatRecord(BaseModel):
    id: str
    user_id: str
    query: str
    answer: str
    sources: List[Dict]
    feedback_score: Optional[int]
    created_at: datetime

class Document(BaseModel):
    id: str
    filename: str
    file_path: str
    upload_time: datetime
    uploader_id: str
    kb_version: str
    status: str  # "indexing", "indexed", "failed"

class KBStatus(BaseModel):
    version: str
    total_documents: int
    total_chunks: int
    last_update: datetime
    status: str  # "ready", "building", "error"
```

**集成关系**:
```
customer-service-api
  ├─ 依赖 rag-service (Step 4)
  │  └─ 问答核心逻辑
  │
  ├─ 依赖 kb-builder (Step 3)
  │  └─ 知识库管理
  │
  ├─ 新增组件
  │  ├─ 用户认证 (OAuth/JWT)
  │  ├─ 数据库 (MySQL/PostgreSQL)
  │  ├─ 缓存 (Redis)
  │  ├─ 日志 (ELK Stack)
  │  └─ 监控 (Prometheus)
  │
  └─ 外部服务
     ├─ MySQL 用户数据库
     ├─ Redis 缓存层
     └─ 消息队列 (可选: 异步任务)
```

**API 端点汇总**:
```
认证:
  POST /auth/login
  POST /auth/logout
  POST /auth/refresh

问答:
  POST /chat
  GET /chat-history
  POST /feedback
  GET /feedback-stats

知识库:
  POST /kb/upload
  POST /kb/rebuild
  GET /kb/status
  GET /kb/documents
  DELETE /kb/documents/{doc_id}

管理:
  GET /metrics
  GET /logs
  GET /health
```

**特点**:
- ✅ 完整的业务逻辑层
- ✅ 用户认证和授权
- ✅ 历史记录管理
- ✅ 知识库管理
- ✅ 监控和可观测性

---

### 第6步: Customer Service Web (customer-service-web) ⏳ 待开发

**目标**: 提供用户友好的 Web 界面

**职责**:
```
组件:
  ├─ 问答界面 (Chat UI)
  │  ├─ 输入框 + 发送按钮
  │  ├─ 消息历史显示
  │  ├─ 源文档引用
  │  └─ 回答质量反馈
  │
  ├─ 知识库管理界面
  │  ├─ 文档上传
  │  ├─ 文档列表
  │  ├─ KB 版本管理
  │  └─ 构建进度显示
  │
  ├─ 用户管理界面 (仅管理员)
  │  ├─ 用户列表
  │  ├─ 权限管理
  │  └─ 审计日志
  │
  ├─ 个人中心
  │  ├─ 问答历史
  │  ├─ 收藏的问题
  │  ├─ 个人设置
  │  └─ API Key 管理
  │
  └─ 监控面板 (仅管理员)
     ├─ 服务状态
     ├─ 性能指标
     ├─ 错误日志
     └─ 用户分析
```

**技术栈**:
```
前端框架: React / Vue 3 / Svelte
元框架: Next.js (推荐) / Nuxt / SvelteKit
UI 库: Tailwind CSS, shadcn/ui, Material-UI
状态管理: Redux, Zustand, Pinia
实时通信: WebSocket (用于流式回答)
API 通信: fetch / axios / TanStack Query
```

**核心组件结构**:
```
customer-service-web/
├── app/
│   ├── layout.tsx          # 全局布局
│   ├── page.tsx            # 首页
│   ├── chat/
│   │   ├── page.tsx        # 问答页面
│   │   └── [id]/           # 对话详情
│   ├── kb-management/
│   │   ├── page.tsx        # KB 管理
│   │   └── documents/      # 文档列表
│   ├── profile/
│   │   ├── page.tsx        # 个人中心
│   │   └── settings/       # 设置
│   └── admin/              # 管理后台
│
├── components/
│   ├── ChatBox.tsx         # 聊天框
│   ├── SourcesList.tsx     # 源文档列表
│   ├── DocumentUpload.tsx   # 文档上传
│   ├── KBStatus.tsx        # KB 状态
│   ├── UserList.tsx        # 用户列表
│   └── Charts.tsx          # 统计图表
│
├── hooks/
│   ├── useChat.ts          # 聊天逻辑
│   ├── useAuth.ts          # 认证逻辑
│   └── useKB.ts            # KB 管理逻辑
│
├── lib/
│   ├── api-client.ts       # API 客户端
│   ├── websocket.ts        # WebSocket 连接
│   └── storage.ts          # 本地存储
│
└── styles/
    └── globals.css         # 全局样式
```

**主要页面示例**:

#### 问答页面 (Chat Page)
```tsx
export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const { sendQuery } = useChat();
  
  const handleSendQuery = async () => {
    setLoading(true);
    const response = await sendQuery(query);
    setMessages([
      ...messages,
      { role: "user", content: query },
      { role: "assistant", content: response.answer, sources: response.sources }
    ]);
    setQuery("");
    setLoading(false);
  };
  
  return (
    <div className="flex flex-col h-screen">
      {/* 消息历史 */}
      <div className="flex-1 overflow-auto p-4">
        {messages.map((msg, idx) => (
          <ChatMessage key={idx} message={msg} />
        ))}
      </div>
      
      {/* 输入框 */}
      <div className="border-t p-4">
        <div className="flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="输入你的问题..."
            className="flex-1 px-4 py-2 border rounded"
            onKeyDown={(e) => e.key === "Enter" && handleSendQuery()}
          />
          <button
            onClick={handleSendQuery}
            disabled={loading}
            className="px-6 py-2 bg-blue-500 text-white rounded"
          >
            {loading ? "发送中..." : "发送"}
          </button>
        </div>
      </div>
    </div>
  );
}
```

#### 知识库管理页面
```tsx
export default function KBManagementPage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [rebuilding, setRebuilding] = useState(false);
  const { uploadDocuments, rebuildKB, getDocuments } = useKB();
  
  useEffect(() => {
    loadDocuments();
  }, []);
  
  const loadDocuments = async () => {
    const docs = await getDocuments();
    setDocuments(docs);
  };
  
  const handleUpload = async (files: File[]) => {
    await uploadDocuments(files);
    loadDocuments();
  };
  
  const handleRebuild = async () => {
    setRebuilding(true);
    await rebuildKB();
    setRebuilding(false);
    loadDocuments();
  };
  
  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-8">知识库管理</h1>
      
      {/* 上传区 */}
      <DocumentUpload onUpload={handleUpload} />
      
      {/* 文档列表 */}
      <div className="mt-8">
        <h2 className="text-xl font-bold mb-4">文档列表</h2>
        <DocumentList documents={documents} />
      </div>
      
      {/* 重建按钮 */}
      <button
        onClick={handleRebuild}
        disabled={rebuilding}
        className="mt-4 px-6 py-2 bg-green-500 text-white rounded"
      >
        {rebuilding ? "重建中..." : "重建知识库"}
      </button>
    </div>
  );
}
```

**API 集成**:
```typescript
// lib/api-client.ts
class APIClient {
  constructor(baseURL: string, token?: string) {
    this.client = axios.create({
      baseURL,
      headers: { Authorization: `Bearer ${token}` }
    });
  }
  
  async chat(query: string, k: number = 5) {
    return this.client.post("/chat", { query, k });
  }
  
  async uploadDocuments(files: File[]) {
    const formData = new FormData();
    files.forEach(f => formData.append("files", f));
    return this.client.post("/kb/upload", formData);
  }
  
  async getChatHistory() {
    return this.client.get("/chat-history");
  }
}
```

**特点**:
- ✅ 现代化的 React/Vue UI
- ✅ 实时聊天体验
- ✅ 知识库管理功能
- ✅ 用户认证集成
- ✅ 响应式设计
- ✅ 性能优化 (Code Splitting, Lazy Loading)

---

## 🔄 数据流和集成

### 端到端数据流

```
1️⃣ 用户上传文档
   User → Web (Step 6) → API (Step 5) → KB Builder (Step 3)
   
2️⃣ 构建知识库
   KB Builder (Step 3) 
     ├─ 调用 docx-parser (Step 1) 解析文档
     ├─ 调用 embedding-service (Step 2) 生成向量
     ├─ 构建 FAISS 索引
     └─ 返回 Manifest 信息给 API (Step 5)

3️⃣ 用户发起查询
   User → Web (Step 6) → API (Step 5) → RAG Service (Step 4)
   
4️⃣ RAG 处理流程
   RAG Service (Step 4)
     ├─ 调用 embedding-service (Step 2) 向量化查询
     ├─ 调用 KB Loader (Step 3) 加载知识库
     ├─ 使用 FAISS 检索相似块
     ├─ 用检索结果调用 LLM 生成回答
     └─ 返回回答 + 源引用给 API (Step 5)

5️⃣ 返回结果给用户
   RAG Service (Step 4) → API (Step 5) → Web (Step 6) → User
```

### 服务依赖关系

```
Web (Step 6)
    ↓ (REST API)
API (Step 5)
    ├─ ↓
    │ RAG Service (Step 4)
    │    ├─ ↓
    │    │ KB Builder Loader (Step 3)
    │    │    ├─ FAISS Index
    │    │    └─ chunks.jsonl
    │    │
    │    └─ ↓
    │      embedding-service (Step 2)
    │
    ├─ ↓
    │ KB Builder (Step 3)
    │    ├─ ↓
    │    │ docx-parser (Step 1)
    │    │
    │    └─ ↓
    │      embedding-service (Step 2)
    │
    └─ ↓
     Database (MySQL)
     Cache (Redis)
```

---

## 📊 部署架构

### 本地开发

```
localhost:3000   ← Next.js Web
         ↓ (http://localhost:8005)
localhost:8005   ← FastAPI Customer Service API
         ├─ ↓ (http://localhost:8004)
         │ localhost:8004 ← FastAPI RAG Service
         │         ├─ ↓
         │         │ ./kb (FAISS Index)
         │         │
         │         └─ ↓ (HTTP)
         │           localhost:11434 (Ollama)
         │
         ├─ ↓
         │ localhost:3306 (MySQL)
         │
         └─ ↓
           localhost:6379 (Redis)
```

### 生产部署 (Docker Compose)

```yaml
version: "3"

services:
  # 前端
  web:
    build: ./customer-service-web
    ports:
      - "3000:3000"
    depends_on:
      - api
    environment:
      NEXT_PUBLIC_API_URL: http://api:8005

  # 业务 API
  api:
    build: ./customer-service-api
    ports:
      - "8005:8000"
    depends_on:
      - rag-service
      - mysql
      - redis
    environment:
      RAG_SERVICE_URL: http://rag-service:8004
      DATABASE_URL: mysql://root:pass@mysql:3306/kb_db
      REDIS_URL: redis://redis:6379

  # RAG 服务
  rag-service:
    build: ./rag-service
    ports:
      - "8004:8000"
    depends_on:
      - ollama
    volumes:
      - ./kb:/app/kb
    environment:
      OLLAMA_BASE_URL: http://ollama:11434

  # Ollama (LLM + Embedding)
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  # 数据库
  mysql:
    image: mysql:8.0
    ports:
      - "3306:3306"
    environment:
      MYSQL_ROOT_PASSWORD: root_password
      MYSQL_DATABASE: kb_db
    volumes:
      - mysql_data:/var/lib/mysql

  # 缓存
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  ollama_data:
  mysql_data:
  redis_data:
```

---

## 🎯 项目进度追踪

| 步骤 | 项目名 | 描述 | 状态 | 测试 | 文档 |
|-----|--------|------|------|------|------|
| 1 | docx-parser | 文档解析 | ✅ | ✅ | ✅ |
| 2 | embedding-service | 向量嵌入 | ✅ | ✅ | ✅ |
| 3 | kb-builder | 知识库构建 | ✅ | ✅ 11/11 | ✅ |
| 4 | rag-service | RAG 查询 | ⏳ 待开发 | - | - |
| 5 | customer-service-api | 业务 API | ⏳ 待开发 | - | - |
| 6 | customer-service-web | Web 前端 | ⏳ 待开发 | - | - |

---

## 📝 关键设计决策

### 1. 为什么分成 6 个步骤？

✅ **解耦**: 每个步骤独立开发、测试、部署  
✅ **重用**: 每个步骤可以单独使用或组合  
✅ **灵活**: 可以替换其中任何一步的实现  
✅ **可测试**: 清晰的输入/输出接口便于测试  

### 2. 为什么使用 FAISS？

- 最快的开源向量检索库
- 支持 GPU 加速 (faiss-gpu)
- 支持多种量化方式 (PQ, IVF)
- Python 易用性好

### 3. 为什么支持 Ollama + OpenAI？

- Ollama: 本地部署，隐私性好，成本低
- OpenAI: API 调用，体验好，模型强大
- 灵活切换，不被锁定

### 4. 为什么使用 FastAPI？

- 性能高 (异步支持)
- 自动 API 文档 (Swagger/OpenAPI)
- 类型检查 (Pydantic)
- 易于扩展

---

## 🚀 快速开始

### 开发模式

```bash
# 1️⃣ 启动 Ollama
ollama serve

# 2️⃣ 拉取模型
ollama pull mxbai-embed-large
ollama pull qwen2.5:3b

# 3️⃣ 启动 MySQL 和 Redis
docker-compose -f docker-compose.dev.yml up

# 4️⃣ 构建知识库
cd kb-builder
python example_build.py

# 5️⃣ 启动 RAG 服务 (后续)
cd rag-service
uvicorn main:app --reload --port 8004

# 6️⃣ 启动 API (后续)
cd customer-service-api
uvicorn main:app --reload --port 8005

# 7️⃣ 启动 Web (后续)
cd customer-service-web
npm run dev
```

### 测试工作流

```bash
# 测试 Step 1: docx-parser
cd docx-parser
pytest

# 测试 Step 2: embedding-service
cd embedding-service
pytest

# 测试 Step 3: kb-builder
cd kb-builder
make test

# 测试 Step 4-6 (后续)
cd rag-service && pytest
cd customer-service-api && pytest
cd customer-service-web && npm test
```

---

## 总结

这个项目采用了**微服务架构**的思想，但使用了更轻量级的实现方式：

- **每个步骤都是独立的 Python/Node.js 项目**
- **通过 REST API 进行通信**
- **支持本地开发和 Docker 部署**
- **逐步构建，可以随时停止和继续**

从 Step 1 开始，每一步都：
1. 有清晰的输入输出
2. 有完整的测试覆盖
3. 可以独立运行和使用
4. 文档齐全

这样做的好处：
✅ 学习成本低 (每次只关注一个步骤)  
✅ 开发效率高 (可以并行开发)  
✅ 系统稳定性好 (模块化设计)  
✅ 生产就绪 (每个模块都经过验证)  

**下一步**: 开始 Step 4 (rag-service) 的开发！

