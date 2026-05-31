# AgentX Backend — Complete Technical Documentation

> **Production-grade Agentic AI Backend** | FastAPI · LangGraph · Google Gemini · Qdrant · MongoDB · Redis · MCP
> 
> **New**: Dual-layer observability (LangSmith trace + native cost analytics), Admin Dashboard with real per-token cost tracking

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Complete Directory Structure](#3-complete-directory-structure)
4. [Application Startup & Lifecycle](#4-application-startup--lifecycle)
5. [API Routes Reference](#5-api-routes-reference)
6. [The Chat Pipeline (Agentic Conversation Flow)](#6-the-chat-pipeline-agentic-conversation-flow)
7. [The Agentic RAG Pipeline](#7-the-agentic-rag-pipeline)
8. [Model Context Protocol (MCP) Integration](#8-model-context-protocol-mcp-integration)
9. [Persistent Memory Bank](#9-persistent-memory-bank)
10. [Redis: Caching, Rate-Limiting & Background Jobs](#10-redis-caching-rate-limiting--background-jobs)
11. [Authentication & Security](#11-authentication--security)
12. [Dual-Layer Observability: LangSmith + Native Cost Analytics](#12-dual-layer-observability-langsmith--native-cost-analytics)
13. [Admin Analytics Dashboard](#13-admin-analytics-dashboard)
14. [Hook System (Middleware for Tool Calls)](#14-hook-system-middleware-for-tool-calls)
15. [Database Schema (MongoDB)](#15-database-schema-mongodb)
16. [Environment Variables Reference](#16-environment-variables-reference)
17. [Running Locally & E2E Testing](#17-running-locally--e2e-testing)

---

## 1. Project Overview

AgentX is not a simple chatbot — it is a full **Agentic AI Workspace** built on a multi-phase, production-grade backend architecture. The backend orchestrates:

- **Stateful, streaming conversations** via a LangGraph state machine
- **Agentic RAG**: Self-correcting document retrieval with hallucination detection
- **External Tool Use**: Connecting to real-world services via the Model Context Protocol (MCP)
- **Long-term Memory**: Silently learning and storing user facts across sessions
- **Asynchronous Background Processing**: Non-blocking PDF ingestion queue with real-time job polling
- **Enterprise Hardening**: Rate limiting, correlation-ID logging, distributed caching, and health probes

The system is designed to run on Render/Vercel in production and gracefully degrade when optional services (Redis) are unavailable in local development.

---

## 2. Architecture Overview

```mermaid
graph TD
    subgraph client [Client]
        FE["React Frontend"]
    end

    subgraph fastapi_app [FastAPI Application]
        MW["Middleware Stack\n(CORS · Logging · Rate-Limit · Correlation-ID)"]
        ROUTES["API Routers\n(Auth · Chat · RAG · Ingest · MCP · Tools · User)"]
    end

    subgraph chat_pipeline [Chat Pipeline]
        CC["ChatController\n(HTTP Layer)"]
        CS["ChatService\n(Orchestrator)"]
        HS["HistoryService\n(Redis Cache → MongoDB)"]
        PB["PromptBuilder\n(Dynamic System Prompt)"]
        MS["MemoryService\n(Background Extraction)"]
        CG["LangGraph: Chat Graph"]
    end

    subgraph chat_nodes [Chat Graph Nodes]
        SN["setup_node\n(Load History + Memory)"]
        CM["chat_model_node\n(Gemini LLM + Dynamic Tools)"]
        RT{"route_tools\nConditional Router"}
        NTN["native_tool_node\n(Internal Python Tools)"]
        MTN["mcp_tool_node\n(External MCP Servers)"]
    end

    subgraph rag_pipeline [RAG Pipeline]
        RW["RAG Workflow\n(LangGraph)"]
        PRN["parallel_retrieve_node\n(Qdrant + Tavily concurrently)"]
        GN["grade_documents\n(Relevance Grader)"]
        AN["agent_node\n(Agentic Reasoning Loop)"]
        HN["hallucination_node\n(Groundedness Check)"]
    end

    subgraph storage [Storage]
        MONGO[("MongoDB\n(Atlas)\nUsers · Chats · Memory")]
        QDRANT[("Qdrant Cloud\nVector DB")]
        REDIS[("Redis\n(Upstash)\nCache · Jobs · Rate-Limit")]
    end

    subgraph external [External Services]
        GEMINI["Google Gemini\n(LLM + Embeddings)"]
        LLAMA["LlamaParse Cloud\n(PDF Parsing)"]
        TAVILY["Tavily\n(Web Search)"]
        MCP_SERVER["MCP Servers\n(Google Drive, Filesystem, etc.)"]
    end

    FE -->|"HTTP/SSE"| MW
    MW --> ROUTES
    ROUTES --> CC
    CC --> CS
    CS --> HS
    CS --> PB
    CS --> CG
    CG --> SN --> CM --> RT
    RT -->|Tool Call| NTN
    RT -->|MCP Call| MTN
    RT -->|Text Response| FE
    NTN --> CM
    MTN --> CM
    MTN --> MCP_SERVER
    CS -->|asyncio.create_task| MS
    MS --> MONGO

    ROUTES --> RW
    RW --> PRN
    PRN -->|"asyncio.gather"| QDRANT
    PRN -->|"asyncio.gather"| TAVILY
    PRN --> GN --> AN --> HN

    HS --> REDIS
    HS --> MONGO
    CM --> GEMINI
    PRN --> GEMINI
    AN --> GEMINI
    HN --> GEMINI
```

---

## 3. Complete Directory Structure

```text
backend/
├── main.py                    # FastAPI app, middleware stack, lifespan
│
├── routes/                    # API endpoint definitions (thin HTTP layer only)
│   ├── auth_routes.py         # POST /auth/signup, /auth/login, /auth/logout, GET /auth/me
│   ├── chat_routes.py         # POST /chat/stream (SSE)
│   ├── conversation_routes.py # GET/DELETE /conversations, GET /conversations/{id}/messages
│   ├── upload_routes.py       # POST /api/v1/ingest/upload, GET /api/v1/ingest/job/{id}
│   ├── rag_routes.py          # POST /api/v1/rag/chat, /api/v1/rag/retrieve, GET /api/v1/rag/files
│   ├── mcp_server_routes.py   # GET/POST/DELETE /api/mcp/servers
│   ├── tool_routes.py         # GET /api/tools, PUT /api/tools/{id}/toggle
│   ├── user_routes.py         # GET/DELETE /api/users/memories
│   ├── admin_routes.py        # GET /admin/* — 7 cost analytics endpoints (admin-only)
│   ├── oauth_routes.py        # Google OAuth flow endpoints
│   └── auth_status_routes.py  # GET /api/auth/status
│
├── controllers/               # Request parsing + delegation to services
│   ├── auth_controller.py     # JWT creation, cookie setting, user lookup
│   └── chat_controller.py     # Cloudinary upload handling, SSE response wrapping
│
├── services/                  # Business logic (the heavy lifting)
│   ├── chat_service.py        # Main streaming orchestrator (8-step pipeline)
│   ├── history_service.py     # Redis-cached conversation history
│   ├── prompt_builder.py      # Dynamic system prompt assembly
│   ├── memory_service.py      # Long-term memory extraction (Google ADK-style)
│   └── ingestion_job_service.py # Redis-backed job queue for PDF ingestion
│
├── graph/                     # LangGraph Chat Graph
│   ├── builder.py             # Assembles + compiles the StateGraph
│   ├── router.py              # Conditional edge: text vs. tool call
│   ├── llm_registry.py        # Singleton cache of Gemini LLM instances
│   ├── nodes/
│   │   ├── common.py          # ChatState TypedDict schema
│   │   ├── setup_node.py      # Loads history + memories into state
│   │   ├── native_tool_node.py # Executes Python tools (with hooks + timer)
│   │   └── mcp_tool_node.py   # Proxies calls to MCP servers (with hooks + timer)
│
├── rag/                       # Agentic RAG System
│   ├── ingestion_service.py   # Orchestrates document → chunk → embed pipeline
│   ├── graph/
│   │   ├── workflow.py        # RAGWorkflow: the LangGraph RAG state machine
│   │   ├── state.py           # RAGGraphState TypedDict
│   │   └── nodes/
│   │       ├── retrieval_node.py    # parallel_retrieve_node (Qdrant + Tavily)
│   │       ├── grader_node.py       # LLM-based document relevance grader
│   │       ├── agent_node.py        # Reasoning + tool-use loop (AgentNode)
│   │       └── hallucination_node.py # Groundedness verification
│   ├── parsers/
│   │   └── llama_parse_client.py   # LlamaParse API wrapper (advanced PDF parsing)
│   ├── tools/
│   │   └── retrieval_tool.py       # search_knowledge_base @tool (used by AgentNode)
│   └── vector_store/
│       └── qdrant_manager.py       # Qdrant CRUD: upsert, search, delete, index
│
├── core/                      # Infrastructure & cross-cutting concerns
│   ├── database.py            # Motor (async MongoDB) client + collection references
│   ├── auth.py                # JWT encode/decode, password hashing (bcrypt)
│   ├── middleware.py          # get_current_user dependency (cookie + header auth)
│   ├── cache.py               # Async Redis pool + cache_set/get/delete with fallback
│   ├── limiter.py             # slowapi Limiter (Redis-backed or in-memory fallback)
│   ├── logging_config.py      # structlog JSON processor chain configuration
│   └── request_context.py     # CorrelationIdMiddleware (injects X-Request-ID)
│
├── tools/                     # Native tool definitions (@tool decorated functions)
│   ├── __init__.py            # AVAILABLE_TOOLS registry + get_all_tools()
│   ├── search_tool.py         # Tavily web search tool
│   ├── file_tool.py           # Local file read tool
│   └── ...                    # Other native tools (dice, time, weather, etc.)
│
├── utils/                     # Utilities
│   ├── mcp_connection_manager.py # Singleton MCPConnectionManager (SSE + HTTP + stdio)
│   └── hooks.py               # Decorator-based pre/post tool call hook system
│
└── models/                    # Pydantic schemas
    ├── user.py                # UserCreate, UserLogin, UserResponse (with is_admin field)
    └── ...

### Admin-specific files
```text
backend/
├── routes/admin_routes.py     # 7 analytics endpoints (GET /admin/*)
│                              # Protected by require_admin() dependency
│                              # Aggregates cost/token data from MongoDB
└── create_admin.py            # One-time script to promote a user to admin
```

---

## 4. Application Startup & Lifecycle

The FastAPI lifespan context manager (`main.py`) runs a structured initialization sequence every time the application starts:

```mermaid
sequenceDiagram
    participant U as Uvicorn
    participant APP as FastAPI Lifespan
    participant REDIS as Redis (Upstash)
    participant MONGO as MongoDB (Atlas)
    participant QDRANT as Qdrant Cloud
    participant LS as LangSmith

    U->>APP: startup
    APP->>REDIS: init_redis() — connect pool, ping
    Note over REDIS: Falls back to in-memory if unavailable
    APP->>MONGO: ensure_indexes() — messages, conversations, users
    APP->>MONGO: Register native tools (upsert to tools_collection)
    APP->>LS: Check LANGCHAIN_TRACING_V2 env var
    APP-->>U: yield (app is LIVE)
    Note over U: Handles all HTTP traffic
    U->>APP: shutdown signal
    APP->>APP: mcp_manager.disconnect_all()
    APP->>REDIS: close_redis()
```

### What happens on startup:

| Step | What it Does | Failure Behavior |
|------|-------------|-----------------|
| `init_redis()` | Creates async connection pool to Upstash/local Redis | **Non-fatal** — falls back to in-memory dict |
| `ensure_indexes()` | Creates MongoDB indexes on `conversation_id`, `user_id`, `email` | **Non-fatal** — warns and continues |
| Tool Registration | Upserts all native tools into `tools_collection` | **Non-fatal** — warns and continues |
| LangSmith Check | Reads `LANGCHAIN_TRACING_V2` env var | Informational only |

### Middleware Stack (applied bottom-to-top by FastAPI):

```
Request enters →
  1. CorrelationIdMiddleware   — Injects X-Request-ID into every request
  2. LoggingMiddleware (ASGI)  — Records method, path, status, duration_ms
  3. SlowAPIMiddleware         — Rate-limit enforcement (Redis or in-memory)
  4. CORSMiddleware            — Validates Origin against ALLOWED_ORIGINS env var
  → Route Handler
```

---

## 5. API Routes Reference

### Standard Routes

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | ❌ | Root health ping |
| GET | `/health` | ❌ | Deep health check (MongoDB, Redis, Qdrant) |
| POST | `/auth/signup` | ❌ | Register new user, sets JWT cookie |
| POST | `/auth/login` | ❌ | Authenticate, sets JWT cookie |
| POST | `/auth/logout` | ❌ | Clears JWT cookie |
| GET | `/auth/me` | ✅ | Returns current user info (includes `is_admin`) |
| POST | `/chat/stream` | ✅ | **Main Chat SSE endpoint** (rate-limited: 20/min) |
| GET | `/conversations` | ✅ | List user's conversations |
| GET | `/conversations/{id}/messages` | ✅ | Get messages in a conversation |
| DELETE | `/conversations/{id}` | ✅ | Delete a conversation + its messages |
| POST | `/api/v1/ingest/upload` | ✅ | Upload PDF for ingestion (rate-limited: 5/min) |
| GET | `/api/v1/ingest/job/{job_id}` | ✅ | Poll ingestion job status |
| POST | `/api/v1/rag/chat` | ✅ | Agentic RAG chat endpoint |
| POST | `/api/v1/rag/retrieve` | ✅ | Raw vector retrieval (debugging) |
| GET | `/api/v1/rag/files` | ✅ | List ingested files for current user |
| DELETE | `/api/v1/rag/file/{file_id}` | ✅ | Delete an ingested file from Qdrant |
| GET | `/api/mcp/servers` | ✅ | List registered MCP servers |
| POST | `/api/mcp/servers` | ✅ | Register a new MCP server |
| DELETE | `/api/mcp/servers/{id}` | ✅ | Unregister an MCP server |
| GET | `/api/tools` | ✅ | List all available tools |
| PUT | `/api/tools/{id}/toggle` | ✅ | Enable/disable a native tool |
| GET | `/api/users/memories` | ✅ | Get user's persistent memories |
| DELETE | `/api/users/memories` | ✅ | Clear all user memories |

### Admin Routes (`is_admin: true` required)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/admin/overview` | Platform-wide totals: users, messages, tokens, cost |
| GET | `/admin/users` | Paginated user list with per-user token & cost aggregates |
| GET | `/admin/users/{user_id}/sessions` | All conversations for a user with per-session cost |
| GET | `/admin/users/{user_id}/sessions/{conv_id}` | Turn-by-turn detail with per-message token counts |
| GET | `/admin/usage/daily` | AI response volume + cost per day (last 30 days) |
| GET | `/admin/usage/models` | Token & cost breakdown per Gemini model variant |
| GET | `/admin/usage/tools` | Tool call frequency ranking |

> ⚠️ Admin routes return **HTTP 403** for non-admin users. Promote a user via MongoDB:
> ```js
> db.users.updateOne({ email: "you@example.com" }, { $set: { is_admin: true } })
> ```

---

## 6. The Chat Pipeline (Agentic Conversation Flow)

Every message sent to `POST /chat/stream` triggers an 8-step pipeline inside `ChatService.stream()`. The endpoint returns a **Server-Sent Events (SSE)** stream — the AI's response is streamed token-by-token to the client in real time.

```mermaid
sequenceDiagram
    participant FE as Frontend
    participant CC as ChatController
    participant CS as ChatService
    participant HS as HistoryService
    participant PB as PromptBuilder
    participant MS as MemoryService
    participant CG as Chat Graph (LangGraph)
    participant GEMINI as Gemini LLM
    participant MONGO as MongoDB
    participant REDIS as Redis

    FE->>CC: POST /chat/stream {message, model, tools...}
    CC->>CS: ChatService.stream(...)

    CS->>MONGO: Step 1: Upsert conversation (create if new)
    CS->>MONGO: Step 2: Insert user message
    CS->>CS: Step 3: Connect requested MCP servers
    CS->>REDIS: Step 4: HistoryService.get_history() (cache lookup)
    REDIS-->>CS: HIT: return cached history
    MONGO-->>CS: MISS: query MongoDB, cache in Redis
    CS->>MONGO: Step 4b: Fetch user memories
    CS->>PB: Step 5: Build system prompt (tools + memories + MCP context)
    CS->>CG: Step 6: astream_events(graph_input)
    
    loop LangGraph Agent Loop
        CG->>GEMINI: ainvoke(messages)
        GEMINI-->>CG: AI response (text or tool_call)
        alt Tool Call Requested
            CG->>CG: native_tool_node / mcp_tool_node
            CG->>CS: on_tool_start SSE event
            CG->>CS: on_tool_end SSE event
            CG->>GEMINI: Re-invoke with tool result
        else Text Response
            CG->>CS: on_chat_model_stream SSE events
            CS-->>FE: data chunk... streamed
        end
    end

    CS->>MONGO: Step 8a: Save AI response + tool_steps + input_tokens + output_tokens + cost_usd + model
    CS->>MS: Step 8b: asyncio.create_task(extract_and_store) — NON-BLOCKING
    CS->>REDIS: Step 8c: Invalidate history cache
    CS-->>FE: data done true, conversation_id
```

### SSE Event Format

The frontend receives a stream of JSON-encoded SSE events:

| Event | Payload | Meaning |
|-------|---------|---------|
| `chunk` | `{"chunk": "Hello"}` | A token of the LLM's text response |
| `status` | `{"status": "Using tool: tavily_search"}` | LLM is calling a tool |
| `tool_call` | `{"tool_call": {"name": "...", "args": {...}}}` | Tool invocation details |
| `tool_output` | `{"tool_output": {"name": "...", "result": "..."}}` | Tool execution result |
| `done` | `{"done": true, "conversation_id": "..."}` | Stream complete |
| `error` | `{"error": "..."}` | An error occurred |

### The LangGraph Chat State Machine

```mermaid
stateDiagram-v2
    [*] --> setup_node
    setup_node --> chat_model: history + memories loaded
    chat_model --> route_tools: LLM response received

    state route_tools {
        [*] --> check_response
        check_response --> has_text: text only
        check_response --> has_native_tool: native tool call
        check_response --> has_mcp_tool: MCP tool call
    }

    route_tools --> END: has_text
    route_tools --> native_tool_node: has_native_tool
    route_tools --> mcp_tool_node: has_mcp_tool

    native_tool_node --> chat_model: tool result appended
    mcp_tool_node --> chat_model: tool result appended
```

---

## 7. The Agentic RAG Pipeline

The RAG system is a separate, self-contained LangGraph workflow designed to handle document-grounded Q&A. It lives in `rag/graph/` and is called from `POST /api/v1/rag/chat`.

### The 5-Node RAG State Machine

```mermaid
stateDiagram-v2
    [*] --> parallel_retrieve
    
    parallel_retrieve --> grade_documents: Qdrant results + Tavily results merged
    
    grade_documents --> agent: All docs evaluated
    
    note right of grade_documents
        Each document scored:
        relevant / not relevant
        by gemini-2.5-flash-lite
    end note
    
    agent --> hallucination_check: Answer generated
    
    note right of agent
        Agentic loop:
        LLM can call search_knowledge_base
        tool multiple times to gather
        more context before answering
    end note
    
    hallucination_check --> END: Always returns answer
    
    note right of hallucination_check
        If grounded → hallucination_warning=False
        If not grounded → hallucination_warning=True
        (No retry to avoid rate-limit cascades)
    end note
```

### Step-by-Step RAG Flow

```mermaid
sequenceDiagram
    participant CLIENT as Client
    participant RR as RAG Route
    participant RW as RAG Workflow
    participant PRN as parallel_retrieve_node
    participant QDRANT as Qdrant
    participant TAVILY as Tavily Web
    participant GN as grade_documents
    participant AN as agent_node
    participant TOOL as search_knowledge_base Tool
    participant GEMINI as Gemini LLM
    participant HN as hallucination_node

    CLIENT->>RR: POST /api/v1/rag/chat question, selected_files
    RR->>RW: workflow.app.ainvoke question

    par Parallel Retrieval
        RW->>PRN: asyncio.gather(...)
        PRN->>QDRANT: similarity_search(question, k=5, filter=user_id)
        PRN->>TAVILY: search(question)
    end
    PRN->>RW: merged unique documents

    RW->>GN: grade all retrieved documents
    loop For each document
        GN->>GEMINI: "Is this relevant to the question? yes/no"
        GEMINI-->>GN: relevance score
    end
    GN->>RW: filtered relevant documents

    RW->>AN: generate answer with relevant docs
    loop Agent Reasoning (can loop)
        AN->>GEMINI: ainvoke(docs + question)
        GEMINI->>TOOL: search_knowledge_base(query="...")
        TOOL->>QDRANT: similarity_search(query, k=5)
        QDRANT-->>TOOL: chunks
        TOOL-->>GEMINI: results
        GEMINI-->>AN: final answer text
    end

    RW->>HN: check if answer is grounded
    HN->>GEMINI: "Is this answer supported by these docs? yes/no"
    GEMINI-->>HN: grounded / not grounded
    HN->>RW: final state (hallucination_warning: bool)

    RW-->>RR: generation, documents, hallucination_warning
    RR-->>CLIENT: answer, sources, hallucination_warning
```

### Ingestion Pipeline (Background Queue)

When a file is uploaded, ingestion is asynchronous — the API returns instantly with a `job_id`, and processing happens in a FastAPI `BackgroundTask`.

```mermaid
sequenceDiagram
    participant FE as Frontend
    participant UPR as /upload Route
    participant BG as BackgroundTask
    participant IJS as IngestionJobService
    participant IS as IngestionService
    participant LP as LlamaParse
    participant EMBED as Gemini Embeddings
    participant QDRANT as Qdrant

    FE->>UPR: POST /api/v1/ingest/upload (file)
    UPR->>IJS: create_job(filename, user_id) → job_id
    UPR->>BG: add_task(_run_ingestion_background)
    UPR-->>FE: 200 job_id, status queued -- INSTANT RESPONSE

    Note over BG: Running asynchronously in background

    BG->>IJS: update_job(status="parsing")
    BG->>IS: process_upload_from_path(file_path)
    IS->>LP: LlamaParse.aload_data(file) — Cloud API
    LP-->>IS: parsed Document objects (markdown)
    IS->>IS: Semantic Chunking (split by headers + size)
    BG->>IJS: update_job(status="embedding")
    IS->>EMBED: embed_documents(chunks)
    EMBED-->>IS: float32 vectors
    IS->>QDRANT: upsert(vectors + metadata)
    BG->>IJS: update_job(status="complete", file_id=..., chunks=N)

    FE->>UPR: GET /api/v1/ingest/job/{job_id} (polling)
    UPR->>IJS: get_job(job_id)
    IJS-->>UPR: status, chunks_count, file_id
    UPR-->>FE: job status
```

---

## 8. Model Context Protocol (MCP) Integration

AgentX implements the open-source [Model Context Protocol](https://modelcontextprotocol.io/) to connect Gemini to external services in real-time.

### How It Works

```mermaid
graph LR
    subgraph agentx [AgentX Backend]
        CM["chat_model_node\n(Gemini LLM)"]
        MCPMgr["MCPConnectionManager\n(Singleton)"]
        CACHE["Tool Cache\n(5-min TTL)"]
        MTN["mcp_tool_node"]
    end

    subgraph external [MCP Servers External]
        GD["Google Drive MCP Server\n(SSE/HTTP)"]
        FS["File System MCP Server\n(stdio)"]
        CUSTOM["Custom MCP Server\n(HTTP/SSE)"]
    end

    CM -->|"get_all_langchain_tools()"| MCPMgr
    MCPMgr --> CACHE
    MCPMgr -->|"MultiServerMCPClient"| GD
    MCPMgr -->|"MultiServerMCPClient"| FS
    MCPMgr -->|"MultiServerMCPClient"| CUSTOM
    CM -->|"StructuredTool call"| MTN
    MTN -->|"Proxy tool execution"| MCPMgr
    MCPMgr -->|"Execute on server"| GD
```

### MCP Architecture Details

- **`MCPConnectionManager`** is a **singleton** (`__new__` pattern) — only one instance exists per process.
- **Tool Discovery Caching**: `get_all_langchain_tools()` caches discovered tools for **5 minutes** per server URL, preventing expensive SSE network calls on every chat turn.
- **Adapter Pattern**: Raw MCP tool definitions are converted into **LangChain `StructuredTool` objects** so the LLM can invoke them with full schema validation.
- **Transport Support**: Supports **HTTP**, **SSE**, and **stdio** transports, auto-detected from the URL format.
- **Lifecycle**: All connections are gracefully closed on application shutdown via `mcp_manager.disconnect_all()`.

---

## 9. Persistent Memory Bank

Inspired by Google ADK's Memory Bank pattern, AgentX silently extracts and stores facts about users to personalize future sessions.

```mermaid
sequenceDiagram
    participant CS as ChatService
    participant MS as MemoryService
    participant GEMINI as gemini-2.5-flash-lite
    participant MONGO as MongoDB (user_memories)
    participant PB as PromptBuilder

    Note over CS: After each AI response is saved...
    CS->>MS: asyncio.create_task(extract_and_store(...))
    Note over CS: NON-BLOCKING — chat stream continues

    MS->>GEMINI: Extract durable facts from this conversation...
    GEMINI-->>MS: topic tech stack, content Uses FastAPI
    MS->>MONGO: upsert user_id, memories -- merge with existing

    Note over PB: At the START of every chat turn...
    CS->>MONGO: MemoryService.get_user_memories(user_id)
    MONGO-->>CS: topic, content...
    CS->>PB: assemble user_memories
    PB->>PB: Inject memories into System Prompt
```

### Memory Schema (MongoDB `user_memories` collection)
```json
{
  "user_id": "64abc123...",
  "memories": [
    {"topic": "tech stack", "content": "Uses Python, FastAPI, and MongoDB", "created_at": "..."},
    {"topic": "project", "content": "Building an AI agent platform called AgentX", "updated_at": "..."}
  ],
  "updated_at": "2026-01-14T12:00:00Z"
}
```

**Key Design Decisions:**
- **Why MongoDB, not Redis?** Memories are permanent — they must survive Redis flushes or server restarts.
- **Why LLM extraction?** Rule-based extraction misses implicit facts. The LLM understands context and nuance.
- **Why cap at 10 memories?** More than 10 makes the system prompt too long and dilutes the LLM's attention.
- **Why `gemini-2.5-flash-lite`?** Cheapest model, fastest, and the extraction task is simple enough for it.

---

## 10. Redis: Caching, Rate-Limiting & Background Jobs

Redis (Upstash) serves three critical functions:

### 10.1 Conversation History Cache (`HistoryService`)
```mermaid
flowchart TD
    A["ChatService requests history"] --> B{Redis cache hit?}
    B -->|HIT| C["Return cached list[BaseMessage]"]
    B -->|MISS| D["Query MongoDB messages_collection"]
    D --> E["Format as LangChain messages"]
    E --> F["cache_set(key, messages, ttl=300s)"]
    F --> C
    G["After AI responds"] --> H["HistoryService.invalidate(conversation_id)"]
    H --> I["cache_delete(key)"]
```

- Cache key: `history:{conversation_id}:{user_id}`
- TTL: **5 minutes** (300 seconds)
- Invalidated immediately after each conversation turn completes

### 10.2 Rate Limiting (`slowapi`)
- `POST /chat/stream`: **20 requests per minute** per IP
- `POST /api/v1/ingest/upload`: **5 requests per minute** per IP
- Counts are stored in Redis atomically; falls back to in-memory on Redis failure

### 10.3 Ingestion Job Queue (`IngestionJobService`)
- Job state stored as JSON with a **24-hour TTL**
- States: `queued → parsing → embedding → complete | failed`
- Frontend polls `GET /api/v1/ingest/job/{job_id}` to check progress

### 10.4 OAuth State (`oauth_controller.py`)
- OAuth CSRF state tokens stored in Redis with a **10-minute TTL**
- Prevents broken OAuth flows when Render auto-scales across multiple instances

### Graceful Degradation
If Redis is unavailable (e.g., local dev without Docker):
- `core/cache.py` falls back to an in-memory Python `dict` (`_fallback_cache`)
- `core/limiter.py` pings Redis on startup; if it fails, uses `slowapi`'s in-memory backend
- The application starts and works fully — only distributed features (multi-instance rate-limiting) are degraded

---

## 11. Authentication & Security

```mermaid
sequenceDiagram
    participant FE as Frontend
    participant AUTH as /auth/login
    participant MW as get_current_user Dependency
    participant MONGO as MongoDB

    FE->>AUTH: POST email, password
    AUTH->>MONGO: find_one email
    MONGO-->>AUTH: user document
    AUTH->>AUTH: bcrypt.verify(password, hash)
    AUTH->>AUTH: create_access_token user_id, email
    AUTH-->>FE: 200 + Set-Cookie access_token=JWT HttpOnly SameSite=Lax

    Note over FE: Cookie auto-sent on all subsequent requests

    FE->>MW: Any protected route
    MW->>MW: request.cookies.get("access_token")
    Note over MW: Falls back to Authorization: Bearer header
    MW->>MW: jwt.decode(token, SECRET_KEY)
    MW->>MONGO: find_one({_id: user_id})
    MONGO-->>MW: user document
    MW->>MW: Inject user into route handler
```

**Security Details:**
- **JWT** signed with `HS256` and `JWT_SECRET_KEY`; stored in an **`HttpOnly` cookie**
- `secure=True` + `samesite="none"` in **production** (HTTPS only)
- `secure=False` + `samesite="lax"` in **development** (HTTP localhost)
- Password hashing: **bcrypt** via `passlib`
- The `ENVIRONMENT` env var controls which cookie mode is active

---

## 12. Dual-Layer Observability: LangSmith + Native Cost Analytics

AgentX has **two complementary observability layers** that work simultaneously — each covering what the other cannot:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Every Chat Request                               │
├───────────────────────┬─────────────────────────────────────────────────┤
│   LangSmith Tracing   │        Native Token + Cost Tracking             │
│   (External SaaS)     │        (Stored in YOUR MongoDB)                 │
├───────────────────────┼─────────────────────────────────────────────────┤
│ ✅ Visual graph trace │ ✅ Per-message token counts (exact)              │
│ ✅ Full prompt text   │ ✅ Per-turn USD cost                             │
│ ✅ Latency per node   │ ✅ Per-user aggregate spend                      │
│ ✅ Debugging / replay │ ✅ Per-session aggregate spend                   │
│ ❌ No per-user cost   │ ✅ Admin dashboard with drill-down               │
│ ❌ SaaS dependency    │ ✅ No external dependency                        │
│ ❌ Data leaves system │ ✅ Data stays in your MongoDB                    │
└───────────────────────┴─────────────────────────────────────────────────┘
```

### Layer 1 — LangSmith (External Tracing)

When `LANGCHAIN_TRACING_V2=true`, every LangGraph run is automatically traced:
- Visual graph of node execution order and timing
- Full prompts sent to Gemini (tokens, temperature, model)
- Tool call inputs and outputs
- Hallucination check results
- Token cost per run

Enable with: `LANGCHAIN_API_KEY=lsv2_...` and `LANGCHAIN_PROJECT="AgentX"`

### Layer 2 — Native Cost Tracking (Built-In)

This is the **primary cost accounting system** — it captures real token usage directly from the Gemini API response and stores it permanently in MongoDB.

#### How it works (the technical detail)

The key insight is that `astream_events` must be called with **`version="v2"`**. With `v1` (deprecated), events from LLM calls *inside* LangGraph nodes are swallowed and never surface. With `v2`, they bubble up correctly with `metadata.langgraph_node` populated:

```python
# In chat_service.py — the exact pattern used:
async for event in chat_graph.astream_events(graph_input, version="v2", config=config):
    event_type = event.get("event")
    # v2 gives us the node name — filter to avoid double-counting on tool loops
    node_name  = event.get("metadata", {}).get("langgraph_node", "")

    elif event_type == "on_chat_model_end" and node_name == "chat_model":
        usage = event["data"]["output"].usage_metadata
        # usage = {"input_tokens": 523, "output_tokens": 148, "total_tokens": 671}
        total_input_tokens  += usage.get("input_tokens", 0)
        total_output_tokens += usage.get("output_tokens", 0)
```

> **Why `node_name == "chat_model"` filter?**
> When the LLM loops back after a tool call, `on_chat_model_end` fires multiple times.
> Without the node filter, tokens would be double-counted per tool round-trip.

#### Pricing constants (Gemini 2.5 Flash)

```python
INPUT_PRICE_PER_TOKEN  = 0.075 / 1_000_000   # $0.075 per 1M input tokens
OUTPUT_PRICE_PER_TOKEN = 0.30  / 1_000_000   # $0.30  per 1M output tokens
cost_usd = total_input_tokens * INPUT_PRICE_PER_TOKEN + total_output_tokens * OUTPUT_PRICE_PER_TOKEN
```

#### Saved to MongoDB on every AI response

```json
{
  "role": "model",
  "content": "...",
  "model": "gemini-2.5-flash",
  "input_tokens": 523,
  "output_tokens": 148,
  "cost_usd": 0.00005393,
  "tool_steps": [...],
  "timestamp": "2026-05-31T..."
}
```

### Structlog (Per-Request Correlation IDs)

`CorrelationIdMiddleware` injects a unique `X-Request-ID` UUID into every request. All downstream logs — even from deep inside LangGraph nodes — automatically carry this ID.

```
2026-05-31T09:57:10Z [info] http.request method=POST path=/chat/stream
    request_id=bca8b9ea status=200 duration_ms=3241.0
2026-05-31T09:57:10Z [info] token.usage user_id=64abc... conversation_id=...
    model=gemini-2.5-flash input_tokens=523 output_tokens=148
```

---

## 13. Admin Analytics Dashboard

The admin system gives platform operators a full **drill-down cost analytics dashboard** with data sourced from the native tracking layer (Layer 2 above).

### Access Control

All `/admin/*` routes are protected by the `require_admin()` FastAPI dependency:

```python
async def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if not current_user.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required.")
    return current_user
```

The `is_admin` field lives on the `users` document. Promote a user once via MongoDB:
```js
db.users.updateOne({ email: "you@example.com" }, { $set: { is_admin: true } })
```
Or run the helper script: `python create_admin.py`

### Dashboard Drill-Down Hierarchy

```
/admin                        ← Platform overview cards + charts
  └── /admin/users            ← All users table (cost per user)
        └── /admin/users/{id}/sessions        ← All sessions for one user
              └── /admin/users/{id}/sessions/{conv_id}  ← Turn-by-turn breakdown
```

### Drill-Down Example

```mermaid
flowchart TD
    A["🖥️ /admin\nOverview: 42 users · $0.0012 total"]
    A --> B["👤 /admin/users\nAll users table\nName · Sessions · Input Tokens · Cost"]
    B --> C["🗂️ /admin/users/:id/sessions\nAll sessions for Alice\nTitle · Date · Turns · Cost"]
    C --> D["💬 /admin/users/:id/sessions/:convId\nTurn-by-turn detail\nUser msg | AI msg (523 in · 148 out · $0.000054)"]
```

### Available Endpoints & What They Return

| Endpoint | MongoDB Aggregation | Returns |
|---|---|---|
| `GET /admin/overview` | `$group` over all model messages | Total users, messages, tokens, cost |
| `GET /admin/users` | Per-user `$group` + conv count | Name, email, sessions, AI turns, cost |
| `GET /admin/users/{id}/sessions` | Per-conversation `$group` | Title, turns, input/output tokens, cost |
| `GET /admin/users/{id}/sessions/{cid}` | Raw find + sort | All messages with token fields |
| `GET /admin/usage/daily` | `$dayOfMonth` group (30 days) | Messages + cost per day |
| `GET /admin/usage/models` | `$group` by `model` field | Count, cost, tokens per model |
| `GET /admin/usage/tools` | `$unwind tool_steps` + `$group` | Top tools by call count |

### Frontend Pages

| Route | Component | Purpose |
|---|---|---|
| `/admin` | `AdminDashboard.jsx` | Overview cards, line chart, bar charts, users table |
| `/admin/users/:userId` | `AdminUserPage.jsx` | Session list with cost per session |
| `/admin/users/:userId/sessions/:convId` | `AdminSessionPage.jsx` | Turn-by-turn detail with expandable messages |

All admin frontend routes are wrapped in `<AdminRoute>` which checks `user.is_admin` from React context and redirects non-admins to `/chat`.

---

## 14. Hook System (Middleware for Tool Calls)

`utils/hooks.py` implements a **decorator-based pre/post hook system** for all tool calls — both native and MCP.

```python
# Example hook registration
@register_pre_tool_hook("tavily_search")
async def log_web_search(tool_name, args):
    logger.info(f"About to search web: {args['query']}")
    # return {"deny": True} to block the call
    # return {"modify": True, "args": {...}} to mutate arguments

@register_post_tool_hook("tavily_search")
async def audit_search_result(tool_name, result):
    metrics.record(tool_name, result)
```

Every tool execution is also wrapped in a **`ToolTimer`** context manager that measures latency to the millisecond and logs it automatically.

---

## 15. Database Schema (MongoDB)

### `users` collection
```json
{
  "_id": "ObjectId",
  "email": "user@example.com",       // unique index
  "name": "John Doe",
  "password": "$2b$12$...",           // bcrypt hash
  "is_admin": false,                  // true → access to /admin/* routes
  "created_at": "ISODate",
  "updated_at": "ISODate"
}
```

### `conversations` collection
```json
{
  "_id": "ObjectId",
  "user_id": "string",                // indexed with updated_at
  "title": "First 50 chars of message",
  "mcp_server_url": "string | null",
  "created_at": "ISODate",
  "updated_at": "ISODate"             // indexed DESC for sorting
}
```

### `messages` collection
```json
{
  "_id": "ObjectId",
  "conversation_id": "string",        // compound index with user_id + timestamp
  "user_id": "string",
  "role": "user | model",
  "content": "string",
  "attachments": ["cloudinary_url"],  // optional media
  "tool_steps": [                     // only on model messages
    {"name": "tavily_search", "args": {...}, "result": "...", "status": "completed"}
  ],
  // ── Cost tracking fields (model messages only) ──
  "model": "gemini-2.5-flash",        // which model generated this response
  "input_tokens": 523,                // exact count from Gemini usage_metadata
  "output_tokens": 148,               // exact count from Gemini usage_metadata
  "cost_usd": 0.00005393,             // calculated: tokens × per-token price
  "timestamp": "ISODate"
}
```

> **How token counts are captured**: The `on_chat_model_end` event from `astream_events(version="v2")` exposes `output.usage_metadata`. We filter by `metadata.langgraph_node == "chat_model"` to avoid double-counting on tool-call loops.

### `tools` collection
```json
{
  "_id": "ObjectId",
  "tool_id": "tavily_search",          // unique index
  "name": "tavily_search",
  "description": "...",
  "category": "search",
  "requires_auth": false,
  "is_enabled": true,
  "updated_at": "ISODate"
}
```

### `user_memories` collection
```json
{
  "_id": "ObjectId",
  "user_id": "string",                // unique
  "memories": [
    {"topic": "tech stack", "content": "...", "created_at": "...", "updated_at": "..."}
  ],
  "updated_at": "ISODate"
}
```

---

## 16. Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | ✅ | Gemini API key (also used for embeddings) |
| `MONGO_URI` | ✅ | MongoDB Atlas connection string |
| `JWT_SECRET_KEY` | ✅ | Secret for signing JWTs |
| `REDIS_URL` | ✅ | `rediss://default:pass@host:6379` (Upstash) |
| `QDRANT_URL` | ✅ | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | ✅ | Qdrant API key |
| `QDRANT_COLLECTION` | ✅ | Vector collection name (default: `agentic_rag_v1`) |
| `LLAMA_CLOUD_API_KEY` | ✅ | LlamaParse API key for PDF parsing |
| `TAVILY_API_KEY` | ⚠️ Optional | Web search (skipped if missing) |
| `CLOUDINARY_CLOUD_NAME` | ⚠️ Optional | For image upload support |
| `CLOUDINARY_API_KEY` | ⚠️ Optional | Cloudinary API key |
| `CLOUDINARY_API_SECRET` | ⚠️ Optional | Cloudinary API secret |
| `CLIENT_ID` | ⚠️ Optional | Google OAuth client ID |
| `CLIENT_SECRET` | ⚠️ Optional | Google OAuth client secret |
| `LANGCHAIN_TRACING_V2` | ⚠️ Optional | `"true"` to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | ⚠️ Optional | LangSmith API key |
| `LANGCHAIN_PROJECT` | ⚠️ Optional | LangSmith project name |
| `ENVIRONMENT` | ⚠️ Optional | `"production"` to enable secure cookies |
| `ALLOWED_ORIGINS` | ⚠️ Optional | Comma-separated CORS origins |
| `BACKEND_URL` | ⚠️ Optional | Backend URL (for OAuth callbacks) |
| `FRONTEND_URL` | ⚠️ Optional | Frontend URL (for OAuth redirects) |
| `CHAT_RATE_LIMIT` | ⚠️ Optional | Requests/minute for `/chat/stream` (default: 20) |
| `UPLOAD_RATE_LIMIT` | ⚠️ Optional | Requests/minute for `/ingest/upload` (default: 5) |
| `PORT` | ⚠️ Optional | Server port (default: 8000) |

---

## 17. Running Locally & E2E Testing

### Start the backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
cp .env.sample .env            # Fill in your keys

uvicorn main:app --reload --port 8000
```

### Verify health
```bash
curl http://localhost:8000/health
# {"status": "healthy", "checks": {"mongodb": "ok", "redis": "ok", "qdrant": "ok"}}
```

### Run the full E2E test suite (22 tests)
```bash
$env:PYTHONIOENCODING="utf8"  # Windows PowerShell
python run_e2e_fix.py
```

The suite validates: Auth (signup/login/logout) → Conversation CRUD → SSE Streaming Chat → Tool Call → PDF Upload & Ingestion → RAG Retrieve → Agentic Chat → Memory Cleanup.

**Expected result:**
```
Total 22 | Pass 22 | Fail 0 | Error 0
```

### Interactive API Docs
Visit: `http://localhost:8000/docs` (Swagger UI)
