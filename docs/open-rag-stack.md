# Fully Open RAG Stack

Notes for building the indexing → retrieval → generation pipeline with OSI-approved, self-hostable components only.

## Pipeline

```
INDEX (offline)     load → split → embed → store
RETRIEVE (query)    embed query → similarity search → top-k chunks
GENERATE (query)    prompt template + context → LLM → answer
```

## Recommended stack

| Stage | Tool | License |
|---|---|---|
| HTTP / text extraction | `requests` + `trafilatura` | Apache 2.0 |
| Chunking | `chonkie` | MIT |
| Token counting | HuggingFace `tokenizers` | Apache 2.0 |
| Embeddings | `sentence-transformers` + Qwen3-Embedding-0.6B | Apache 2.0 |
| Vector store | `chromadb` | Apache 2.0 |
| LLM runtime | `ollama` | MIT |
| LLM | Qwen3.5-9B / Qwen3.6-27B | Apache 2.0 |
| Prompt templating | f-string or Jinja2 | BSD-3 |

```bash
pip install requests trafilatura chonkie sentence-transformers chromadb ollama jinja2
```

## Alternatives by stage

### Chunking
| Tool | Notes | License |
|---|---|---|
| chonkie | TokenChunker, RecursiveChunker, SemanticChunker, SDPMChunker, CodeChunker, TableChunker. ~15 MB install | MIT |
| unstructured | Heavy, many formats, many optional deps | Apache 2.0 |
| custom | Recursive splitter is ~30 lines | — |

### Embeddings
| Model | Notes | License |
|---|---|---|
| Qwen3-Embedding (0.6B / 4B / 8B) | Current top open model on MTEB | Apache 2.0 |
| BGE-M3 | Multilingual, long context, well-tested | MIT |
| Nomic Embed v1.5 | 8k context, small | Apache 2.0 |
| EmbeddingGemma-300M | Verify license — reports conflict | Gemma Terms? |

Runtimes: `sentence-transformers` (Apache 2.0), `fastembed` (ONNX, lightweight, Apache 2.0), `model2vec` (static embeddings, MIT), HuggingFace TEI (Apache 2.0), Infinity (MIT).

### Vector store
| Tool | License |
|---|---|
| Chroma | Apache 2.0 |
| Qdrant | Apache 2.0 |
| FAISS | MIT |
| LanceDB | Apache 2.0 |
| sqlite-vec | MIT / Apache 2.0 |
| pgvector | PostgreSQL License |
| Weaviate | BSD-3 |
| Milvus | Apache 2.0 |

### LLM runtime
| Tool | Use case | License |
|---|---|---|
| Ollama | Local dev, one command, OpenAI-compatible API | MIT |
| llama.cpp | CPU / edge / unusual hardware, GGUF | MIT |
| vLLM | Production, high concurrency (PagedAttention + continuous batching) | Apache 2.0 |
| TGI | HF-native serving | Apache 2.0 |

### LLM
| Model | Notes | License |
|---|---|---|
| Qwen3.5-9B | Runs from ~6 GB | Apache 2.0 |
| Qwen3.6-27B | Single 24 GB GPU | Apache 2.0 |
| Mistral Small 4 | MoE, 6.5B active | Apache 2.0 |
| DeepSeek V4-Flash | 284B MoE, 13B active — needs a large box | MIT |
| GPT-oss 20B | Fits ~11 GB | Apache 2.0 |
| Phi-4 14B | Fits ~8 GB | MIT |
| Hunyuan Hy3 | 295B MoE, 21B active | Apache 2.0 |
| GLM-5.2 | 753B MoE | MIT |

### Observability (LangSmith replacement)
| Tool | Notes | License |
|---|---|---|
| Langfuse | Closest analog. Self-host needs ClickHouse, Postgres, Redis (5+ services). SSO/RBAC/advanced evals gated | MIT core |
| MLflow | No enterprise paywalls, Apache/Linux Foundation | Apache 2.0 |
| TruLens | OpenTelemetry-native, provider/framework packages opt-in | MIT |
| OpenLIT | OpenTelemetry-based | Apache 2.0 |
| LangWatch | Testing + tracing | Apache 2.0 |
| OpenTelemetry + Jaeger | Roll your own | Apache 2.0 |

### Evaluation
| Tool | Notes | License |
|---|---|---|
| DeepEval | Framework-agnostic, has its own test-case model | Apache 2.0 |
| TruLens | RAG triad metrics, from-scratch walkthrough | MIT |
| Ragas | **Pulls in `langchain-core`; API is LangChain-shaped** | Apache 2.0 |

### Framework (optional — not required)
| Framework | Notes | License |
|---|---|---|
| Haystack | Cleanest non-LangChain pipeline/component model | Apache 2.0 |
| LlamaIndex | Library is MIT; hosted service optional | MIT |
| txtai | Embeddings DB + pipeline | Apache 2.0 |
| DSPy | Prompt/program optimization, not orchestration | MIT |
| RAGatouille | ColBERT late-interaction retrieval | Apache 2.0 |

## Component API equivalents

| Operation | Call |
|---|---|
| Load page | `trafilatura.extract(requests.get(url).text)` |
| Split | `RecursiveChunker(chunk_size=...)` from chonkie |
| Embed docs | `SentenceTransformer("Qwen/Qwen3-Embedding-0.6B").encode(texts)` |
| Store | `client = chromadb.PersistentClient(...)` → `collection.add(ids, documents, embeddings, metadatas)` |
| Retrieve | `collection.query(query_embeddings=[...], n_results=k)` |
| Prompt | f-string / `jinja2.Template` |
| Generate | `ollama.chat(model="qwen3.5:9b", messages=[...])` |
| OpenAI-compatible client against local runtime | `OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")` |
| Full chain | one plain function `def rag(question) -> str:` |

## Licensing traps

**"Open weights" ≠ open source.** Excluded from a strictly-open build:

| Model family | License | Problem |
|---|---|---|
| Llama (all versions) | Llama Community License | Custom, source-available, not OSI |
| Gemma | Gemma Terms | Custom, not OSI |
| Command R+ | CC-BY-NC 4.0 | Non-commercial only |
| Kimi K2.x | Modified MIT | Custom attribution terms |
| Nemotron | OpenMDW-1.1 | Custom |
| MiniMax | MiniMax Community | Custom |
| Qwen 72B / large tiers | Qwen License | Only the smaller sizes are Apache 2.0 |

**Soft-open tools:**
- Arize Phoenix — Elastic License 2.0 (source-available, not OSI)
- LM Studio — free but proprietary
- Langfuse — MIT core, paid tier gates SSO/RBAC/advanced evals
- Databricks / Falcon — bespoke licenses with acceptable-use caps

**Hosted "open model" APIs** (DeepSeek, Together, Fireworks, OpenRouter) serve MIT/Apache weights but are still commercial dependencies. Self-host or run locally if the goal is no vendor at all.

## Transitive dependency trap

Avoiding a dependency in your code does not avoid it in your dependency tree.

- `ragas` → `langchain-core`
- `langchain-text-splitters` ships as its own PyPI package; uninstalling `langchain` does not remove it
- CrewAI, gpt-researcher, embedchain → heavy LangChain deps
- Some `unstructured` and `docling` extras pull LangChain adapters

Audit with:
```bash
pipdeptree
uv tree
pip list | Select-String langchain
```

## Build gotchas

1. **Re-embed everything when changing models.** Vector dimensions differ (OpenAI 1536 vs Qwen3-Embedding 1024/2560). Old collections are unusable.
2. **Use the target model's tokenizer.** `cl100k_base` is calibrated for OpenAI models; counting chunks with it mis-sizes context for Qwen/Phi/DeepSeek.
3. **`temperature=0` is not fully deterministic** on most open models. Don't assert exact string output in tests.
4. **Hardware:** Q4 8–9B ≈ 6–8 GB VRAM; 27B ≈ 16 GB; 70B+ needs multi-GPU or a large unified-memory box. Embedding models run acceptably on CPU.
5. **Context budget:** chunk size × k must fit the model's context window alongside the prompt. Smaller windows than frontier APIs are common — verify per model.
6. **Chunk sizing is token-based, not character-based**, once you remove the OpenAI tokenizer.
