# Open RAG From Scratch

A from-scratch walkthrough of retrieval-augmented generation (RAG), built with open, self-hostable tools and run entirely on your own machine.

```
INDEX (offline)     load → split → embed → store
RETRIEVE (query)    embed query → similarity search → top-k chunks
GENERATE (query)    prompt template + context → LLM → answer
```

## Read the guide

The full theory guide (Parts 1–4, with diagrams) is published as a website:

**→ https://itsmusa.github.io/open-rag/**

## What's in this repo

| What | Where |
|---|---|
| The guide (website source) | [`docs/`](docs/) |
| Images used by the guide | [`docs/assets/`](docs/assets/) |
| The runnable implementation | [`notebooks/`](notebooks/) |
| Dependencies | [`requirements.txt`](requirements.txt) |

## Setup

Requires Python 3.10+ and [Ollama](https://ollama.com) (the local model runtime, installed separately).

```bash
git clone https://github.com/itsmusa/open-rag.git
cd open-rag

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
ollama pull gpt-oss:20b
```

## The stack

Every component is open and runs on your own machine.

| Stage | Tool | License |
|---|---|---|
| Fetch a page | `requests` + `trafilatura` | Apache 2.0 |
| Count tokens | HuggingFace `tokenizers` | Apache 2.0 |
| Split text | `chonkie` | MIT |
| Embed text | `sentence-transformers` + Qwen3-Embedding-0.6B | Apache 2.0 |
| Store vectors | `chromadb` | Apache 2.0 |
| Generate | `ollama` + `gpt-oss:20b` | Apache 2.0 |
| Prompt templating | f-string / Jinja2 | BSD-3 |

## Credits

Based on the *RAG From Scratch* video series and repo by LangChain:

- Video: <https://www.youtube.com/watch?v=sVcwVQRHIc8&t=1245s>
- Repo: <https://github.com/langchain-ai/rag-from-scratch>

## License

MIT — see [LICENSE](LICENSE).
