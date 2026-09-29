# Open RAG From Scratch

Build a retrieval-augmented generation (RAG) system from the ground up, using only open, self-hostable tools.
No cloud account, no API keys, no LangChain — everything runs locally.

The whole idea fits on one line:

```
INDEX (offline)     load → split → embed → store
RETRIEVE (query)    embed query → similarity search → top-k chunks
GENERATE (query)    prompt template + context → LLM → answer
```

## What you get

| What | Where |
|---|---|
| Theory guide, Parts 1–4 | [`docs/theory.md`](docs/theory.md) |
| The open-stack reference (models, tools, licenses) | [`docs/open-rag-stack.md`](docs/open-rag-stack.md) |
| The runnable implementation | [`notebooks/open_rag_from_scratch_1_to_4.ipynb`](notebooks/open_rag_from_scratch_1_to_4.ipynb) |

## How to follow this

1. **Read the theory first** — [`docs/theory.md`](docs/theory.md) walks through the four parts in plain language,
   with diagrams and worked examples. No code required.
2. **Then run the notebook** — [`notebooks/open_rag_from_scratch_1_to_4.ipynb`](notebooks/open_rag_from_scratch_1_to_4.ipynb)
   builds the same chain the theory describes, one cell at a time. Each markdown cell explains one idea; the code
   cell below it does exactly that and prints the result. Run one cell, look at what it produced, read on.
3. **Refer back** — the notebook deliberately mirrors the theory's structure, so when a cell is confusing the
   matching theory section is one link away.

## Setup

Requires Python 3.10+ and [Ollama](https://ollama.com) (the local model runtime, installed separately).

```bash
git clone https://github.com/<your-account>/open-rag.git
cd open-rag

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

Then pull the generation model:

```bash
ollama pull gpt-oss:20b
```

Start the notebook with a local kernel (needed for the local `ollama` runtime) and run it top to bottom:

```bash
jupyter notebook notebooks/open_rag_from_scratch_1_to_4.ipynb
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
