# AI-Powered Intelligent Assistant

A Streamlit application that combines multi-turn chat, document question-answering, and task automation. Users talk to Google Gemini, optionally ground answers in an uploaded PDF or text file, and create or complete action items in natural language.

## Features

- **Multi-turn chat.** Conversation history is kept in the Streamlit session and sent with each request.
- **Document Q&A.** Upload a PDF or TXT file. The file is split into chunks, embedded, and searched with FAISS. The top matching passages are added to the prompt before Gemini answers.
- **Task automation.** The model can call tools to create a task (title, priority, due date) or mark a task complete by ID. Tasks are written to `tasks.json` and shown in the sidebar.
- **Session controls.** Clear the loaded document or reset the chat without restarting the app.

## Tech stack

| Layer | Choice |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| Chat model | Google Gemini (`LLM_MODEL` in `.env`, temperature `0.3`) |
| Embeddings | `EMBEDDING_MODEL` in `.env` |
| Vector search | FAISS (in memory, per session) |
| Documents | PyPDF for PDFs; UTF-8 decode for text files |
| Configuration | `python-dotenv` (`.env`) |

## Project structure

```text
ai-intelligent-assistant/
├── app.py              # Streamlit UI, chat loop, and tool execution
├── rag_engine.py       # Ingest, chunk, embed, and retrieve document context
├── tools.py            # Task tools and JSON persistence
├── requirements.txt    # Python dependencies
├── tasks.json          # Created at runtime; listed in .gitignore
├── .env.example        # Configuration template (safe to commit)
└── .env                # Local secrets and model names; listed in .gitignore
```

## Prerequisites

- Python 3.10 or newer
- A Google AI API key with access to the chat and embedding models set in `.env`

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy the template and fill in your key:

```bash
cp .env.example .env
```

```text
GOOGLE_API_KEY=your_api_key_here
LLM_MODEL=gemini-3.5-flash
EMBEDDING_MODEL=gemini-embedding-2-preview
```

The app stops on startup if `GOOGLE_API_KEY` or `LLM_MODEL` is missing. Document upload fails with a clear error if `EMBEDDING_MODEL` is missing.

## Run

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (typically `http://localhost:8501`).

## Usage

1. **Chat.** Type a question in the input at the bottom of the page. Prior turns stay in context until you reset the chat.
2. **Ask a document.** In the sidebar, upload a `.pdf` or `.txt` file. Wait for indexing to finish, then ask questions about that file. **Clear Document** drops the in-memory index.
3. **Manage tasks.** Ask the assistant to log or remind you of something, for example: “Remind me to submit the report tomorrow, high priority.” Completing a task requires its ID, for example: “Mark task 2 as done.” The sidebar lists each item with status, priority, and due date.

## How it works

**Retrieval.** `process_uploaded_file` extracts text, splits it with a recursive character splitter (`chunk_size=1000`, `chunk_overlap=150`), and builds a FAISS index with Gemini embeddings. `retrieve_context` returns the three nearest chunks for the current question. The index lives in session state and is discarded when the session ends or the document is cleared.

**Tools.** Gemini is bound to two LangChain tools:

- `create_action_item(title, priority="Medium", due_date="Not specified")` appends a pending task.
- `complete_action_item(task_id)` sets that task’s status to `Completed`.

If the model returns tool calls, the app runs each tool, appends the tool result, and calls the model once more for the user-facing reply. The page then reruns so the sidebar reflects the updated `tasks.json`.

## Configuration

Set these in `.env`. `.env.example` lists the same names without a real API key.

| Variable | Purpose |
|---|---|
| `GOOGLE_API_KEY` | Google AI API key |
| `LLM_MODEL` | Chat model passed to `ChatGoogleGenerativeAI` |
| `EMBEDDING_MODEL` | Embedding model passed to `GoogleGenerativeAIEmbeddings` |

Change the model values when your API key is provisioned for different model IDs. No source change is required.

## Limitations

- One document at a time, held in memory for the current browser session.
- Scanned PDFs with no extractable text are rejected.
- Task IDs are `len(tasks) + 1` and are not reused after deletion. There is no delete or edit tool.
- `tasks.json` is a local file shared by every session on the same machine.
- Tool execution is a single round: the model may call tools once, then answer. It does not loop until a stop condition.

## License

No license file is included in this repository.
