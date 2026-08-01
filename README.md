# Local RAG Emergency Assistant

A fully offline RAG-based emergency and disaster safety assistant with **code-level hallucination control**.

> The assistant responds in Turkish. All source documents are in Turkish.

## Why This Project?

In health and safety domains, an AI generating fabricated (hallucinated) information is unacceptable. Most existing RAG systems try to solve this by simply adding "if you're not sure, don't answer" to the prompt — leaving the decision to the model's discretion. This project takes a different approach: **whether the response stays within the provided documents is enforced programmatically, both before the model generates an answer and after.** The LLM only handles generation; both the entry and exit gates are in code.

## Architecture

```
User Query
       │
       ▼
┌─────────────────────────┐
│  Instruction-Prefixed   │   Qwen3-Embedding-0.6B
│  Query Embedding        │   (asymmetric: prefix applied to queries only)
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  GATE 1 — Entry Gate    │   database.py → search_with_scores()
│  Cosine Similarity      │   Below threshold → LLM is never called
│  + Threshold Filter     │   Above threshold → continue ↓
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  LLM — Generation Only  │   Phi-3.5 Mini (local, offline)
│  Response held in buffer │   Not yet shown to the user
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  GATE 2 — Exit Gate     │   rag_core.py → is_answer_grounded()
│  a) Grounding: each     │   Does the answer follow the chunks?
│     sentence checked    │
│  b) Source verification  │   Is the cited source actually retrieved?
└──────────┬──────────────┘
           ▼
    ┌──────┴──────┐
    │  Passed?    │
    ├── Yes ─────→ Answer shown to user
    └── No ──────→ Fallback: "Insufficient information."
```

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Phi-3.5 Mini (local, offline) |
| Embedding | Qwen3-Embedding-0.6B (asymmetric, instruction-prefixed) |
| Infrastructure | Microsoft Foundry Local SDK |
| Database | SQLite (vectors stored as JSON) |
| Interface | CustomTkinter (desktop GUI) + CLI |
| Language | Python |

## Project Structure

| File | Purpose |
|---|---|
| `rag_core.py` | Shared core — system prompt, Gate 2 verification, instruction-prefixed embedding |
| `database.py` | SQLite management, Gate 1 structured retrieval |
| `gui_tkinter.py` | Desktop interface (CustomTkinter) |
| `app.py` | Command-line interface (CLI) |
| `main.py` | Data ingestion tool |
| `test_rag.py` | Automated test suite (full pipeline) |
| `test_debug.py` | Gate 1/2 detailed diagnostic tool |
| `survival_database.json` | Source dataset (25 chunks, 8 guides) |

## Setup

### 1. Requirements

Python must be installed. Foundry Local SDK must be set up beforehand following the [Microsoft documentation](https://learn.microsoft.com/en-us/ai/foundry-local/).

```bash
pip install -r requirements.txt
```

### 2. Prepare the Database

Source documents are vectorized and loaded into SQLite. The embedding model is downloaded automatically on first run:

```bash
python main.py
```

### 3. Launch the Assistant

**Desktop interface (Tkinter):**
```bash
python gui_tkinter.py
```

**Command-line interface (CLI):**
```bash
python app.py
```

### 4. Run Tests

To see how the system handles different scenarios (correct information, missing information, nonsensical input):

```bash
python test_rag.py
```

## Design Decisions

**Why SQLite?** For portability and zero external dependencies on a single machine. Heavy-duty vector databases were unnecessary for this scale. Vectors are stored as JSON and processed in Python.

**Why two-gate control?** In health and safety contexts, showing a wrong answer even briefly is unacceptable. Gate 1 checks whether the question is answerable before sending it to the LLM; Gate 2 verifies the generated answer's faithfulness to the sources before showing it to the user. The model sits in the middle of the chain, handling generation only; both gates are programmatic.

**Why no streaming (Route A)?** If the answer is streamed token-by-token to the screen, Gate 2 can only run after the full answer is generated — but by then the user has already seen the (potentially wrong) answer. Instead, the answer is collected in a buffer, verified, and only displayed if it passes. The status bar shows "Generating..." and "Verifying..." so the user knows the system is working.

**Why instruction-prefixed embedding?** Qwen3-Embedding is an asymmetric model. Without the `Instruct:` prefix on queries, cosine similarity scores for correct matches drop below the threshold, causing valid questions to be falsely rejected. Documents are embedded raw; the prefix is applied only on the query side.

## Known Limitations

**Embedding model capacity (0.6B).** For certain query-chunk pairs, retrieval cannot rank the correct chunk first (e.g., "what to do in a flood while in a car" — the word "vehicle" pulls toward the earthquake-vehicle chunk instead). This is the natural ceiling of a small model. The few-shot prompt design prevents the model from fabricating answers in these cases, but does not fix the retrieval limitation itself.

**Local hardware constraints.** Since the model uses the device's RAM/VRAM, excessively long contexts may cause memory allocation errors. This is a device capacity limit, not a system bug.

---
