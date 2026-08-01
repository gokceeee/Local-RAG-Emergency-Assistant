
Fully offline RAG-based emergency & disaster safety assistant with code-level hallucination control. Built with Microsoft Foundry Local + SQLite + Phi-3.5 Mini.

A question-answering assistant that runs entirely offline — no internet required. It answers user questions about earthquakes, floods, fires, first aid, and wilderness survival using pre-embedded local documents. The assistant responds in Turkish and all source documents are in Turkish.

The key design decision: hallucination control is not left to the model's judgment. Whether the response stays within the provided documents is enforced programmatically through two software gates:

Gate 1 (Entry — Pre-Generation): The user query is embedded with an instruction-prefixed asymmetric model and compared against document chunks via cosine similarity. If no chunk passes the relevance threshold, the LLM is never called.
Gate 2 (Exit — Post-Generation): After the LLM produces an answer, it is held in a buffer — not shown to the user — and verified: every sentence is checked for grounding against the retrieved chunks via embedding similarity, and every cited source is validated against the actually retrieved chunk metadata. Only answers that pass both checks are displayed; failed answers are replaced with a fallback message.
Tech Stack
LLM: Phi-3.5 Mini (local, offline)
Embedding: Qwen3-Embedding-0.6B (asymmetric, instruction-prefixed queries)
Infrastructure: Microsoft Foundry Local SDK
Database: SQLite (vectors stored as JSON)
Interface: CustomTkinter (desktop GUI) + CLI
Language: Python
