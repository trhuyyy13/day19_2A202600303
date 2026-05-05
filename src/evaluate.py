import json

import pandas as pd

from src.config import (
    BENCHMARK_QUESTIONS_PATH,
    COMPARISON_RESULTS_PATH,
    FLAT_RAG_RESULTS_PATH,
    GRAPH_RAG_RESULTS_PATH,
    TOP_K_CHUNKS,
    ensure_dirs,
)
from src.flat_rag import flat_rag_answer
from src.graph_rag import graph_rag_answer


BENCHMARK_ROWS = [
    (1, "Using the graph paths, how is Microsoft connected to OpenAI through investment, Azure, and Copilot?", "multi-hop"),
    (2, "Trace the ownership path from OpenAI Global, LLC to OpenAI LP and OpenAI, Inc.", "multi-hop"),
    (3, "How does Microsoft connect to OpenAI Group PBC after OpenAI's restructuring?", "multi-hop"),
    (4, "What graph path connects ChatGPT to Microsoft through OpenAI?", "multi-hop"),
    (5, "How are Sam Altman and Microsoft connected through OpenAI's 2023 leadership crisis?", "multi-hop"),
    (6, "How is Elon Musk connected to OpenAI and Sam Altman through lawsuits and bids?", "multi-hop"),
    (7, "Which cloud and compute providers are connected to OpenAI, and what roles do they play?", "multi-hop"),
    (8, "How is The Stargate Project connected to OpenAI and other infrastructure partners?", "multi-hop"),
    (9, "Which OpenAI products or models are connected to GPT-2, GPT-3, GPT-4, DALL-E, Sora, Operator, and ChatGPT?", "multi-hop"),
    (10, "How is OpenAI connected to government or defense entities such as DoD, AFRICOM, and Anduril?", "multi-hop"),
    (11, "How is Ilya Sutskever connected to OpenAI's safety and superalignment story?", "multi-hop"),
    (12, "What path connects Microsoft to MS-DOS, Windows, Windows NT, and Xbox?", "multi-hop"),
    (13, "How are Bill Gates, Paul Allen, Traf-O-Data, and Microsoft connected?", "multi-hop"),
    (14, "How is Microsoft connected to GitHub, LinkedIn, and Activision Blizzard through acquisitions?", "multi-hop"),
    (15, "How does Satya Nadella connect to Steve Ballmer, Microsoft, and Sam Altman?", "multi-hop"),
    (16, "How is OpenAI connected to News Corp, The New York Times, and copyright-related disputes?", "multi-hop"),
    (17, "How does OpenAI connect to Whisper, YouTube transcription, and GPT-4 training?", "multi-hop"),
    (18, "How are OpenAI Foundation, OpenAI Group PBC, the nonprofit, and Microsoft connected in the graph?", "multi-hop"),
    (19, "How is OpenAI connected to Nvidia, AMD, Oracle, Google Cloud TPUs, and Azure services?", "multi-hop"),
    (20, "Which graph paths show Microsoft expanding from operating systems into cloud, developer tools, gaming, and AI?", "multi-hop"),
]


def ensure_benchmark_questions() -> None:
    ensure_dirs()
    if BENCHMARK_QUESTIONS_PATH.exists():
        return
    df = pd.DataFrame(BENCHMARK_ROWS, columns=["id", "question", "expected_type"])
    df.to_csv(BENCHMARK_QUESTIONS_PATH, index=False)
    print(f"Created benchmark questions -> {BENCHMARK_QUESTIONS_PATH}")


def _is_insufficient(answer: str) -> bool:
    normalized = answer.lower()
    return "insufficient" in normalized or "does not contain enough information" in normalized


def _choose_winner(flat_result: dict, graph_result: dict) -> tuple[str, str]:
    flat_insufficient = _is_insufficient(flat_result["answer"])
    graph_insufficient = _is_insufficient(graph_result["answer"])
    graph_has_edges = bool(graph_result.get("graph_edges"))

    if graph_has_edges and not graph_insufficient and flat_insufficient:
        return "GraphRAG", "GraphRAG found a graph path while Flat RAG lacked context."
    if not flat_insufficient and graph_insufficient:
        return "Flat RAG", "Flat RAG had usable retrieved text while graph context was insufficient."
    if graph_has_edges and not graph_insufficient:
        return "GraphRAG", "GraphRAG answered with an explicit graph path."
    if not flat_insufficient:
        return "Flat RAG", "Flat RAG gave a direct answer from retrieved chunks."
    return "Tie", "Both methods had insufficient context."


def run_evaluation(graph, chunks: list[dict]) -> pd.DataFrame:
    ensure_benchmark_questions()
    questions = pd.read_csv(BENCHMARK_QUESTIONS_PATH)
    flat_rows = []
    graph_rows = []
    comparison_rows = []

    for _, row in questions.iterrows():
        query = row["question"]
        flat_result = flat_rag_answer(query, chunks, top_k=TOP_K_CHUNKS)
        graph_result = graph_rag_answer(query, graph)
        winner, notes = _choose_winner(flat_result, graph_result)

        flat_rows.append(
            {
                "id": row["id"],
                "question": query,
                "answer": flat_result["answer"],
                "retrieved_chunks": json.dumps(flat_result["retrieved_chunks"]),
                "method": flat_result["method"],
            }
        )
        graph_rows.append(
            {
                "id": row["id"],
                "question": query,
                "answer": graph_result["answer"],
                "seed_entity": graph_result["seed_entity"],
                "graph_edges": json.dumps(graph_result["graph_edges"]),
                "method": graph_result["method"],
            }
        )
        comparison_rows.append(
            {
                "id": row["id"],
                "question": query,
                "expected_type": row["expected_type"],
                "flat_rag_answer": flat_result["answer"],
                "graph_rag_answer": graph_result["answer"],
                "winner": winner,
                "notes": notes,
            }
        )

    pd.DataFrame(flat_rows).to_csv(FLAT_RAG_RESULTS_PATH, index=False)
    pd.DataFrame(graph_rows).to_csv(GRAPH_RAG_RESULTS_PATH, index=False)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(COMPARISON_RESULTS_PATH, index=False)
    print(f"Saved evaluation results -> {COMPARISON_RESULTS_PATH}")
    return comparison
