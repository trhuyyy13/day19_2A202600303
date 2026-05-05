import ast
import json
from collections import Counter
from typing import Any

import networkx as nx
import pandas as pd

from src.config import (
    BENCHMARK_QUESTIONS_PATH,
    CHUNKS_PATH,
    COMPARISON_RESULTS_PATH,
    EVALUATION_METRICS_PATH,
    FLAT_RAG_RESULTS_PATH,
    GRAPH_EDGES_PATH,
    GRAPH_NODES_PATH,
    GRAPH_RAG_RESULTS_PATH,
    QUESTION_METRICS_PATH,
    RAW_ARTICLES_PATH,
    RELATION_COUNTS_PATH,
    SOURCE_COVERAGE_PATH,
    TRIPLES_PATH,
    ensure_dirs,
)


def _read_json(path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            return []
    return parsed if isinstance(parsed, list) else []


def _is_insufficient(answer: Any) -> bool:
    text = str(answer).lower()
    return "insufficient" in text or "does not contain enough information" in text


def _answer_words(answer: Any) -> int:
    return len(str(answer).split())


def _build_graph(edges_df: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    for _, row in edges_df.iterrows():
        graph.add_edge(
            str(row["subject"]),
            str(row["object"]),
            relation=str(row["relation"]),
            source_title=str(row.get("source_title", "")),
        )
    return graph


def generate_metrics() -> dict:
    ensure_dirs()
    articles = _read_json(RAW_ARTICLES_PATH) if RAW_ARTICLES_PATH.exists() else []
    chunks = _read_json(CHUNKS_PATH) if CHUNKS_PATH.exists() else []
    triples = _read_json(TRIPLES_PATH) if TRIPLES_PATH.exists() else []
    nodes_df = pd.read_csv(GRAPH_NODES_PATH) if GRAPH_NODES_PATH.exists() else pd.DataFrame()
    edges_df = pd.read_csv(GRAPH_EDGES_PATH) if GRAPH_EDGES_PATH.exists() else pd.DataFrame()
    questions_df = (
        pd.read_csv(BENCHMARK_QUESTIONS_PATH)
        if BENCHMARK_QUESTIONS_PATH.exists()
        else pd.DataFrame()
    )
    flat_df = (
        pd.read_csv(FLAT_RAG_RESULTS_PATH)
        if FLAT_RAG_RESULTS_PATH.exists()
        else pd.DataFrame()
    )
    graph_df = (
        pd.read_csv(GRAPH_RAG_RESULTS_PATH)
        if GRAPH_RAG_RESULTS_PATH.exists()
        else pd.DataFrame()
    )
    comparison_df = (
        pd.read_csv(COMPARISON_RESULTS_PATH)
        if COMPARISON_RESULTS_PATH.exists()
        else pd.DataFrame()
    )

    graph = _build_graph(edges_df) if not edges_df.empty else nx.DiGraph()
    weak_components = list(nx.weakly_connected_components(graph)) if graph.number_of_nodes() else []
    largest_component_size = max((len(component) for component in weak_components), default=0)

    relation_counts = (
        edges_df["relation"].value_counts().rename_axis("relation").reset_index(name="edge_count")
        if not edges_df.empty
        else pd.DataFrame(columns=["relation", "edge_count"])
    )
    relation_counts.to_csv(RELATION_COUNTS_PATH, index=False)

    chunk_counter = Counter(chunk["title"] for chunk in chunks)
    triple_counter = Counter(triple["source_title"] for triple in triples)
    source_rows = []
    for article in articles:
        title = article["title"]
        source_rows.append(
            {
                "source_title": title,
                "chunks": chunk_counter.get(title, 0),
                "triples": triple_counter.get(title, 0),
                "triples_per_chunk": round(
                    triple_counter.get(title, 0) / max(1, chunk_counter.get(title, 0)), 3
                ),
                "url": article.get("url", ""),
            }
        )
    source_df = pd.DataFrame(source_rows)
    source_df.to_csv(SOURCE_COVERAGE_PATH, index=False)

    question_rows = []
    if not comparison_df.empty:
        flat_by_id = {int(row["id"]): row for _, row in flat_df.iterrows()} if not flat_df.empty else {}
        graph_by_id = {int(row["id"]): row for _, row in graph_df.iterrows()} if not graph_df.empty else {}
        for _, row in comparison_df.iterrows():
            qid = int(row["id"])
            flat_row = flat_by_id.get(qid, {})
            graph_row = graph_by_id.get(qid, {})
            retrieved_chunks = _parse_list(flat_row.get("retrieved_chunks", "[]"))
            graph_edges = _parse_list(graph_row.get("graph_edges", "[]"))
            seed_entity = str(graph_row.get("seed_entity", "") or "")
            question = str(row["question"])
            seed_exact_match = bool(seed_entity and seed_entity.lower() in question.lower())
            question_rows.append(
                {
                    "id": qid,
                    "question": question,
                    "expected_type": row["expected_type"],
                    "winner": row["winner"],
                    "flat_answer_words": _answer_words(row["flat_rag_answer"]),
                    "graph_answer_words": _answer_words(row["graph_rag_answer"]),
                    "flat_insufficient": _is_insufficient(row["flat_rag_answer"]),
                    "graph_insufficient": _is_insufficient(row["graph_rag_answer"]),
                    "retrieved_chunk_count": len(retrieved_chunks),
                    "graph_edge_count": len(graph_edges),
                    "seed_entity": seed_entity,
                    "seed_exact_match": seed_exact_match,
                }
            )
    question_metrics_df = pd.DataFrame(question_rows)
    question_metrics_df.to_csv(QUESTION_METRICS_PATH, index=False)

    total_questions = len(comparison_df)
    winner_counts = (
        comparison_df["winner"].value_counts().to_dict() if not comparison_df.empty else {}
    )
    type_winner_counts = (
        comparison_df.groupby(["expected_type", "winner"]).size().unstack(fill_value=0).to_dict()
        if not comparison_df.empty
        else {}
    )

    graph_answerable = (
        int((~question_metrics_df["graph_insufficient"]).sum())
        if not question_metrics_df.empty
        else 0
    )
    flat_answerable = (
        int((~question_metrics_df["flat_insufficient"]).sum())
        if not question_metrics_df.empty
        else 0
    )
    seed_exact = (
        int(question_metrics_df["seed_exact_match"].sum()) if not question_metrics_df.empty else 0
    )

    metrics = {
        "corpus": {
            "articles": len(articles),
            "chunks": len(chunks),
            "triples": len(triples),
            "avg_chunks_per_article": round(len(chunks) / max(1, len(articles)), 2),
            "avg_triples_per_chunk": round(len(triples) / max(1, len(chunks)), 2),
        },
        "graph": {
            "nodes": int(graph.number_of_nodes()),
            "edges": int(graph.number_of_edges()),
            "density": round(nx.density(graph), 6) if graph.number_of_nodes() else 0,
            "weak_components": len(weak_components),
            "largest_component_nodes": largest_component_size,
            "largest_component_share": round(
                largest_component_size / max(1, graph.number_of_nodes()), 3
            ),
            "avg_degree": round(
                sum(dict(graph.degree()).values()) / max(1, graph.number_of_nodes()), 2
            ),
            "top_nodes_by_degree": nodes_df.head(10).to_dict("records")
            if not nodes_df.empty
            else [],
            "top_relations": relation_counts.head(10).to_dict("records"),
        },
        "rag": {
            "questions": total_questions,
            "winner_counts": winner_counts,
            "winner_rate": {
                key: round(value / max(1, total_questions), 3)
                for key, value in winner_counts.items()
            },
            "flat_answerable": flat_answerable,
            "graph_answerable": graph_answerable,
            "flat_answerable_rate": round(flat_answerable / max(1, total_questions), 3),
            "graph_answerable_rate": round(graph_answerable / max(1, total_questions), 3),
            "seed_exact_match": seed_exact,
            "seed_exact_match_rate": round(seed_exact / max(1, total_questions), 3),
            "avg_retrieved_chunks": round(
                question_metrics_df["retrieved_chunk_count"].mean(), 2
            )
            if not question_metrics_df.empty
            else 0,
            "avg_graph_edges_used": round(question_metrics_df["graph_edge_count"].mean(), 2)
            if not question_metrics_df.empty
            else 0,
            "by_expected_type": type_winner_counts,
        },
    }

    EVALUATION_METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(f"Saved metrics -> {EVALUATION_METRICS_PATH}")
    return metrics
