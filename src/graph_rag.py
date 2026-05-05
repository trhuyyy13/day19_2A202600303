import networkx as nx

from src.config import GRAPH_HOP_K
from src.openai_utils import chat_completion


GRAPH_ANSWER_PROMPT = """Answer the question using only the graph context below.
Explain the reasoning path briefly.
If the graph context is insufficient, say that the graph does not contain enough information.

Question:
{query}

Graph context:
{graph_context}
"""


def detect_seed_entity(query: str, nodes) -> str | None:
    query_lower = query.lower()
    sorted_nodes = sorted(nodes, key=lambda node: len(str(node)), reverse=True)
    for node in sorted_nodes:
        if str(node).lower() in query_lower:
            return node

    query_terms = set(query_lower.replace("?", "").split())
    best_node = None
    best_score = 0
    for node in nodes:
        node_terms = set(str(node).lower().split())
        score = len(query_terms & node_terms)
        if score > best_score:
            best_node = node
            best_score = score
    return best_node if best_score > 0 else None


def get_k_hop_subgraph(graph: nx.DiGraph, seed_node: str, k: int = GRAPH_HOP_K) -> nx.DiGraph:
    visited = {seed_node}
    frontier = {seed_node}

    for _ in range(k):
        next_frontier = set()
        for node in frontier:
            neighbors = set(graph.successors(node)) | set(graph.predecessors(node))
            next_frontier.update(neighbors)
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break

    return graph.subgraph(visited).copy()


def textualize_graph_context(subgraph: nx.DiGraph) -> str:
    lines = []
    for source, target, attrs in subgraph.edges(data=True):
        evidence = attrs.get("evidence", "")
        relation = attrs.get("relation", "RELATED_TO")
        lines.append(f"{source} {relation} {target}. Evidence: {evidence}")
    return "\n".join(lines)


def graph_rag_answer(query: str, graph: nx.DiGraph, k: int = GRAPH_HOP_K) -> dict:
    seed = detect_seed_entity(query, graph.nodes)
    if not seed:
        return {
            "query": query,
            "answer": "The graph does not contain enough information.",
            "seed_entity": "",
            "graph_edges": [],
            "method": "GraphRAG",
        }

    subgraph = get_k_hop_subgraph(graph, seed, k=k)
    graph_context = textualize_graph_context(subgraph)
    if not graph_context:
        answer = "The graph does not contain enough information."
    else:
        answer = chat_completion(
            GRAPH_ANSWER_PROMPT.format(query=query, graph_context=graph_context)
        )

    edge_ids = [
        f"{source} -[{attrs.get('relation', '')}]-> {target}"
        for source, target, attrs in subgraph.edges(data=True)
    ]
    return {
        "query": query,
        "answer": answer,
        "seed_entity": seed,
        "graph_edges": edge_ids,
        "method": "GraphRAG",
    }
