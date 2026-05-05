import json

import networkx as nx
import pandas as pd

from src.config import GRAPH_EDGES_PATH, GRAPH_NODES_PATH, TRIPLES_PATH, ensure_dirs


def build_graph() -> nx.DiGraph:
    ensure_dirs()
    triples = json.loads(TRIPLES_PATH.read_text(encoding="utf-8"))
    graph = nx.DiGraph()

    for triple in triples:
        subject = triple["subject"]
        object_ = triple["object"]
        graph.add_node(subject)
        graph.add_node(object_)
        graph.add_edge(
            subject,
            object_,
            relation=triple["relation"],
            evidence=triple["evidence"],
            chunk_id=triple["chunk_id"],
            source_title=triple["source_title"],
        )

    node_rows = [
        {"node": node, "degree": degree}
        for node, degree in sorted(graph.degree(), key=lambda item: item[1], reverse=True)
    ]
    edge_rows = [
        {
            "subject": source,
            "relation": attrs.get("relation", ""),
            "object": target,
            "evidence": attrs.get("evidence", ""),
            "chunk_id": attrs.get("chunk_id", ""),
            "source_title": attrs.get("source_title", ""),
        }
        for source, target, attrs in graph.edges(data=True)
    ]

    pd.DataFrame(node_rows).to_csv(GRAPH_NODES_PATH, index=False)
    pd.DataFrame(edge_rows).to_csv(GRAPH_EDGES_PATH, index=False)
    print(
        f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
    )
    return graph
