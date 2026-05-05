import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib.pyplot as plt
import networkx as nx

from src.config import GRAPH_IMAGE_PATH, ensure_dirs


def visualize_graph(graph: nx.DiGraph) -> None:
    ensure_dirs()
    if graph.number_of_nodes() == 0:
        raise RuntimeError("Cannot visualize an empty graph.")

    if graph.number_of_nodes() > 20:
        top_nodes = [
            node
            for node, _ in sorted(graph.degree(), key=lambda item: item[1], reverse=True)[:20]
        ]
        graph_to_draw = graph.subgraph(top_nodes).copy()
    else:
        graph_to_draw = graph

    plt.figure(figsize=(16, 11))
    pos = nx.spring_layout(graph_to_draw, seed=42, k=0.9)
    nx.draw_networkx_nodes(
        graph_to_draw,
        pos,
        node_size=1700,
        node_color="#d8ecff",
        edgecolors="#2f5f8f",
        linewidths=1.2,
    )
    nx.draw_networkx_edges(
        graph_to_draw,
        pos,
        arrowstyle="-|>",
        arrowsize=14,
        edge_color="#707070",
        width=1.2,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_labels(graph_to_draw, pos, font_size=8)
    edge_labels = {
        (source, target): attrs.get("relation", "")
        for source, target, attrs in graph_to_draw.edges(data=True)
    }
    nx.draw_networkx_edge_labels(graph_to_draw, pos, edge_labels=edge_labels, font_size=7)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(GRAPH_IMAGE_PATH, dpi=180)
    plt.close()
    print(f"Saved graph image -> {GRAPH_IMAGE_PATH}")
