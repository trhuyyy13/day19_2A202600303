import json

from src.build_graph import build_graph
from src.chunking import chunk_articles
from src.collect_wikipedia import collect_wikipedia_articles
from src.config import CHUNKS_PATH, RAW_ARTICLES_PATH, TRIPLES_PATH, ensure_dirs
from src.evaluate import run_evaluation
from src.extract_triples_openai import extract_triples
from src.dashboard import generate_dashboard
from src.visualize_graph import visualize_graph


def main() -> None:
    ensure_dirs()

    if RAW_ARTICLES_PATH.exists():
        print(f"Using existing articles -> {RAW_ARTICLES_PATH}")
    else:
        collect_wikipedia_articles()

    if CHUNKS_PATH.exists():
        print(f"Using existing chunks -> {CHUNKS_PATH}")
        chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    else:
        chunks = chunk_articles()

    if TRIPLES_PATH.exists() and json.loads(TRIPLES_PATH.read_text(encoding="utf-8")):
        print(f"Using existing triples -> {TRIPLES_PATH}")
    else:
        extract_triples()

    graph = build_graph()
    visualize_graph(graph)
    run_evaluation(graph, chunks)
    generate_dashboard()
    print("Done.")


if __name__ == "__main__":
    main()
