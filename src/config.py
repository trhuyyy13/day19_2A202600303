from pathlib import Path

from dotenv import load_dotenv
import os


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"

RAW_ARTICLES_PATH = DATA_DIR / "raw_articles.json"
CHUNKS_PATH = DATA_DIR / "chunks.json"
TRIPLES_PATH = DATA_DIR / "triples.json"
BENCHMARK_QUESTIONS_PATH = DATA_DIR / "benchmark_questions.csv"

GRAPH_NODES_PATH = OUTPUTS_DIR / "graph_nodes.csv"
GRAPH_EDGES_PATH = OUTPUTS_DIR / "graph_edges.csv"
GRAPH_IMAGE_PATH = OUTPUTS_DIR / "graph.png"
FLAT_RAG_RESULTS_PATH = OUTPUTS_DIR / "flat_rag_results.csv"
GRAPH_RAG_RESULTS_PATH = OUTPUTS_DIR / "graph_rag_results.csv"
COMPARISON_RESULTS_PATH = OUTPUTS_DIR / "comparison_results.csv"
EVALUATION_METRICS_PATH = OUTPUTS_DIR / "evaluation_metrics.json"
RELATION_COUNTS_PATH = OUTPUTS_DIR / "relation_counts.csv"
SOURCE_COVERAGE_PATH = OUTPUTS_DIR / "source_coverage.csv"
QUESTION_METRICS_PATH = OUTPUTS_DIR / "question_metrics.csv"
DASHBOARD_HTML_PATH = OUTPUTS_DIR / "dashboard.html"

WIKI_TOPICS = [
    "OpenAI",
    "Microsoft",
    "Google",
    "Amazon Web Services",
    "Cloud computing",
    "Artificial intelligence",
    "Machine learning",
    "Kubernetes",
    "Docker (software)",
    "PostgreSQL",
    "Python (programming language)",
    "Microservices",
    "DevOps",
    "GitHub",
    "Linux",
]

CHUNK_SIZE_WORDS = 400
CHUNK_OVERLAP_WORDS = 50
MAX_CHUNKS_FOR_EXTRACTION = 40
GRAPH_HOP_K = 2
TOP_K_CHUNKS = 3

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_settings() -> tuple[str | None, str]:
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    return api_key, model
