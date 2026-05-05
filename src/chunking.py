import json
import re

from src.config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    CHUNKS_PATH,
    RAW_ARTICLES_PATH,
    ensure_dirs,
)


def _words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def chunk_articles() -> list[dict]:
    ensure_dirs()
    articles = json.loads(RAW_ARTICLES_PATH.read_text(encoding="utf-8"))
    chunks: list[dict] = []
    step = max(1, CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS)

    for article in articles:
        words = _words(article["text"])
        title_slug = re.sub(r"[^A-Za-z0-9]+", "_", article["title"]).strip("_")
        for index, start in enumerate(range(0, len(words), step)):
            window = words[start : start + CHUNK_SIZE_WORDS]
            if len(window) < 80:
                continue
            chunks.append(
                {
                    "chunk_id": f"{title_slug}_chunk_{index}",
                    "title": article["title"],
                    "text": " ".join(window),
                    "source": "Wikipedia",
                    "url": article.get("url", ""),
                }
            )

    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Created {len(chunks)} chunks -> {CHUNKS_PATH}")
    return chunks
