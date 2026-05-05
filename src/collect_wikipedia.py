import json

import wikipediaapi

from src.config import RAW_ARTICLES_PATH, WIKI_TOPICS, ensure_dirs


def collect_wikipedia_articles() -> list[dict]:
    ensure_dirs()
    wiki = wikipediaapi.Wikipedia(
        user_agent="GraphRAGWikipediaDemo/1.0 (learning demo)",
        language="en",
    )
    articles: list[dict] = []

    for topic in WIKI_TOPICS:
        page = wiki.page(topic)
        if not page.exists() or not page.text.strip():
            print(f"Skip missing Wikipedia page: {topic}")
            continue

        articles.append(
            {
                "title": page.title,
                "summary": page.summary,
                "text": page.text,
                "url": page.fullurl,
            }
        )

    RAW_ARTICLES_PATH.write_text(
        json.dumps(articles, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(f"Collected {len(articles)} Wikipedia articles -> {RAW_ARTICLES_PATH}")
    if len(articles) < 10:
        raise RuntimeError(f"Need at least 10 articles, collected {len(articles)}.")
    return articles
