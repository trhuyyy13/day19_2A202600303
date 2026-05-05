import json

from tqdm import tqdm

from src.config import CHUNKS_PATH, MAX_CHUNKS_FOR_EXTRACTION, TRIPLES_PATH, ensure_dirs
from src.config import load_settings
from src.openai_utils import chat_completion, parse_json_array


TRIPLE_PROMPT = """You are an information extraction system for building a knowledge graph.

Extract factual knowledge graph triples from the text.

Rules:
- Return JSON only.
- Each triple must have: subject, relation, object, evidence.
- Use concise entity names.
- Do not invent facts.
- Only extract facts explicitly supported by the text.
- Prefer relations such as:
  CREATED_BY, DEVELOPED_BY, OWNED_BY, PART_OF, RELATED_TO, USES,
  BASED_ON, PROVIDES, SUPPORTS, DEPENDS_ON, FOUNDED_BY, LOCATED_IN.

Text:
{chunk_text}

Output JSON format:
[
  {{
    "subject": "...",
    "relation": "...",
    "object": "...",
    "evidence": "..."
  }}
]"""


def extract_triples() -> list[dict]:
    ensure_dirs()
    api_key, _ = load_settings()
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Copy .env.example to .env and fill OPENAI_API_KEY."
        )

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    chunks_for_extraction = chunks[:MAX_CHUNKS_FOR_EXTRACTION]
    triples: list[dict] = []

    for chunk in tqdm(chunks_for_extraction, desc="Extract triples"):
        prompt = TRIPLE_PROMPT.format(chunk_text=chunk["text"])
        try:
            response_text = chat_completion(prompt)
            extracted = parse_json_array(response_text)
        except Exception as exc:
            print(f"Skip chunk {chunk['chunk_id']} due to extraction error: {exc}")
            continue

        for item in extracted:
            subject = str(item.get("subject", "")).strip()
            relation = str(item.get("relation", "")).strip().upper().replace(" ", "_")
            object_ = str(item.get("object", "")).strip()
            evidence = str(item.get("evidence", "")).strip()
            if not subject or not relation or not object_ or not evidence:
                continue
            triples.append(
                {
                    "subject": subject,
                    "relation": relation,
                    "object": object_,
                    "evidence": evidence,
                    "chunk_id": chunk["chunk_id"],
                    "source_title": chunk["title"],
                }
            )

    if not triples:
        raise RuntimeError("No triples were extracted. Check OpenAI responses and logs.")

    TRIPLES_PATH.write_text(json.dumps(triples, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Extracted {len(triples)} triples -> {TRIPLES_PATH}")
    return triples
