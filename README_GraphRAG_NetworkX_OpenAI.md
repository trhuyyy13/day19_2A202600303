# GraphRAG Wikipedia Demo với NetworkX + OpenAI API

## 1. Mục tiêu

Xây dựng demo GraphRAG đơn giản nhưng chạy thật:

- Knowledge base: 10–20 bài Wikipedia.
- Dùng **OpenAI API** để trích xuất triples: `(subject, relation, object)`.
- Dùng **NetworkX** để xây dựng và duyệt graph.
- Dùng **Flat RAG** làm baseline.
- Dùng **GraphRAG** để truy vấn entity, duyệt BFS 2-hop và sinh câu trả lời.
- Xuất ảnh graph và bảng so sánh kết quả.

Không mock dữ liệu. Không hard-code answer. Không dùng Neo4j. Không dùng NodeRAG.

---

## 2. Tech stack

Sử dụng:

- `openai`: gọi OpenAI API để extract triples và generate answer.
- `networkx`: xây dựng knowledge graph.
- `wikipedia-api`: lấy dữ liệu Wikipedia.
- `scikit-learn`: làm Flat RAG bằng TF-IDF.
- `matplotlib`: vẽ graph.
- `pandas`: lưu bảng kết quả.
- `python-dotenv`: đọc API key từ `.env`.

---

## 3. Cấu trúc project

```text
graph_rag_wikipedia_demo/
├── README.md
├── requirements.txt
├── .env
├── .env.example
├── main.py
├── data/
│   ├── raw_articles.json
│   ├── chunks.json
│   ├── triples.json
│   └── benchmark_questions.csv
├── outputs/
│   ├── graph.png
│   ├── graph_nodes.csv
│   ├── graph_edges.csv
│   ├── flat_rag_results.csv
│   ├── graph_rag_results.csv
│   └── comparison_results.csv
└── src/
    ├── config.py
    ├── collect_wikipedia.py
    ├── chunking.py
    ├── extract_triples_openai.py
    ├── build_graph.py
    ├── flat_rag.py
    ├── graph_rag.py
    ├── evaluate.py
    └── visualize_graph.py
```

---

## 4. Tạo môi trường

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

---

## 5. File requirements.txt

```txt
openai
networkx
matplotlib
pandas
numpy
wikipedia-api
scikit-learn
tqdm
python-dotenv
```

---

## 6. File .env

Tạo file `.env` ở thư mục gốc:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

Không commit file `.env` lên GitHub.

Tạo thêm `.env.example`:

```env
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
```

---

## 7. Corpus Wikipedia

Trong `src/config.py`, dùng khoảng 10–20 bài:

```python
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
    "Linux"
]
```

Các config khác:

```python
CHUNK_SIZE_WORDS = 400
CHUNK_OVERLAP_WORDS = 50
MAX_CHUNKS_FOR_EXTRACTION = 40
GRAPH_HOP_K = 2
TOP_K_CHUNKS = 3
```

---

## 8. Pipeline tổng quát

`main.py` chạy theo thứ tự:

```text
1. Tải bài Wikipedia
2. Chia bài viết thành chunks
3. Gọi OpenAI API để extract triples
4. Build graph bằng NetworkX
5. Vẽ graph ra graph.png
6. Chạy Flat RAG baseline
7. Chạy GraphRAG
8. Lưu bảng comparison_results.csv
```

Chạy project:

```bash
python main.py
```

---

## 9. Bước 1: Collect Wikipedia

File: `src/collect_wikipedia.py`

Nhiệm vụ:

- Lấy dữ liệu từ Wikipedia theo danh sách `WIKI_TOPICS`.
- Lưu `title`, `summary`, `text`, `url`.
- Output: `data/raw_articles.json`.

Format:

```json
[
  {
    "title": "OpenAI",
    "summary": "...",
    "text": "...",
    "url": "..."
  }
]
```

Yêu cầu:

- Nếu bài không lấy được thì bỏ qua.
- Log số bài lấy thành công.
- Cần lấy được ít nhất 10 bài.

---

## 10. Bước 2: Chunking

File: `src/chunking.py`

Nhiệm vụ:

- Đọc `data/raw_articles.json`.
- Chia text thành chunk 300–500 từ.
- Lưu vào `data/chunks.json`.

Format:

```json
[
  {
    "chunk_id": "OpenAI_chunk_0",
    "title": "OpenAI",
    "text": "OpenAI is an artificial intelligence organization...",
    "source": "Wikipedia"
  }
]
```

---

## 11. Bước 3: Extract triples bằng OpenAI

File: `src/extract_triples_openai.py`

Nhiệm vụ:

- Đọc `data/chunks.json`.
- Gửi từng chunk vào OpenAI API.
- Yêu cầu model trả về JSON triples.
- Lưu kết quả vào `data/triples.json`.

Prompt gợi ý:

```text
You are an information extraction system for building a knowledge graph.

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
  {
    "subject": "...",
    "relation": "...",
    "object": "...",
    "evidence": "..."
  }
]
```

Output `data/triples.json`:

```json
[
  {
    "subject": "OpenAI",
    "relation": "PARTNERED_WITH",
    "object": "Microsoft",
    "evidence": "...",
    "chunk_id": "OpenAI_chunk_0",
    "source_title": "OpenAI"
  }
]
```

Lưu ý implementation:

- Dùng model từ `.env`, mặc định `gpt-4o-mini`.
- Parse JSON cẩn thận.
- Nếu response lỗi JSON, bỏ chunk đó và log lỗi.
- Có thể giới hạn `MAX_CHUNKS_FOR_EXTRACTION = 40` để tiết kiệm chi phí.
- Không tạo triples thủ công thay cho OpenAI.

---

## 12. Bước 4: Build graph bằng NetworkX

File: `src/build_graph.py`

Nhiệm vụ:

- Đọc `data/triples.json`.
- Tạo directed graph `nx.DiGraph()`.
- Mỗi `subject` và `object` là node.
- Mỗi triple là edge.
- Edge lưu:
  - `relation`
  - `evidence`
  - `chunk_id`
  - `source_title`

Ví dụ:

```python
G.add_node(subject)
G.add_node(object_)
G.add_edge(
    subject,
    object_,
    relation=relation,
    evidence=evidence,
    chunk_id=chunk_id,
    source_title=source_title
)
```

Xuất:

```text
outputs/graph_nodes.csv
outputs/graph_edges.csv
```

---

## 13. Bước 5: Visualize graph

File: `src/visualize_graph.py`

Nhiệm vụ:

- Vẽ graph bằng Matplotlib.
- Nếu graph lớn, lấy top 20 node có degree cao nhất.
- Lưu ảnh:

```text
outputs/graph.png
```

Ảnh cần có:

- Node labels.
- Edge labels là relation.
- Bố cục nhìn được khi đưa vào báo cáo.

---

## 14. Bước 6: Flat RAG baseline

File: `src/flat_rag.py`

Flat RAG chỉ dùng text chunks, không dùng graph.

Cách làm:

- Đọc `data/chunks.json`.
- Dùng TF-IDF để retrieve top-k chunk liên quan đến query.
- Gửi query + retrieved chunks vào OpenAI API để generate answer.

Prompt answer:

```text
Answer the question using only the context below.
If the context is insufficient, say that the context is insufficient.

Question:
{query}

Context:
{retrieved_context}
```

Function chính:

```python
def flat_rag_answer(query: str, chunks: list, top_k: int = 3) -> dict:
    ...
```

Output mỗi câu hỏi gồm:

```json
{
  "query": "...",
  "answer": "...",
  "retrieved_chunks": ["..."],
  "method": "Flat RAG"
}
```

---

## 15. Bước 7: GraphRAG bằng NetworkX

File: `src/graph_rag.py`

GraphRAG chạy như sau:

```text
query
→ detect seed entity
→ tìm node trong graph
→ BFS 2-hop
→ lấy các edges liên quan
→ textualize graph context
→ gửi query + graph context vào OpenAI
→ trả lời
```

### Detect seed entity

Đơn giản:

```python
def detect_seed_entity(query, nodes):
    query_lower = query.lower()
    for node in nodes:
        if node.lower() in query_lower:
            return node
    return None
```

Nếu không tìm thấy seed entity, có thể fallback bằng cách chọn node có nhiều từ trùng với query nhất.

### BFS 2-hop

```python
def get_k_hop_subgraph(G, seed_node, k=2):
    visited = {seed_node}
    frontier = {seed_node}

    for _ in range(k):
        next_frontier = set()
        for node in frontier:
            neighbors = set(G.successors(node)) | set(G.predecessors(node))
            next_frontier.update(neighbors)
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier

    return G.subgraph(visited).copy()
```

### Textualize graph context

Chuyển edge thành text:

```text
OpenAI FOUNDED_BY Sam Altman. Evidence: ...
OpenAI PARTNERED_WITH Microsoft. Evidence: ...
Microsoft PROVIDES Azure. Evidence: ...
```

### Generate answer bằng OpenAI

Prompt:

```text
Answer the question using only the graph context below.
Explain the reasoning path briefly.
If the graph context is insufficient, say that the graph does not contain enough information.

Question:
{query}

Graph context:
{graph_context}
```

Function chính:

```python
def graph_rag_answer(query: str, G, k: int = 2) -> dict:
    ...
```

---

## 16. Benchmark questions

Tạo file `data/benchmark_questions.csv`:

```csv
id,question,expected_type
1,What is OpenAI related to?,single-hop
2,How is Microsoft connected to OpenAI?,single-hop
3,How is Kubernetes related to cloud computing?,multi-hop
4,How is Docker related to Linux?,multi-hop
5,Which entities are connected to artificial intelligence?,single-hop
6,How is Python related to machine learning?,multi-hop
7,Which entities are connected to Amazon Web Services?,single-hop
8,How is microservices related to cloud computing?,multi-hop
9,How is GitHub connected to software engineering?,multi-hop
10,Which entities are connected to PostgreSQL?,single-hop
```

Có thể tạo 10 câu cho demo ngắn. Nếu cần đủ deliverables thì tăng lên 20 câu.

---

## 17. Evaluation

File: `src/evaluate.py`

Nhiệm vụ:

- Đọc `benchmark_questions.csv`.
- Chạy từng câu hỏi qua Flat RAG và GraphRAG.
- Lưu kết quả:

```text
outputs/flat_rag_results.csv
outputs/graph_rag_results.csv
outputs/comparison_results.csv
```

Format `comparison_results.csv`:

```csv
id,question,expected_type,flat_rag_answer,graph_rag_answer,winner,notes
```

Cách điền `winner` đơn giản:

- Nếu GraphRAG có graph path rõ ràng và answer đúng trọng tâm hơn, chọn `GraphRAG`.
- Nếu Flat RAG trả lời trực tiếp hơn, chọn `Flat RAG`.
- Nếu chưa chắc, chọn `Tie`.

---

## 18. Output bắt buộc

Sau khi chạy xong, cần có:

```text
data/raw_articles.json
data/chunks.json
data/triples.json
data/benchmark_questions.csv
outputs/graph.png
outputs/graph_nodes.csv
outputs/graph_edges.csv
outputs/flat_rag_results.csv
outputs/graph_rag_results.csv
outputs/comparison_results.csv
```

---

## 19. Lệnh chạy cuối

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Sau đó điền OPENAI_API_KEY vào .env
python main.py
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Sau đó điền OPENAI_API_KEY vào .env
python main.py
```

---

## 20. Tiêu chí hoàn thành

Project hoàn thành khi:

- Dùng OpenAI API thật để extract triples.
- Dùng OpenAI API thật để generate answer.
- Không mock answer.
- Không hard-code triples.
- Chạy được `python main.py`.
- Có graph bằng NetworkX.
- Có ảnh `outputs/graph.png`.
- Có bảng `outputs/comparison_results.csv`.
- Demo được ít nhất 10 câu hỏi benchmark.

---

## 21. Ghi chú cho coding agent

Làm đơn giản, nhưng phải chạy thật.

Ưu tiên:

```text
1. End-to-end chạy được.
2. OpenAI extract triples thật từ Wikipedia chunks.
3. Build graph bằng NetworkX.
4. GraphRAG dùng BFS 2-hop.
5. Flat RAG dùng TF-IDF + OpenAI answer.
6. Xuất graph.png và comparison_results.csv.
```

Không cần UI.  
Không cần Neo4j.  
Không cần NodeRAG.  
Không cần làm quá phức tạp phần đánh giá.  
