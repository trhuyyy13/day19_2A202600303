# GraphRAG Wikipedia Demo

Demo GraphRAG tối giản dùng Wikipedia, NetworkX và OpenAI API.

## Luồng tổng thể: từ đầu tới cuối

Pipeline của project này đi theo thứ tự sau:

1. Tải dữ liệu Wikipedia về máy.
2. Chia bài viết thành các đoạn nhỏ.
3. Dùng OpenAI để trích xuất triple dạng `subject - relation - object`.
4. Dựng graph từ các triple bằng NetworkX.
5. Chạy 2 kiểu trả lời:
	 - Flat RAG: tìm chunk liên quan trực tiếp từ text.
	 - GraphRAG: tìm entity trong graph rồi đi theo các cạnh lân cận.
6. So sánh kết quả trên bộ benchmark.
7. Tính metrics và xuất dashboard để xem trực quan.

Nói ngắn gọn: project này lấy dữ liệu Wikipedia, biến text thành graph tri thức, rồi dùng graph đó để trả lời câu hỏi khó theo nhiều bước liên kết.

## Từng file code làm gì

### `main.py`

File điều phối chính. Khi chạy `python main.py`, chương trình sẽ chạy lần lượt toàn bộ pipeline:

- đảm bảo thư mục output tồn tại
- lấy Wikipedia nếu chưa có dữ liệu
- chia chunk nếu chưa có chunk
- trích triple nếu chưa có triple
- build graph
- vẽ graph
- chạy benchmark so sánh Flat RAG vs GraphRAG
- tạo dashboard HTML

Đây là điểm bắt đầu của toàn bộ project.

### `src/config.py`

File cấu hình trung tâm.

Chứa:

- đường dẫn tới file dữ liệu và file output
- danh sách topic Wikipedia cần lấy
- tham số như `CHUNK_SIZE_WORDS`, `CHUNK_OVERLAP_WORDS`, `TOP_K_CHUNKS`, `GRAPH_HOP_K`
- hàm `ensure_dirs()` để tạo thư mục cần thiết
- hàm `load_settings()` để đọc `OPENAI_API_KEY` và model từ `.env`

File này giúp các phần khác dùng chung cấu hình, không bị hard-code rải rác.

### `src/collect_wikipedia.py`

Lấy dữ liệu raw từ Wikipedia.

Luồng:

- tạo client Wikipedia
- duyệt từng topic trong `WIKI_TOPICS`
- tải `title`, `summary`, `text`, `url`
- bỏ qua page không tồn tại hoặc rỗng
- lưu toàn bộ bài viết vào `data/raw_articles.json`

Kết quả: có bộ bài Wikipedia thô để xử lý tiếp.

### `src/chunking.py`

Chia bài Wikipedia thành nhiều đoạn nhỏ hơn.

Luồng:

- đọc `raw_articles.json`
- tách text thành danh sách từ
- dùng cửa sổ trượt theo `CHUNK_SIZE_WORDS`
- chồng lấn một phần theo `CHUNK_OVERLAP_WORDS`
- bỏ các đoạn quá ngắn
- lưu ra `data/chunks.json`

Mục đích: giúp retrieval dễ hơn, vì LLM và TF-IDF làm việc tốt hơn với đoạn văn ngắn thay vì cả bài dài.

### `src/openai_utils.py`

Tiện ích làm việc với OpenAI.

Chức năng chính:

- tạo `OpenAI` client từ API key
- gọi chat completion qua `chat_completion()`
- parse JSON array từ phản hồi của model qua `parse_json_array()`

File này là lớp trung gian để các phần khác chỉ cần gọi một hàm đơn giản.

### `src/extract_triples_openai.py`

Trích xuất triple từ từng chunk bằng OpenAI.

Luồng:

- đọc `chunks.json`
- chỉ lấy tối đa `MAX_CHUNKS_FOR_EXTRACTION` chunk đầu tiên
- tạo prompt yêu cầu model trích xuất triple
- mỗi triple phải có `subject`, `relation`, `object`, `evidence`
- chuẩn hóa relation thành dạng in hoa, có dấu gạch dưới
- thêm thông tin nguồn như `chunk_id` và `source_title`
- lưu ra `data/triples.json`

Đây là bước biến text tự nhiên thành dữ liệu có cấu trúc để dựng graph.

### `src/build_graph.py`

Dựng graph tri thức từ triples.

Luồng:

- đọc `triples.json`
- tạo graph có hướng bằng NetworkX
- mỗi `subject` và `object` trở thành một node
- mỗi triple trở thành một edge có metadata như relation, evidence, chunk_id, source_title
- xuất danh sách node sang `outputs/graph_nodes.csv`
- xuất danh sách edge sang `outputs/graph_edges.csv`

Kết quả: ta có một knowledge graph để truy vấn theo quan hệ.

### `src/visualize_graph.py`

Vẽ graph ra ảnh tĩnh.

Luồng:

- nếu graph quá lớn, chỉ lấy top node theo degree để dễ nhìn
- dùng `spring_layout` để sắp xếp node
- vẽ node, edge, label quan hệ
- lưu ảnh vào `outputs/graph.png`

File này phục vụ phần quan sát trực quan, không ảnh hưởng đến logic trả lời câu hỏi.

### `src/flat_rag.py`

Đây là baseline Flat RAG.

Luồng:

- lấy text của tất cả chunk
- dùng TF-IDF để tính độ giống giữa câu hỏi và từng chunk
- chọn top-k chunk liên quan nhất
- ghép context từ các chunk đó
- gọi OpenAI để trả lời chỉ dựa trên text đã lấy

Ưu điểm: đơn giản, nhanh. Nhược điểm: khó xử lý câu hỏi nhiều bước liên kết.

### `src/graph_rag.py`

Đây là phần GraphRAG.

Luồng:

- tìm seed entity trong câu hỏi bằng `detect_seed_entity()`
- nếu tìm được, lấy subgraph theo `k-hop`
- biến các edge trong subgraph thành context dạng text
- gọi OpenAI để trả lời dựa trên graph context

Ý tưởng chính: không chỉ nhìn text gần câu hỏi, mà đi theo quan hệ giữa các entity để tìm đường nối thông tin.

### `src/evaluate.py`

Chạy benchmark so sánh Flat RAG và GraphRAG.

Luồng:

- tạo danh sách câu hỏi benchmark nếu file chưa tồn tại
- chạy cả 2 phương pháp trên từng câu hỏi
- chọn bên thắng dựa trên việc bên nào có đủ ngữ cảnh và trả lời tốt hơn
- lưu kết quả Flat RAG vào `outputs/flat_rag_results.csv`
- lưu kết quả GraphRAG vào `outputs/graph_rag_results.csv`
- lưu bảng so sánh vào `outputs/comparison_results.csv`

File này giúp kiểm tra GraphRAG có thực sự tốt hơn Flat RAG trong các câu hỏi multi-hop hay không.

### `src/metrics.py`

Tính các chỉ số tổng hợp cho corpus, graph và benchmark.

Luồng:

- đọc toàn bộ file data/output đã tạo
- đếm số bài, số chunk, số triple
- tính số node, edge, density, thành phần liên thông
- thống kê relation xuất hiện nhiều nhất
- tính coverage theo từng nguồn Wikipedia
- thống kê câu hỏi nào Flat RAG/GraphRAG trả lời được
- lưu metrics ra `outputs/evaluation_metrics.json`

Ngoài ra còn tạo:

- `outputs/relation_counts.csv`
- `outputs/source_coverage.csv`
- `outputs/question_metrics.csv`

### `src/dashboard.py`

Sinh dashboard HTML tĩnh.

Luồng:

- đọc metrics và các file CSV/JSON đã xuất
- gom dữ liệu thành một gói `dashboard_data`
- tự build một trang HTML có:
	- metric cards
	- graph tương tác bằng canvas
	- bảng benchmark
	- bảng edge
	- phần xem source coverage
- lưu ra `outputs/dashboard.html`

File này dùng khi muốn mở nhanh report dạng web mà không cần Streamlit.

### `streamlit_app.py`

Dashboard tương tác bằng Streamlit.

Luồng:

- đọc data từ các file output
- nếu metrics chưa có thì tự gọi `generate_metrics()`
- hiển thị các tab như:
	- Tổng quan
	- Graph Explorer
	- Benchmark
	- Dữ liệu & Triples
	- Các bước pipeline
	- Thuật ngữ
	- Raw Data

Đây là giao diện dễ dùng nhất để explore kết quả từng bước.

## Luồng chạy thực tế khi bấm `python main.py`

1. `main.py` gọi `ensure_dirs()`.
2. Nếu chưa có `raw_articles.json` thì gọi `collect_wikipedia_articles()`.
3. Nếu chưa có `chunks.json` thì gọi `chunk_articles()`.
4. Nếu chưa có `triples.json` thì gọi `extract_triples()`.
5. `build_graph()` dựng graph và xuất CSV.
6. `visualize_graph()` xuất ảnh graph.
7. `run_evaluation()` chạy benchmark Flat RAG và GraphRAG.
8. `generate_dashboard()` tạo file report HTML.
9. Kết thúc pipeline.

## Đầu vào và đầu ra chính

### Đầu vào

- Wikipedia topics trong `src/config.py`
- `OPENAI_API_KEY` trong `.env`

### Đầu ra

- `data/raw_articles.json`
- `data/chunks.json`
- `data/triples.json`
- `outputs/graph_nodes.csv`
- `outputs/graph_edges.csv`
- `outputs/graph.png`
- `outputs/flat_rag_results.csv`
- `outputs/graph_rag_results.csv`
- `outputs/comparison_results.csv`
- `outputs/evaluation_metrics.json`
- `outputs/dashboard.html`

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# điền OPENAI_API_KEY trong .env
python main.py
```

Outputs được ghi vào `data/` và `outputs/`.

## Streamlit Dashboard

Sau khi pipeline đã tạo outputs, mở dashboard local:

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

Dashboard gồm các tab:

- `Tổng quan`: chỉ số corpus, graph, phương pháp thắng Flat RAG vs GraphRAG.
- `Graph Explorer`: chọn node, hop depth và xem subgraph tương tác.
- `Benchmark`: xem từng câu hỏi, answer, retrieved chunks và graph path.
- `Dữ liệu & Triples`: xem bài Wikipedia, độ phủ chunks/triples, triples đã extract.
- `Các bước pipeline`: giải thích từng bước từ Wikipedia tới GraphRAG.
- `Thuật ngữ`: giải thích tile/card, chunk, triple, node, edge, hop, seed entity.
- `Raw Data`: xem/search/download các CSV output.
# day19_2A202600303
