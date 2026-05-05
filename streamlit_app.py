import ast
import json
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st

from src.config import (
    BENCHMARK_QUESTIONS_PATH,
    CHUNKS_PATH,
    COMPARISON_RESULTS_PATH,
    EVALUATION_METRICS_PATH,
    FLAT_RAG_RESULTS_PATH,
    GRAPH_EDGES_PATH,
    GRAPH_IMAGE_PATH,
    GRAPH_NODES_PATH,
    GRAPH_RAG_RESULTS_PATH,
    QUESTION_METRICS_PATH,
    RAW_ARTICLES_PATH,
    RELATION_COUNTS_PATH,
    SOURCE_COVERAGE_PATH,
    TRIPLES_PATH,
)
from src.metrics import generate_metrics


st.set_page_config(
    page_title="Dashboard GraphRAG Wikipedia",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _json(path: Path):
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_data() -> dict:
    if not EVALUATION_METRICS_PATH.exists():
        generate_metrics()

    def read_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    return {
        "articles": _json(RAW_ARTICLES_PATH),
        "benchmark": read_csv(BENCHMARK_QUESTIONS_PATH),
        "chunks": _json(CHUNKS_PATH),
        "triples": _json(TRIPLES_PATH),
        "metrics": json.loads(EVALUATION_METRICS_PATH.read_text(encoding="utf-8")),
        "nodes": read_csv(GRAPH_NODES_PATH),
        "edges": read_csv(GRAPH_EDGES_PATH),
        "comparison": read_csv(COMPARISON_RESULTS_PATH),
        "flat": read_csv(FLAT_RAG_RESULTS_PATH),
        "graph": read_csv(GRAPH_RAG_RESULTS_PATH),
        "question_metrics": read_csv(QUESTION_METRICS_PATH),
        "relations": read_csv(RELATION_COUNTS_PATH),
        "sources": read_csv(SOURCE_COVERAGE_PATH),
    }


def parse_list(value) -> list:
    if isinstance(value, list):
        return value
    if value is None or pd.isna(value):
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            pass
    return []


def build_graph(edges_df: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    if edges_df.empty:
        return graph
    for _, row in edges_df.iterrows():
        graph.add_edge(
            str(row["subject"]),
            str(row["object"]),
            relation=str(row["relation"]),
            evidence=str(row.get("evidence", "")),
            source_title=str(row.get("source_title", "")),
        )
    return graph


def graph_to_dot(graph: nx.DiGraph, focus: str | None, max_edges: int) -> str:
    rows = ["digraph G {", "rankdir=LR;", 'graph [bgcolor="transparent"];']
    rows.append('node [shape=box, style="rounded,filled", fillcolor="#E7F5FF", color="#1864AB", fontname="Helvetica"];')
    rows.append('edge [color="#495057", fontname="Helvetica", fontsize=10];')
    edges = list(graph.edges(data=True))[:max_edges]
    for source, target, attrs in edges:
        source_label = str(source).replace('"', '\\"')
        target_label = str(target).replace('"', '\\"')
        relation = str(attrs.get("relation", "")).replace('"', '\\"')
        if focus and (str(source) == focus or str(target) == focus):
            rows.append(f'"{source_label}" [fillcolor="#FFF3BF", color="#F08C00"];')
            rows.append(f'"{target_label}" [fillcolor="#FFF3BF", color="#F08C00"];')
            rows.append(f'"{source_label}" -> "{target_label}" [label="{relation}", color="#F08C00", penwidth=2];')
        else:
            rows.append(f'"{source_label}" -> "{target_label}" [label="{relation}"];')
    rows.append("}")
    return "\n".join(rows)


def make_subgraph(edges_df: pd.DataFrame, nodes_df: pd.DataFrame, focus: str, hops: int, top_n: int):
    graph = build_graph(edges_df)
    if focus and focus in graph:
        visited = {focus}
        frontier = {focus}
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(graph.successors(node))
                next_frontier.update(graph.predecessors(node))
            next_frontier -= visited
            visited.update(next_frontier)
            frontier = next_frontier
        return graph.subgraph(visited).copy()

    top_nodes = set(nodes_df.head(top_n)["node"].astype(str).tolist()) if not nodes_df.empty else set()
    if not top_nodes:
        return nx.DiGraph()
    return graph.subgraph(top_nodes).copy()


def metric_card(label: str, value, help_text: str | None = None):
    st.metric(label, value, help=help_text)


def show_header(data: dict):
    st.title("Dashboard GraphRAG Wikipedia")
    st.caption(
        "Dashboard local để xem dữ liệu Wikipedia, triples, graph NetworkX, so sánh Flat RAG với GraphRAG và giải thích từng bước xử lý."
    )

    metrics = data["metrics"]
    cols = st.columns(8)
    with cols[0]:
        metric_card("Bài Wiki", metrics["corpus"]["articles"], "Số bài Wikipedia đã tải về.")
    with cols[1]:
        metric_card("Chunks", metrics["corpus"]["chunks"], "Số đoạn văn bản sau khi chia nhỏ bài Wikipedia.")
    with cols[2]:
        metric_card("Triples", metrics["corpus"]["triples"], "Số quan hệ dạng subject - relation - object do OpenAI trích xuất.")
    with cols[3]:
        metric_card("Nodes", metrics["graph"]["nodes"], "Số thực thể trong graph, ví dụ OpenAI, Microsoft, Linux.")
    with cols[4]:
        metric_card("Edges", metrics["graph"]["edges"], "Số cạnh/quan hệ nối giữa các node.")
    with cols[5]:
        metric_card("Density", metrics["graph"]["density"], "Độ dày của graph. Càng cao nghĩa là các node càng kết nối nhiều.")
    with cols[6]:
        metric_card("Flat trả lời được", f"{metrics['rag']['flat_answerable_rate']:.0%}")
    with cols[7]:
        metric_card("Graph trả lời được", f"{metrics['rag']['graph_answerable_rate']:.0%}")

    benchmark_count = len(data["benchmark"])
    comparison_count = len(data["comparison"])
    if benchmark_count and comparison_count and benchmark_count != comparison_count:
        st.warning(
            f"Benchmark hiện có {benchmark_count} câu, nhưng kết quả evaluation hiện có {comparison_count} câu. "
            "Bạn đã đổi bộ câu hỏi, cần chạy lại `python main.py` để cập nhật kết quả Flat RAG, GraphRAG và metrics."
        )


def overview_tab(data: dict):
    metrics = data["metrics"]
    st.subheader("Tổng quan kết quả")
    st.markdown(
        """
        Mỗi ô chỉ số phía trên là một **tile/card**: một khung nhỏ dùng để hiển thị nhanh một con số quan trọng.
        Ví dụ `Nodes = 613` nghĩa là graph có 613 thực thể; `Edges = 586` nghĩa là có 586 quan hệ giữa các thực thể.
        """
    )
    left, mid, right = st.columns([1, 1, 1])

    with left:
        st.subheader("Phương pháp thắng")
        winner_counts = pd.Series(metrics["rag"]["winner_counts"], name="count")
        st.bar_chart(winner_counts)
        st.dataframe(
            winner_counts.reset_index().rename(columns={"index": "method"}),
            hide_index=True,
            use_container_width=True,
        )

    with mid:
        st.subheader("Quan hệ xuất hiện nhiều")
        relations = data["relations"].head(15)
        st.bar_chart(relations.set_index("relation")["edge_count"])
        st.dataframe(relations, hide_index=True, use_container_width=True)

    with right:
        st.subheader("Node quan trọng")
        nodes = data["nodes"].head(15)
        st.bar_chart(nodes.set_index("node")["degree"])
        st.dataframe(nodes, hide_index=True, use_container_width=True)

    st.subheader("Đọc nhanh kết quả")
    st.info(
        f"Graph hiện có {metrics['graph']['nodes']} node và {metrics['graph']['edges']} edge. "
        f"Thành phần liên thông lớn nhất chiếm {metrics['graph']['largest_component_share']:.0%} số node, "
        "nghĩa là phần lớn thực thể đang nằm trong một cụm quan hệ lớn. "
        f"Trong benchmark, Flat RAG trả lời đủ ngữ cảnh {metrics['rag']['flat_answerable_rate']:.0%}, "
        f"còn GraphRAG trả lời đủ ngữ cảnh {metrics['rag']['graph_answerable_rate']:.0%}. "
        "Kết quả này cho thấy graph hiện chưa phủ đều mọi chủ đề vì mới extract triples từ một phần chunks."
    )


def graph_tab(data: dict):
    st.subheader("Khám phá input graph")
    st.markdown(
        """
        Graph là mạng lưới tri thức được tạo từ triples.  
        **Node** là thực thể, ví dụ `OpenAI`. **Edge** là quan hệ, ví dụ `OpenAI - FOUNDED_BY -> Sam Altman`.
        Chọn một node để xem các quan hệ xung quanh nó trong phạm vi `hop`.
        """
    )
    nodes_df = data["nodes"]
    edges_df = data["edges"]

    col_a, col_b, col_c = st.columns([1.2, 1, 1])
    with col_a:
        node_options = [""] + nodes_df["node"].astype(str).head(250).tolist()
        focus = st.selectbox("Node trung tâm", node_options, index=0)
    with col_b:
        hops = st.slider("Độ sâu hop", 1, 3, 2, help="1-hop là hàng xóm trực tiếp. 2-hop là hàng xóm của hàng xóm.")
    with col_c:
        top_n = st.slider("Top node khi chưa chọn", 20, 160, 80, 10)

    subgraph = make_subgraph(edges_df, nodes_df, focus, hops, top_n)
    st.caption(
        f"Đang hiển thị {subgraph.number_of_nodes()} node và {subgraph.number_of_edges()} edge. "
        "Biểu đồ được giới hạn số edge để dễ nhìn."
    )
    if subgraph.number_of_edges():
        st.graphviz_chart(graph_to_dot(subgraph, focus or None, max_edges=160), use_container_width=True)
    else:
        st.warning("Không có edge cho node hoặc bộ lọc hiện tại.")

    edge_rows = [
        {
            "subject": source,
            "relation": attrs.get("relation", ""),
            "object": target,
            "source_title": attrs.get("source_title", ""),
            "evidence": attrs.get("evidence", ""),
        }
        for source, target, attrs in subgraph.edges(data=True)
    ]
    st.dataframe(pd.DataFrame(edge_rows), hide_index=True, use_container_width=True, height=340)

    with st.expander("Static graph.png from NetworkX"):
        if GRAPH_IMAGE_PATH.exists():
            st.image(str(GRAPH_IMAGE_PATH), use_container_width=True)
        else:
            st.warning("outputs/graph.png chưa tồn tại.")


def benchmark_tab(data: dict):
    st.subheader("So sánh Flat RAG và GraphRAG")
    st.markdown(
        """
        **Flat RAG** tìm các đoạn text liên quan bằng TF-IDF rồi trả lời từ text.  
        **GraphRAG** tìm thực thể trong câu hỏi, duyệt graph bằng BFS theo `k-hop`, rồi trả lời từ các quan hệ tìm được.
        """
    )
    comparison = data["comparison"]
    graph_results = data["graph"]
    flat_results = data["flat"]
    question_metrics = data["question_metrics"]

    filter_cols = st.columns([1, 1, 2])
    with filter_cols[0]:
        winners = ["All"] + sorted(comparison["winner"].dropna().unique().tolist())
        winner_filter = st.selectbox("Phương pháp thắng", winners)
    with filter_cols[1]:
        types = ["All"] + sorted(comparison["expected_type"].dropna().unique().tolist())
        type_filter = st.selectbox("Loại câu hỏi", types)
    with filter_cols[2]:
        query = st.text_input("Tìm câu hỏi")

    filtered = comparison.copy()
    if winner_filter != "All":
        filtered = filtered[filtered["winner"] == winner_filter]
    if type_filter != "All":
        filtered = filtered[filtered["expected_type"] == type_filter]
    if query:
        filtered = filtered[filtered["question"].str.contains(query, case=False, na=False)]

    question_labels = [
        f"{int(row.id)} - {row.question[:90]} ({row.winner})" for row in filtered.itertuples()
    ]
    if not question_labels:
        st.warning("Không có câu hỏi khớp bộ lọc.")
        return
    selected_label = st.selectbox("Câu hỏi", question_labels)
    selected_id = int(selected_label.split(" - ", 1)[0])

    selected = comparison[comparison["id"] == selected_id].iloc[0]
    graph_row = graph_results[graph_results["id"] == selected_id].iloc[0]
    flat_row = flat_results[flat_results["id"] == selected_id].iloc[0]
    metric_row = question_metrics[question_metrics["id"] == selected_id].iloc[0]

    st.markdown(f"### {selected['question']}")
    badge_cols = st.columns(5)
    badge_cols[0].metric("Loại câu hỏi", selected["expected_type"])
    badge_cols[1].metric("Thắng", selected["winner"])
    badge_cols[2].metric("Seed entity", graph_row["seed_entity"])
    badge_cols[3].metric("Chunks lấy được", int(metric_row["retrieved_chunk_count"]))
    badge_cols[4].metric("Edges dùng", int(metric_row["graph_edge_count"]))

    answer_cols = st.columns(2)
    with answer_cols[0]:
        st.markdown("#### Flat RAG")
        st.caption("Dùng TF-IDF lấy top-k chunks gần câu hỏi nhất, rồi gọi OpenAI trả lời từ text context.")
        st.write(selected["flat_rag_answer"])
        st.markdown("Chunks được lấy")
        st.code("\n".join(parse_list(flat_row["retrieved_chunks"])) or "[]")
    with answer_cols[1]:
        st.markdown("#### GraphRAG")
        st.caption("Tìm seed entity, duyệt BFS k-hop trên graph, chuyển edges thành context chữ, rồi gọi OpenAI trả lời.")
        st.write(selected["graph_rag_answer"])
        graph_edges = parse_list(graph_row["graph_edges"])
        st.markdown("Các quan hệ graph được dùng")
        st.code("\n".join(graph_edges[:80]) if graph_edges else "[]")

    st.markdown("#### Toàn bộ benchmark")
    show_cols = [
        "id",
        "question",
        "expected_type",
        "winner",
        "flat_insufficient",
        "graph_insufficient",
        "retrieved_chunk_count",
        "graph_edge_count",
        "seed_entity",
    ]
    st.dataframe(
        question_metrics[show_cols],
        hide_index=True,
        use_container_width=True,
        height=330,
    )


def corpus_tab(data: dict):
    st.subheader("Dữ liệu Wikipedia và độ phủ extraction")
    st.markdown(
        """
        Tab này cho biết dữ liệu đầu vào đến từ đâu và triples được trích xuất từ những bài nào.
        Nếu một bài có nhiều chunks nhưng ít triples, nghĩa là graph chưa phủ tốt bài đó.
        """
    )
    sources = data["sources"]
    articles = pd.DataFrame(
        [
            {
                "title": article.get("title", ""),
                "summary": article.get("summary", ""),
                "url": article.get("url", ""),
                "text_words": len(str(article.get("text", "")).split()),
            }
            for article in data["articles"]
        ]
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### Độ phủ từng bài")
        st.dataframe(sources, hide_index=True, use_container_width=True, height=420)
    with col2:
        st.markdown("#### Chunks và triples theo nguồn")
        if not sources.empty:
            chart_df = sources.set_index("source_title")[["chunks", "triples"]]
            st.bar_chart(chart_df)

    st.markdown("#### Bài Wikipedia gốc")
    selected_title = st.selectbox("Bài viết", articles["title"].tolist())
    row = articles[articles["title"] == selected_title].iloc[0]
    st.link_button("Mở trang Wikipedia", row["url"])
    st.caption(f"{int(row['text_words'])} từ")
    st.write(row["summary"])

    st.markdown("#### Triples đã trích xuất")
    triples_df = pd.DataFrame(data["triples"])
    source_filter = st.selectbox("Lọc triples theo nguồn", ["All"] + sorted(triples_df["source_title"].dropna().unique().tolist()))
    if source_filter != "All":
        triples_df = triples_df[triples_df["source_title"] == source_filter]
    st.dataframe(triples_df, hide_index=True, use_container_width=True, height=420)


def steps_tab():
    st.subheader("Giải thích pipeline từng bước")
    st.markdown(
        """
        Demo này làm đơn giản để dễ học, nhưng mỗi bước đều chạy trên dữ liệu thật.

        1. **Tải bài Wikipedia**
           - `src/collect_wikipedia.py`
           - Đọc danh sách chủ đề `WIKI_TOPICS` trong `src/config.py`.
           - Tải `title`, `summary`, `text`, `url`.
           - Lưu vào `data/raw_articles.json`.

        2. **Chia bài thành chunks**
           - `src/chunking.py`
           - Mỗi bài dài được chia thành các đoạn nhỏ gọi là `chunk`.
           - Config hiện tại: `CHUNK_SIZE_WORDS = 400`, `CHUNK_OVERLAP_WORDS = 50`.
           - Lưu vào `data/chunks.json`.

        3. **Trích xuất triples cho knowledge graph**
           - `src/extract_triples_openai.py`
           - Gửi từng chunk được chọn vào OpenAI.
           - Model trả JSON triples: `subject`, `relation`, `object`, `evidence`.
           - Ví dụ: `OpenAI - FOUNDED_BY -> Sam Altman`.
           - Lưu vào `data/triples.json`.
           - Config hiện tại giới hạn `MAX_CHUNKS_FOR_EXTRACTION = 40`, nên độ phủ giữa các chủ đề chưa đều.

        4. **Build graph bằng NetworkX**
           - `src/build_graph.py`
           - Tạo `nx.DiGraph()`.
           - Mỗi subject/object trở thành một node.
           - Mỗi triple trở thành một edge có hướng, kèm relation, evidence, chunk id, source title.
           - Lưu `outputs/graph_nodes.csv` và `outputs/graph_edges.csv`.

        5. **Vẽ graph**
           - `src/visualize_graph.py`
           - Vẽ ảnh tĩnh top-degree graph vào `outputs/graph.png`.
           - Streamlit bổ sung màn hình chọn node và xem subgraph tương tác hơn.

        6. **Flat RAG baseline**
           - `src/flat_rag.py`
           - Không dùng graph.
           - Dùng TF-IDF để tìm top-k chunks liên quan câu hỏi.
           - Gửi text context đã retrieve cho OpenAI trả lời.
           - Lưu `outputs/flat_rag_results.csv`.

        7. **GraphRAG**
           - `src/graph_rag.py`
           - Tìm seed entity trong câu hỏi.
           - Duyệt BFS k-hop trên graph.
           - Chuyển các edge tìm được thành context dạng text.
           - Gửi graph context cho OpenAI trả lời.
           - Lưu `outputs/graph_rag_results.csv`.

        8. **Đánh giá và dashboard**
           - `src/evaluate.py` tạo bảng so sánh Flat RAG vs GraphRAG.
           - `src/metrics.py` tính metrics corpus, graph, relation, source coverage, benchmark.
           - `streamlit_app.py` trình bày kết quả trong dashboard này.
        """
    )
    st.markdown("#### Cách hiểu kết quả hiện tại")
    st.warning(
        "Chất lượng GraphRAG phụ thuộc mạnh vào độ phủ triples và khả năng tìm seed entity. "
        "Vì hiện chỉ extract 40 chunks đầu, graph đang thiên về các chủ đề xuất hiện sớm như OpenAI và Microsoft. "
        "Do đó Flat RAG có thể thắng ở các câu hỏi mà text chunk có đủ thông tin nhưng graph chưa có triples tương ứng."
    )


def glossary_tab():
    st.subheader("Giải thích thuật ngữ")
    st.markdown(
        """
        - **Tile/Card chỉ số**: ô nhỏ trên dashboard hiển thị nhanh một con số quan trọng, ví dụ số node hoặc số triples.
        - **Wikipedia article**: một bài Wikipedia gốc được tải về làm nguồn tri thức.
        - **Chunk**: một đoạn nhỏ cắt ra từ bài Wikipedia. RAG thường dùng chunk vì model không nên đọc cả bài rất dài cùng lúc.
        - **Triple**: một sự thật dạng `subject - relation - object`. Ví dụ `Microsoft - INVESTED_IN - OpenAI`.
        - **Evidence**: câu hoặc đoạn trong Wikipedia chứng minh triple đó.
        - **Node**: một thực thể trong graph, ví dụ `OpenAI`, `Microsoft`, `Linux`.
        - **Edge**: một quan hệ nối hai node, ví dụ `OpenAI -> Microsoft` với relation `PART_OF`.
        - **Degree**: số quan hệ đi vào/đi ra của một node. Degree cao thường nghĩa là node quan trọng hoặc xuất hiện nhiều.
        - **Density**: độ dày của graph. Graph càng dense thì các node càng có nhiều kết nối với nhau.
        - **Weak component**: một cụm node có thể nối với nhau nếu bỏ qua chiều mũi tên của edge.
        - **Seed entity**: thực thể chính được phát hiện từ câu hỏi. Ví dụ câu hỏi có `OpenAI` thì seed entity là `OpenAI`.
        - **Hop**: một bước đi trên graph. 1-hop là hàng xóm trực tiếp; 2-hop là hàng xóm của hàng xóm.
        - **BFS**: thuật toán duyệt graph theo từng lớp. GraphRAG dùng BFS để lấy các node/edge gần seed entity.
        - **Flat RAG**: RAG chỉ dùng text chunks, không dùng graph. Cách này đơn giản và thường mạnh khi text retrieval tốt.
        - **GraphRAG**: RAG dùng graph context. Cách này hữu ích khi cần giải thích quan hệ giữa thực thể hoặc suy luận nhiều bước.
        - **Answerable rate**: tỷ lệ câu hỏi mà phương pháp trả lời được thay vì nói thiếu ngữ cảnh.
        - **Winner**: phương pháp được đánh giá là trả lời tốt hơn trong bảng benchmark.
        """
    )


def raw_data_tab(data: dict):
    st.subheader("File output thô")
    file_options = {
        "comparison_results.csv": data["comparison"],
        "flat_rag_results.csv": data["flat"],
        "graph_rag_results.csv": data["graph"],
        "graph_nodes.csv": data["nodes"],
        "graph_edges.csv": data["edges"],
        "question_metrics.csv": data["question_metrics"],
        "relation_counts.csv": data["relations"],
        "source_coverage.csv": data["sources"],
    }
    selected = st.selectbox("File", list(file_options))
    df = file_options[selected]
    search = st.text_input("Tìm trong bảng")
    if search and not df.empty:
        mask = df.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        df = df[mask]
    st.dataframe(df, hide_index=True, use_container_width=True, height=650)
    st.download_button(
        "Tải bảng hiện tại dạng CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=selected,
        mime="text/csv",
    )


def main():
    data = load_data()
    with st.sidebar:
        st.header("Điều khiển")
        if st.button("Tính lại metrics"):
            generate_metrics()
            st.cache_data.clear()
            st.rerun()
        st.caption("App chỉ đọc outputs hiện có. App không gọi OpenAI trừ khi bạn chạy lại `main.py`.")
        st.divider()
        st.markdown("Chạy lại pipeline:")
        st.code("source .venv/bin/activate\npython main.py")

    show_header(data)
    tabs = st.tabs(
        [
            "Tổng quan",
            "Graph Explorer",
            "Benchmark",
            "Dữ liệu & Triples",
            "Các bước pipeline",
            "Thuật ngữ",
            "Raw Data",
        ]
    )
    with tabs[0]:
        overview_tab(data)
    with tabs[1]:
        graph_tab(data)
    with tabs[2]:
        benchmark_tab(data)
    with tabs[3]:
        corpus_tab(data)
    with tabs[4]:
        steps_tab()
    with tabs[5]:
        glossary_tab()
    with tabs[6]:
        raw_data_tab(data)


if __name__ == "__main__":
    main()
