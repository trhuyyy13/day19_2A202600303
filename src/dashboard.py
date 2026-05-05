import json
from html import escape

import pandas as pd

from src.config import (
    COMPARISON_RESULTS_PATH,
    DASHBOARD_HTML_PATH,
    EVALUATION_METRICS_PATH,
    GRAPH_EDGES_PATH,
    GRAPH_IMAGE_PATH,
    GRAPH_NODES_PATH,
    GRAPH_RAG_RESULTS_PATH,
    QUESTION_METRICS_PATH,
    RELATION_COUNTS_PATH,
    SOURCE_COVERAGE_PATH,
    ensure_dirs,
)
from src.metrics import generate_metrics


def _read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _records(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    if limit is not None:
        df = df.head(limit)
    return json.loads(df.fillna("").to_json(orient="records"))


def _metric_cards(metrics: dict) -> str:
    cards = [
        ("Articles", metrics["corpus"]["articles"]),
        ("Chunks", metrics["corpus"]["chunks"]),
        ("Triples", metrics["corpus"]["triples"]),
        ("Nodes", metrics["graph"]["nodes"]),
        ("Edges", metrics["graph"]["edges"]),
        ("Density", metrics["graph"]["density"]),
        ("GraphRAG answerable", f"{metrics['rag']['graph_answerable_rate']:.0%}"),
        ("Seed exact match", f"{metrics['rag']['seed_exact_match_rate']:.0%}"),
    ]
    return "\n".join(
        f"<article class='metric-card'><span>{escape(label)}</span><strong>{escape(str(value))}</strong></article>"
        for label, value in cards
    )


def _html_table(records: list[dict], columns: list[str], limit: int = 20) -> str:
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    rows = []
    for record in records[:limit]:
        cells = "".join(
            f"<td>{escape(str(record.get(column, ''))[:500])}</td>" for column in columns
        )
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def generate_dashboard() -> None:
    ensure_dirs()
    metrics = generate_metrics()
    nodes_df = _read_csv(GRAPH_NODES_PATH)
    edges_df = _read_csv(GRAPH_EDGES_PATH)
    comparison_df = _read_csv(COMPARISON_RESULTS_PATH)
    graph_results_df = _read_csv(GRAPH_RAG_RESULTS_PATH)
    relation_counts_df = _read_csv(RELATION_COUNTS_PATH)
    source_df = _read_csv(SOURCE_COVERAGE_PATH)
    question_metrics_df = _read_csv(QUESTION_METRICS_PATH)

    top_nodes = _records(nodes_df, 120)
    top_node_names = set(nodes_df.head(80)["node"].astype(str).tolist()) if not nodes_df.empty else set()
    graph_edges = (
        edges_df[
            edges_df["subject"].astype(str).isin(top_node_names)
            | edges_df["object"].astype(str).isin(top_node_names)
        ].head(350)
        if not edges_df.empty
        else pd.DataFrame()
    )

    dashboard_data = {
        "nodes": top_nodes,
        "edges": _records(graph_edges),
        "relations": _records(relation_counts_df, 30),
        "sources": _records(source_df),
        "questions": _records(comparison_df),
        "questionMetrics": _records(question_metrics_df),
        "graphResults": _records(graph_results_df),
    }

    graph_image = GRAPH_IMAGE_PATH.name if GRAPH_IMAGE_PATH.exists() else ""
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GraphRAG Local Dashboard</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #667085;
      --line: #d8dee9;
      --accent: #1864ab;
      --accent-2: #0b7285;
      --good: #2b8a3e;
      --warn: #c92a2a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.45;
    }}
    header {{
      background: #102033;
      color: white;
      padding: 22px 28px;
    }}
    header h1 {{ margin: 0 0 6px; font-size: 26px; }}
    header p {{ margin: 0; color: #d5deeb; max-width: 980px; }}
    main {{ padding: 22px; max-width: 1500px; margin: 0 auto; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .metric-card, section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }}
    .metric-card {{ padding: 14px 16px; }}
    .metric-card span {{ display: block; color: var(--muted); font-size: 13px; }}
    .metric-card strong {{ display: block; font-size: 25px; margin-top: 4px; }}
    section {{ padding: 16px; margin-bottom: 18px; }}
    h2 {{ margin: 0 0 12px; font-size: 20px; }}
    h3 {{ margin: 14px 0 8px; font-size: 15px; }}
    .grid-2 {{
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(360px, 0.85fr);
      gap: 16px;
    }}
    .grid-3 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }}
    @media (max-width: 980px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
    .toolbar {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      margin-bottom: 12px;
    }}
    input, select {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
      background: white;
      min-height: 38px;
    }}
    input[type="range"] {{ min-height: 0; padding: 0; }}
    canvas {{
      width: 100%;
      height: 620px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfe;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 8px;
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 650; background: #fafbfc; position: sticky; top: 0; }}
    .table-wrap {{ overflow: auto; max-height: 420px; border: 1px solid var(--line); border-radius: 8px; }}
    .answer {{
      white-space: pre-wrap;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      max-height: 360px;
      overflow: auto;
      font-size: 14px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 12px;
      background: #e7f5ff;
      color: #0b4f71;
      margin: 2px 4px 2px 0;
    }}
    .note {{ color: var(--muted); font-size: 13px; }}
    .bar-row {{ display: grid; grid-template-columns: 150px 1fr 52px; align-items: center; gap: 8px; margin: 7px 0; }}
    .bar {{ height: 12px; background: #e9ecef; border-radius: 999px; overflow: hidden; }}
    .bar > span {{ display: block; height: 100%; background: var(--accent); }}
    .status-good {{ color: var(--good); font-weight: 650; }}
    .status-warn {{ color: var(--warn); font-weight: 650; }}
    img.graph-image {{ width: 100%; border: 1px solid var(--line); border-radius: 8px; background: white; }}
  </style>
</head>
<body>
  <header>
    <h1>GraphRAG Wikipedia Dashboard</h1>
    <p>Local report for corpus, extracted triples, NetworkX graph structure, Flat RAG vs GraphRAG benchmark, seed entities, graph paths, and input graph inspection.</p>
  </header>
  <main>
    <div class="metrics">
      {_metric_cards(metrics)}
    </div>

    <section class="grid-2">
      <div>
        <h2>Interactive Graph</h2>
        <div class="toolbar">
          <label>Search node <input id="nodeSearch" placeholder="OpenAI, Microsoft..." /></label>
          <label>Top nodes <input id="nodeLimit" type="range" min="20" max="120" value="70" /></label>
          <span id="nodeLimitLabel" class="note">70</span>
        </div>
        <canvas id="graphCanvas" width="1200" height="720"></canvas>
        <p class="note">Nodes are sized by degree. This view shows the highest-degree slice of the input graph to keep it readable.</p>
      </div>
      <div>
        <h2>Graph Summary</h2>
        <div id="winnerBars"></div>
        <h3>Top Relations</h3>
        <div id="relationBars"></div>
        <h3>Top Nodes</h3>
        <div class="table-wrap">{_html_table(top_nodes, ["node", "degree"], limit=18)}</div>
      </div>
    </section>

    <section>
      <h2>Benchmark Questions</h2>
      <div class="toolbar">
        <label>Question <select id="questionSelect"></select></label>
        <label>Method result <select id="winnerFilter"><option value="">All</option><option>GraphRAG</option><option>Flat RAG</option><option>Tie</option></select></label>
      </div>
      <div class="grid-2">
        <div>
          <h3>Question Detail</h3>
          <div id="questionDetail"></div>
        </div>
        <div>
          <h3>Per-question Metrics</h3>
          <div id="questionMetricDetail"></div>
        </div>
      </div>
    </section>

    <section class="grid-2">
      <div>
        <h2>Source Coverage</h2>
        <p class="note">Shows how much of each Wikipedia article is represented by chunks and extracted triples. Low triple coverage means the current `MAX_CHUNKS_FOR_EXTRACTION` limited extraction before reaching that topic.</p>
        <div class="table-wrap">{_html_table(_records(source_df), ["source_title", "chunks", "triples", "triples_per_chunk"], limit=30)}</div>
      </div>
      <div>
        <h2>Static Graph Image</h2>
        {"<img class='graph-image' src='" + escape(graph_image) + "' alt='Graph image' />" if graph_image else "<p class='note'>graph.png is not available yet.</p>"}
      </div>
    </section>

    <section>
      <h2>Input Graph Edges</h2>
      <div class="toolbar">
        <input id="edgeSearch" placeholder="Filter subject, relation, object, evidence..." />
      </div>
      <div class="table-wrap"><table id="edgeTable"></table></div>
    </section>
  </main>
  <script>
    const metrics = {json.dumps(metrics, ensure_ascii=False)};
    const data = {json.dumps(dashboard_data, ensure_ascii=False)};
  </script>
  <script>
    const $ = (id) => document.getElementById(id);

    function truncate(text, max = 260) {{
      text = String(text ?? "");
      return text.length > max ? text.slice(0, max) + "..." : text;
    }}

    function renderBars(containerId, rows, labelKey, valueKey, color = "#1864ab") {{
      const container = $(containerId);
      const max = Math.max(1, ...rows.map(row => Number(row[valueKey]) || 0));
      container.innerHTML = rows.map(row => {{
        const value = Number(row[valueKey]) || 0;
        const width = Math.max(3, value / max * 100);
        return `<div class="bar-row"><span>${{row[labelKey]}}</span><div class="bar"><span style="width:${{width}}%;background:${{color}}"></span></div><strong>${{value}}</strong></div>`;
      }}).join("");
    }}

    function renderWinnerBars() {{
      const rows = Object.entries(metrics.rag.winner_counts).map(([winner, count]) => ({{ winner, count }}));
      renderBars("winnerBars", rows, "winner", "count", "#0b7285");
    }}

    function populateQuestions() {{
      const filter = $("winnerFilter").value;
      const select = $("questionSelect");
      const rows = data.questions.filter(row => !filter || row.winner === filter);
      select.innerHTML = rows.map(row => `<option value="${{row.id}}">${{row.id}}. ${{truncate(row.question, 80)}} (${{row.winner}})</option>`).join("");
      renderQuestionDetail();
    }}

    function renderQuestionDetail() {{
      const id = Number($("questionSelect").value);
      const row = data.questions.find(item => Number(item.id) === id) || data.questions[0];
      const metric = data.questionMetrics.find(item => Number(item.id) === Number(row?.id)) || {{}};
      const graphResult = data.graphResults.find(item => Number(item.id) === Number(row?.id)) || {{}};
      if (!row) return;
      $("questionDetail").innerHTML = `
        <p><strong>${{row.question}}</strong></p>
        <p>
          <span class="pill">${{row.expected_type}}</span>
          <span class="pill">Winner: ${{row.winner}}</span>
          <span class="pill">Seed: ${{metric.seed_entity || graphResult.seed_entity || "N/A"}}</span>
        </p>
        <h3>Flat RAG Answer</h3>
        <div class="answer">${{row.flat_rag_answer}}</div>
        <h3>GraphRAG Answer</h3>
        <div class="answer">${{row.graph_rag_answer}}</div>
        <h3>Notes</h3>
        <p>${{row.notes}}</p>
      `;
      $("questionMetricDetail").innerHTML = `
        <table><tbody>
          <tr><th>Retrieved chunks</th><td>${{metric.retrieved_chunk_count ?? ""}}</td></tr>
          <tr><th>Graph edges used</th><td>${{metric.graph_edge_count ?? ""}}</td></tr>
          <tr><th>Seed exact match</th><td class="${{metric.seed_exact_match ? "status-good" : "status-warn"}}">${{metric.seed_exact_match}}</td></tr>
          <tr><th>Flat insufficient</th><td>${{metric.flat_insufficient}}</td></tr>
          <tr><th>Graph insufficient</th><td>${{metric.graph_insufficient}}</td></tr>
          <tr><th>Flat answer words</th><td>${{metric.flat_answer_words}}</td></tr>
          <tr><th>Graph answer words</th><td>${{metric.graph_answer_words}}</td></tr>
        </tbody></table>
        <h3>Graph Path Preview</h3>
        <div class="answer">${{truncate(graphResult.graph_edges || "[]", 1800)}}</div>
      `;
    }}

    function renderEdgeTable() {{
      const query = $("edgeSearch").value.toLowerCase();
      const rows = data.edges.filter(row => {{
        const text = `${{row.subject}} ${{row.relation}} ${{row.object}} ${{row.evidence}}`.toLowerCase();
        return !query || text.includes(query);
      }}).slice(0, 120);
      const cols = ["subject", "relation", "object", "evidence", "source_title"];
      $("edgeTable").innerHTML = `<thead><tr>${{cols.map(col => `<th>${{col}}</th>`).join("")}}</tr></thead><tbody>` +
        rows.map(row => `<tr>${{cols.map(col => `<td>${{truncate(row[col], col === "evidence" ? 180 : 80)}}</td>`).join("")}}</tr>`).join("") +
        "</tbody>";
    }}

    function drawGraph() {{
      const canvas = $("graphCanvas");
      const ctx = canvas.getContext("2d");
      const limit = Number($("nodeLimit").value);
      $("nodeLimitLabel").textContent = String(limit);
      const search = $("nodeSearch").value.toLowerCase();
      const nodes = data.nodes.slice(0, limit).map((node, index) => ({{
        ...node,
        x: canvas.width / 2 + Math.cos(index * 2.399) * (80 + index * 4.2),
        y: canvas.height / 2 + Math.sin(index * 2.399) * (65 + index * 3.1),
      }}));
      const nodeMap = new Map(nodes.map(node => [String(node.node), node]));
      const edges = data.edges.filter(edge => nodeMap.has(String(edge.subject)) && nodeMap.has(String(edge.object))).slice(0, 260);

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#fbfcfe";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.lineWidth = 1;
      edges.forEach(edge => {{
        const a = nodeMap.get(String(edge.subject));
        const b = nodeMap.get(String(edge.object));
        if (!a || !b) return;
        ctx.strokeStyle = "rgba(74, 85, 104, 0.22)";
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }});

      const maxDegree = Math.max(1, ...nodes.map(node => Number(node.degree) || 1));
      nodes.forEach(node => {{
        const degree = Number(node.degree) || 1;
        const radius = 5 + Math.sqrt(degree / maxDegree) * 24;
        const isHit = search && String(node.node).toLowerCase().includes(search);
        ctx.beginPath();
        ctx.fillStyle = isHit ? "#f08c00" : "#d0ebff";
        ctx.strokeStyle = isHit ? "#c92a2a" : "#1864ab";
        ctx.lineWidth = isHit ? 3 : 1.2;
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#172033";
        ctx.font = isHit ? "bold 14px system-ui" : "12px system-ui";
        ctx.textAlign = "center";
        ctx.fillText(truncate(node.node, 24), node.x, node.y + radius + 14);
      }});
    }}

    renderWinnerBars();
    renderBars("relationBars", data.relations.slice(0, 12), "relation", "edge_count", "#1864ab");
    populateQuestions();
    renderEdgeTable();
    drawGraph();

    $("winnerFilter").addEventListener("change", populateQuestions);
    $("questionSelect").addEventListener("change", renderQuestionDetail);
    $("edgeSearch").addEventListener("input", renderEdgeTable);
    $("nodeSearch").addEventListener("input", drawGraph);
    $("nodeLimit").addEventListener("input", drawGraph);
  </script>
</body>
</html>
"""
    DASHBOARD_HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Saved dashboard -> {DASHBOARD_HTML_PATH}")
