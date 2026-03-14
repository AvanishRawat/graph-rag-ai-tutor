import os
import networkx as nx
from pyvis.network import Network
from project.graph.build_graph import GRAPH_PATH

OUT_HTML = os.path.join(os.path.dirname(__file__), "kg_view.html")

def to_pyvis(G=None, graph_path=GRAPH_PATH, out_html=OUT_HTML, notebook=False):
    if G is None:
        G = nx.read_gpickle(graph_path)

    net = Network(height="800px", width="100%", notebook=notebook)
    for n, d in G.nodes(data=True):
        ntype = d.get("type", "unknown")
        label = d.get("title") or (d.get("text")[:60] if d.get("text") else str(n))
        title = json_title(d)
        color = "#ffcc80" if ntype == "concept" else ("#90caf9" if ntype == "resource" else "#c8e6c9")
        net.add_node(n, label=label[:60], title=title, color=color)

    for u, v, d in G.edges(data=True):
        rel = d.get("relation", "")
        label = rel
        title = f"{rel} {d.get('similarity','')}"
        net.add_edge(u, v, title=title, label=label)

    net.show(out_html)
    print("Wrote:", out_html)

def json_title(d):
    import html, json
    short = {}
    for k in ("title", "type", "definitions", "aliases", "url", "text"):
        if k in d and d[k]:
            short[k] = d[k]
    return html.escape(json.dumps(short, indent=2, ensure_ascii=False))

