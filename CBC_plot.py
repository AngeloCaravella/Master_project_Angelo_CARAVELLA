# --- FILE: plot_cbc_tree.py ---
import re
import networkx as nx
import matplotlib.pyplot as plt

# --- 1. Leggi il log CBC ---
logPath=r'C:\Users\angel\OneDrive\Desktop\Project_Master\cbc_log.txt'

pattern = r'Node\s+(\d+)\s+Depth\s+(\d+)'
nodes = []

with open(logfile, 'r') as f:
    for line in f:
        match = re.search(pattern, line)
        if match:
            node_id = int(match.group(1))
            depth = int(match.group(2))
            nodes.append((node_id, depth))

if not nodes:
    print("⚠️ Nessun nodo trovato nel log! Verifica che CBC sia partito con msg=1 e logLevel>1.")
    exit()

# --- 2. Crea grafo approssimato ---
G = nx.DiGraph()
for node_id, depth in nodes:
    if depth > 0:
        # collega al primo nodo precedente di profondità minore
        parent = next((n for n, d in reversed(nodes) if d == depth - 1 and n < node_id), None)
        if parent is not None:
            G.add_edge(parent, node_id)

# --- 3. Disegna ---
plt.figure(figsize=(10, 6))
pos = nx.multipartite_layout(G, subset_key=lambda n: dict(nodes)[n])  # layout per profondità
nx.draw(
    G, pos,
    with_labels=True,
    node_color="skyblue",
    node_size=600,
    font_size=8,
    arrows=False
)
plt.title("Albero approssimato di Branch-and-Bound (CBC Log)")
plt.show()
