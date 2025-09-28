# Required libraries
# Make sure to install them with:
# pip install PyMuPDF bibtexparser networkx scikit-learn wordcloud matplotlib
import os
import fitz  # PyMuPDF to extract text from PDFs
import bibtexparser
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ==============================
# 1. Utility functions
# ==============================
def extract_text_from_pdf(pdf_path):
    """Extracts all text from a PDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading file {pdf_path}: {e}")
        return ""  # Return an empty string in case of error

def extract_keywords_tfidf(documents, top_k=10):
    """Extracts keywords from each document using TF-IDF."""
    # Filter out empty documents to avoid errors
    doc_items = {k: v for k, v in documents.items() if v.strip()}
    if not doc_items:
        return {}  # Return empty dict if no valid documents
    
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(doc_items.values())
    feature_names = vectorizer.get_feature_names_out()
    
    keywords_per_doc = {}
    doc_keys = list(doc_items.keys())
    for i, doc in enumerate(doc_keys):
        tfidf_scores = X[i].toarray()[0]
        top_indices = tfidf_scores.argsort()[-top_k:][::-1]
        keywords = [feature_names[idx] for idx in top_indices if tfidf_scores[idx] > 0]
        keywords_per_doc[doc] = keywords
    return keywords_per_doc

# ==============================
# 2. Load documents
# ==============================
# !!! IMPORTANT: Update this path to your folder !!!
pdf_folder = r"C:\Users\angel\OneDrive\Desktop\Project_Master\Papers_read"
references_bib = "references.bib"  # optional

documents = {}

# Load PDFs
if os.path.exists(pdf_folder):
    print(f"Loading PDFs from folder: {pdf_folder}...")
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            path = os.path.join(pdf_folder, file)
            documents[file] = extract_text_from_pdf(path)
    print(f"Found {len(documents)} PDF documents.")

# Load from .bib (optional)
if os.path.exists(references_bib):
    print(f"Loading from .bib file: {references_bib}...")
    with open(references_bib) as bib_file:
        bib_db = bibtexparser.load(bib_file)
    for entry in bib_db.entries:
        title = entry.get("title", "NoTitle")
        abstract = entry.get("abstract", "")
        documents[title] = abstract if abstract else title
    print("Bib file loading completed.")

# ==============================
# 3. Create similarity graph
# ==============================
print("Creating similarity graph...")
G = nx.DiGraph()
G.add_nodes_from(documents.keys())

# Approach: graph based on text similarity
docs_text = list(documents.values())
vectorizer = TfidfVectorizer(stop_words='english', min_df=2)  # min_df ignores very rare terms
X = vectorizer.fit_transform(docs_text)
sim_matrix = cosine_similarity(X)

# Add edges if similarity > threshold
similarity_threshold = 0.2
doc_keys = list(documents.keys())
for i in range(len(doc_keys)):
    for j in range(len(doc_keys)):
        if i != j and sim_matrix[i,j] > similarity_threshold:
            G.add_edge(doc_keys[i], doc_keys[j], weight=sim_matrix[i,j])
print("Graph created.")

# ==============================
# 4. Compute PageRank
# ==============================
print("Computing PageRank...")
pr = nx.pagerank(G, weight='weight')
print("PageRank computed.")

# ==============================
# 5. Extract keywords
# ==============================
print("Extracting keywords with TF-IDF...")
keywords_per_doc = extract_keywords_tfidf(documents, top_k=10)
print("Keywords extracted.")

# ==============================
# 6. Generate PageRank-weighted Word Cloud
# ==============================
print("Generating Word Cloud...")

keyword_aggregated_scores = {}
for doc, score in pr.items():
    keywords = keywords_per_doc.get(doc, [])
    for kw in keywords:
        keyword_aggregated_scores[kw] = keyword_aggregated_scores.get(kw, 0.0) + score

if keyword_aggregated_scores:
    wordcloud = WordCloud(width=1000, height=500, background_color='white', collocations=False).generate_from_frequencies(keyword_aggregated_scores)
    plt.figure(figsize=(15,7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    print("Word Cloud displayed.")
else:
    print("No keywords found to generate the Word Cloud.")

# ==============================
# 7. Visualize and save the graph
# ==============================
print("Generating graph plot...")

if G.number_of_nodes() > 0:
    plt.figure(figsize=(18, 18))
    
    # Choose a layout for better visualization
    pos = nx.spring_layout(G, k=0.5, iterations=50) 

    # Size nodes according to their PageRank to highlight the most important
    node_sizes = [v * 10000 for v in pr.values()]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray', arrowsize=10)

    # Draw labels (file names)
    # To avoid overlap, you may show only the most important node labels
    # In this example, we show all with smaller font
    labels = {node: node.replace('.pdf', '')[:20] + '...' if len(node) > 20 else node.replace('.pdf', '') for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black')

    plt.title("Document Similarity Graph (Node size ~ PageRank)", size=15)
    plt.axis('off')  # Hide axes
    
    # Save the graph as PNG
    try:
        plt.savefig("similarity_graph.png", format="PNG", dpi=300, bbox_inches='tight')
        print("Graph saved as 'similarity_graph.png'")
    except Exception as e:
        print(f"Error saving graph: {e}")

    plt.show()  # Display the graph
else:
    print("Graph is empty, cannot generate plot.")
