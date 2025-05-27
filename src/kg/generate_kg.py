import networkx as nx
import numpy as np
import os

graphml_path = "data_kg/hotel.graphml"
output_path = "embeddings/kg_vectors.npy"
embedding_dim = 200  # hoặc 300 tùy chọn

def generate_kg_vectors():
    print("Đang tạo embedding KG từ .graphml...")

    # Đọc đồ thị từ graphml
    G = nx.read_graphml(graphml_path, force_multigraph=True)

    # Chỉ chọn các node là term::
    term_nodes = [n for n in G.nodes if n.startswith("term::")]

    # Tạo embedding cho từng term
    embeddings = {}
    for node in term_nodes:
        term = node.replace("term::", "").lower()
        embeddings[term] = np.random.uniform(-0.25, 0.25, embedding_dim).astype(np.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Đã lưu xong: {output_path} ({len(embeddings)} từ khóa)")

if __name__ == "__main__":
    print("\nTạo embedding KG từ .graphml...")
    generate_kg_vectors()

    print("\n\n")
