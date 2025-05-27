import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ================== CẤU HÌNH =====================
jsonl_path = "processed/hotel.jsonl"
kg_vector_path = "embeddings/kg_vectors.npy"
output_embedding_path = "processed/hotel_embeddings.pt"

bert_model_name = "vinai/phobert-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================================

# Load PhoBERT model và tokenizer
tokenizer = AutoTokenizer.from_pretrained(bert_model_name, use_fast=False)
bert = AutoModel.from_pretrained(bert_model_name).to(device)
bert.eval()

# Load vector từ Knowledge Graph
kg_vectors = np.load(kg_vector_path, allow_pickle=True).item()
dim_kg = len(next(iter(kg_vectors.values())))

def get_kg_embedding(token, dim=dim_kg):
    token = token.replace("▁", "").lower()
    return torch.tensor(kg_vectors.get(token, np.zeros(dim)), dtype=torch.float)

# Kết hợp BERT + KG bằng phép nối
def combine_bert_kg(bert_embeds, tokens):
    combined_embeds = []
    for i, token in enumerate(tokens):
        kg_embed = get_kg_embedding(token).to(bert_embeds.device)
        combined = torch.cat([bert_embeds[i], kg_embed], dim=-1)
        combined_embeds.append(combined)
    return torch.stack(combined_embeds)

def embed_one_example(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = bert(**inputs)
        last_hidden = outputs.last_hidden_state.squeeze(0)  # [seq_len, 768]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))
    return combine_bert_kg(last_hidden, tokens)

if __name__ == "__main__":
    print("\n\nTạo nhúng kết hợp PhoBERT + KG cho toàn bộ corpus...")
    all_embeddings = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Generating embeddings"):
            obj = json.loads(line)
            text = obj["text"]
            emb = embed_one_example(text)
            all_embeddings.append(emb.cpu())

    torch.save(all_embeddings, output_embedding_path)
    print(f"Đã lưu nhúng vào: {output_embedding_path}")
    