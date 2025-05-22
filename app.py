from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import numpy as np
import pandas as pd
import re
import os
import fitz  # PyMuPDF
from tqdm import tqdm
import pickle
from flask_cors import CORS

from dotenv import load_dotenv
load_dotenv()

# --- Config ---
client = OpenAI()
pdf_folder = "data/pdfs"
image_dir = "data/extracted_images"
cache_file = "data/chunk_cache.pkl"
os.makedirs(image_dir, exist_ok=True)
os.makedirs("data", exist_ok=True)

# --- App Init ---
app = Flask(__name__)
CORS(app)  # enable CORS for all routes and origins

model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Chunk Cleaning Helper ---


def is_valid(chunk):
    chunk = chunk.strip()
    if len(chunk) < 40:
        return False
    if re.match(r"^\s*(chapter|text|figure|table|section)\s*\d+", chunk.lower()):
        return False
    if not re.search(r"[.!?]", chunk):
        return False
    return True

# --- PDF Processing & Chunking ---


def load_chunks():
    if os.path.exists(cache_file):
        print("🔁 Loading chunks from cache...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("📦 Loading and processing PDFs...")
    all_text_data = []
    all_image_data = []

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    print(f"📚 Found {len(pdf_files)} PDFs in {pdf_folder}")

    for pdf_filename in tqdm(pdf_files):
        pdf_path = os.path.join(pdf_folder, pdf_filename)
        pdf_name = os.path.splitext(pdf_filename)[0]
        doc = fitz.open(pdf_path)
        print(f"\n📄 Processing {pdf_filename} ({len(doc)} pages)")

        # Extract raw text
        raw_text_data = []
        for i, page in enumerate(doc):
            blocks = page.get_text("blocks")
            for block in blocks:
                x0, y0, x1, y1, text, *_ = block
                if text.strip():
                    raw_text_data.append({
                        "pdf_name": pdf_name,
                        "page": i + 1,
                        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                        "text": text.strip()
                    })

        # Remove headers/footers
        temp_df = pd.DataFrame(raw_text_data)
        header_footer_candidates = temp_df['text'].value_counts()
        common_header_footer = set(
            header_footer_candidates[header_footer_candidates > 5].index)

        for row in raw_text_data:
            if row['text'] not in common_header_footer and len(row['text'].strip()) > 5:
                all_text_data.append({
                    **row,
                    "text": re.sub(r'\s+', ' ', row['text']).strip()
                })

        # Extract images
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")['blocks']
            for b in blocks:
                if b['type'] == 1:
                    x0, y0, x1, y1 = b['bbox']
                    img = b['image']
                    img_filename = f"{pdf_name}_page_{page_num+1}_img_{len(all_image_data)+1}.png"
                    img_path = os.path.join(image_dir, img_filename)

                    with open(img_path, "wb") as f:
                        f.write(img)

                    all_image_data.append({
                        "pdf_name": pdf_name,
                        "page": page_num + 1,
                        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                        "image_name": img_filename,
                        "image_path": img_path,
                        "content_type": "image",
                        "content": img_path
                    })

    print("✅ Finished extracting text and images.")

    # DataFrames
    text_df = pd.DataFrame(all_text_data)
    image_df = pd.DataFrame(all_image_data)

    print("🔎 Filtering and embedding chunks...")
    image_df["content"] = image_df["content"].apply(lambda x: x.strip())
    image_df = image_df[image_df["content"].str.len() > 50]

    text_chunks = [
        {
            "type": "Text",
            "text": row["text"].strip(),
            "source": f"{row['pdf_name']} (page {row['page']})"
        }
        for _, row in text_df.iterrows()
        if is_valid(row["text"])
    ]

    image_chunks = [
        {
            "type": "Image",
            "text": row["content"].strip(),
            "source": f"{row['pdf_name']} (page {row['page']})"
        }
        for _, row in image_df.iterrows()
        if is_valid(row["content"])
    ]

    all_chunks = text_chunks + image_chunks
    embeddings = model.encode([c["text"]
                              for c in all_chunks], convert_to_tensor=True)

    print(f"✅ Loaded {len(all_chunks)} valid chunks.")

    with open(cache_file, "wb") as f:
        pickle.dump((all_chunks, embeddings), f)

    return all_chunks, embeddings


# Pre-load everything at startup
cleaned_chunks, chunk_embeddings = load_chunks()

# --- RAG Endpoint ---


@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question")
    question_embedding = model.encode(question, convert_to_tensor=True)
    cos_scores = util.cos_sim(question_embedding, chunk_embeddings)[0]

    adjusted_scores = cos_scores.cpu().numpy(
    ) + np.array([len(c["text"]) * 0.0005 for c in cleaned_chunks])
    top_k = 3
    top_results = np.argpartition(-adjusted_scores, range(top_k))[:top_k]
    retrieved_chunks = [cleaned_chunks[i] for i in top_results]

    context_for_prompt = "\n\n".join(
        [chunk["text"] for chunk in retrieved_chunks])

    rag_prompt = f"""You are a helpful assistant. Use only the following context to answer the question.
If the answer is not found in the context, say \"I don't know.\"

Context:
""" + context_for_prompt + f"""

Question: {question}
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": rag_prompt}
        ],
        temperature=0.2
    )

    return jsonify({
        "question": question,
        "answer": response.choices[0].message.content,
        "context": retrieved_chunks
    })

# --- Admin Endpoint: Clear Cache ---


@app.route("/clear-cache", methods=["POST"])
def clear_cache():
    if os.path.exists(cache_file):
        os.remove(cache_file)
        global cleaned_chunks, chunk_embeddings
        cleaned_chunks, chunk_embeddings = load_chunks()
        return jsonify({"message": "✅ Cache cleared and chunks reloaded."})
    else:
        return jsonify({"message": "⚠️ No cache found to clear."})


if __name__ == "__main__":
    app.run(debug=True)
