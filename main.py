# =====================================================
# Gujarati + English RAG pipeline + Gemma (without CSV)
# =====================================================

!pip -q install python-docx sentence-transformers faiss-cpu rank_bm25 tqdm transformers accelerate sentencepiece huggingface_hub requests

import os, re, requests, torch
import numpy as np, pandas as pd
from tqdm.auto import tqdm
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ----------------------------
# CONFIG
# ----------------------------
# 👉 REPLACE with your DOCX path
DOCX_PATH = "/kaggle/input/info.docx"

CHUNK_SIZE_TOKENS    = 350
CHUNK_OVERLAP_TOKENS = 60

EMBED_MODEL_GUJARATI = "l3cube-pune/gujarati-sentence-bert-nli"
EMBED_MODEL_ENGLISH  = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

TOP_K               = 50
KEEP_K              = 3
RETRIEVAL_THRESHOLD = 0.20

# 👉 REPLACE with your HuggingFace token
HF_TOKEN = os.getenv("HF_TOKEN") or "YOUR_HF_TOKEN_HERE"

MODEL_NAME     = "google/gemma-3-12b-it"
TOKENIZER_URL  = "https://huggingface.co/unsloth/gemma-3-12b-it/resolve/main/tokenizer.model"
TOKENIZER_PATH = "./tokenizer.model"

# ----------------------------
# Helpers
# ----------------------------
def read_docx(path):
    doc = Document(path)
    out=[]
    paras=[p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for i in range(0,len(paras),2):
        if i+1 < len(paras):
            out.append(f"Q: {paras[i]} A: {paras[i+1]}")
    return out

def is_gujarati_text(text):
    return bool(re.search(r'[\u0A80-\u0AFF]', text))

def detect_lang(text):
    return "Gujarati" if is_gujarati_text(text) else "English"

def unknown_str(lang):
    return "ઉલ્લેખ નથી." if lang=="Gujarati" else "Not mentioned."

# ----------------------------
# Chunking
# ----------------------------
from transformers import AutoTokenizer as ChunkTok
_chunk_tok = ChunkTok.from_pretrained("bert-base-multilingual-cased", use_fast=True)

def token_chunks(text, size=CHUNK_SIZE_TOKENS, overlap=CHUNK_OVERLAP_TOKENS):
    ids = _chunk_tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
    step = max(1, size - overlap)
    chunks=[]
    for i in range(0, len(ids), step):
        piece = ids[i:i+size]
        if not piece: break
        decoded = _chunk_tok.decode(piece, skip_special_tokens=True).strip()
        if decoded: chunks.append(decoded)
    return chunks

def chunk_text(paragraphs):
    out=[]
    for p in paragraphs:
        if len(p.split()) < 60:
            out.append(p)
            continue
        out.extend(token_chunks(p))
    return out

# ----------------------------
# FAISS + BM25
# ----------------------------
def build_faiss_index(chunks, embed_model):
    emb = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    emb = emb / norms
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index, emb

def tokenize_for_bm25(text):
    return [t.lower() for t in re.findall(r'[\u0A80-\u0AFF]+|[A-Za-z]+|\d+(?:.\d+)?', text)]

# ----------------------------
# Build RAG system
# ----------------------------
def prepare_system(path):
    paragraphs = read_docx(path)
    chunks = chunk_text(paragraphs)
    embed_model_guj = SentenceTransformer(EMBED_MODEL_GUJARATI)
    embed_model_en  = SentenceTransformer(EMBED_MODEL_ENGLISH)
    guj_chunks = [c for c in chunks if is_gujarati_text(c)]
    en_chunks  = [c for c in chunks if not is_gujarati_text(c)]
    index_guj,_ = build_faiss_index(guj_chunks, embed_model_guj) if guj_chunks else (None, None)
    index_en,_  = build_faiss_index(en_chunks , embed_model_en ) if en_chunks  else (None, None)
    bm25 = BM25Okapi([tokenize_for_bm25(c) for c in chunks])
    return {
        "chunks": chunks,
        "guj_chunks": guj_chunks,
        "en_chunks": en_chunks,
        "embed_model_guj": embed_model_guj,
        "embed_model_en": embed_model_en,
        "index_guj": index_guj,
        "index_en": index_en,
        "bm25": bm25
    }

system = prepare_system(DOCX_PATH)

# ----------------------------
# Retrieval
# ----------------------------
def retrieve(query, system, top_k=TOP_K, keep_k=KEEP_K):
    lang = detect_lang(query)
    bm25 = system["bm25"]

    if lang=="Gujarati" and system["index_guj"]:
        embed_model,index,chunks = system["embed_model_guj"], system["index_guj"], system["guj_chunks"]
    else:
        embed_model,index,chunks = system["embed_model_en"], system["index_en"], system["en_chunks"]

    if not chunks: return [], None, 0

    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, ids = index.search(q_emb, top_k)

    faiss_res = [(chunks[i], float(scores[0][j])) for j,i in enumerate(ids[0])]
    bm25_res  = [(c, bm25.get_scores(tokenize_for_bm25(query))[system["chunks"].index(c)]) for c in chunks]
    bm25_res  = sorted(bm25_res, key=lambda x:x[1], reverse=True)[:top_k]

    merged = {c:s for c,s in faiss_res}
    for c,s in bm25_res: merged[c] = merged.get(c,0) + s

    ranked = sorted(merged.items(), key=lambda x:x[1], reverse=True)
    ranked = [(c,s) for c,s in ranked if s >= RETRIEVAL_THRESHOLD]

    return (ranked[:keep_k], "DOCX", ranked[0][1]) if ranked else ([], None, 0)

# ----------------------------
# Gemma model
# ----------------------------
login(HF_TOKEN)
if not os.path.exists(TOKENIZER_PATH):
    r = requests.get(TOKENIZER_URL, headers={"Authorization": f"Bearer {HF_TOKEN}"})
    open(TOKENIZER_PATH, "wb").write(r.content)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                 device_map="auto",
                                                 torch_dtype=torch.float32,
                                                 token=HF_TOKEN)
gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ----------------------------
# Humanize
# ----------------------------
def humanize_answer(question, base_answer):
    prompt=f"""
You are a helpful assistant.
Question: {question}
Base factual answer: {base_answer}
Instructions:
- Reply ONLY in the same language as the question.
- Produce a single smooth paragraph of about 4–5 lines (≈80–120 words).
- Keep all factual details and numbers accurate.
- Do not list bullet points; one continuous paragraph only.
"""
    out = gen(prompt, max_new_tokens=220, temperature=0.8, top_p=0.95, do_sample=True)[0]["generated_text"]
    return out.replace(prompt, "").strip()

# ----------------------------
# Ask
# ----------------------------
def ask_question(query):
    lang = detect_lang(query)

    # Only DOCX RAG
    retrieved, source, score = retrieve(query, system)
    if not retrieved:
        print(unknown_str(lang))
        return

    best_chunk = retrieved[0][0]
    print(humanize_answer(query, best_chunk))
    print(f"[Source: {source} | Match Score: {score:.2f}]")

# ----------------------------
# Loop
# ----------------------------
print("✅ RAG + Gemma system ready. Type your question (Gujarati or English). Type 'stop' to exit.\n")
while True:
    query = input("❓ Your Question: ").strip()
    if query.lower() == "stop":
        print("🛑 Stopping Q&A session. Goodbye!")
        break
    if not query:
        continue
    ask_question(query)
    print("\n" + "="*60 + "\n")
