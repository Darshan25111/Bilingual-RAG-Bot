# Gujarati + English RAG Question-Answering System with Gemma

A Retrieval-Augmented Generation (RAG) system designed for both Gujarati and English languages, leveraging DOCX-based knowledge sources, FAISS embeddings, BM25 lexical scoring, and the Gemma 3 Large Language Model (LLM) to deliver human-like paragraph answers. Optimized for GPU-accelerated Kaggle notebooks.

---

## Features

- **Bilingual Support:** Automatically detects the language of queries in Gujarati and English.
- **DOCX Knowledge Base:** Extracts Questions & Answers from structured DOCX files with alternating paragraphs.
- **FAISS Embeddings:** Retrieves top semantically similar chunks from the knowledge base.
- **BM25 Scoring:** Combines lexical matching with semantic retrieval to enhance relevance.
- **Gemma LLM:** Generates fluent, human-like paragraph answers in the detected language.
- **Interactive CLI:** Real-time Q&A interface; type `stop` to exit.
- **Token-based Chunking:** Efficiently handles large DOCX documents by splitting text into chunks.
- **Offline Capability:** Fully offline-ready once the models are downloaded.

---

## Project Structure


```
├── README.md
├── rag_gemma_kaggle.ipynb # Main notebook for running the system
├── data/
│ └── your_docx.docx # Input Q&A knowledge base in DOCX format
├── models/
│ └── tokenizer.model # Gemma tokenizer model files
└── requirements.txt # Python dependencies and versions
```

---

## Kaggle Setup Instructions

1. **Environment:**
   - Kernel: Python 3.11
   - Hardware accelerator: GPU (Tesla T4 or V100 recommended)

2. **Install dependencies:**

!pip install -q python-docx sentence-transformers faiss-cpu rank_bm25 tqdm transformers accelerate sentencepiece huggingface_hub requests

3. **Upload DOCX Knowledge Base:**
- Upload your DOCX file as a Kaggle dataset.
- Replace `DOCX_PATH` in the notebook with your dataset path:
  ```
  DOCX_PATH = "/kaggle/input/yourdataset/your_docx.docx"
  ```

4. **Set HuggingFace Token:**
- Get a free token at [HuggingFace](https://huggingface.co/settings/tokens).
- Set in the notebook:
  ```
  HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN"
  ```

5. **Enable GPU:**
- Go to Settings → Accelerator → GPU
- Recommended: Tesla T4

6. **Run the notebook:**
- Install dependencies.
- Run all cells sequentially.
- Start asking questions in Gujarati or English.

---

## Usage

Run the interactive question-answer loop in the notebook:


print("Type 'stop' to exit.")
while True:
query = input("❓ Your Question: ").strip()
if query.lower() == "stop":
break
ask_question(query)

**Example Queries:**

- English: *What is the full form of AI?*
- Gujarati: *ભારતનો પ્રથમ રાષ્ટ્રપતિ કોણ હતો?*

---

## How It Works

- **DOCX Parsing:** Reads questions and answers formatted in alternating paragraphs.
- **Chunking:** Splits large paragraphs into smaller, overlapping token chunks for better retrieval.
- **Embedding Retrieval:**
  - Converts chunks into dense embeddings using GuJ-specific (l3cube BERT) and multilingual MPNet (English) models.
  - Retrieves semantically similar chunks with FAISS.
- **BM25 Scoring:** Adds lexical token-based scoring to improve retrieval precision.
- **Rank & Merge:** Combines FAISS and BM25 scores to rank relevant chunks.
- **Gemma LLM:** Uses the top-ranked chunk as a prompt to generate smooth, human-like paragraph answers in the same language as the query.
- **Interactive Output:** Prints answer along with source-based confidence score.

---

## Dependencies

- Python ≥3.10
- PyTorch
- Transformers
- Sentence Transformers
- FAISS
- rank_bm25
- python-docx
- HuggingFace Hub

---

## Notes

- Tesla T4 GPU is recommended for faster inference.
- Gemma 3-12B model requires at least 16GB GPU memory.
- DOCX files must follow the alternating paragraph format: Question → Answer → Question → Answer.

---

## Future Improvements

- Add offline caching of embeddings to avoid re-indexing on restart.
- Support dynamic loading of multiple DOCX files.
- Develop web or GUI interface for user-friendly experience.
- Extend capabilities for multi-turn conversations using RAG + Gemma.

---

## License

MIT License © 2025 [Darshan Surati](https://github.com/Darshan25111)

---

## References

- [Gemma 3 LLM]()
- [Sentence Transformers]()
- [FAISS]()
- [BM25](https://pypi.org/project/rank-bm25/)

---

