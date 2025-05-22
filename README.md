# CMPE-255 Project Report  
## Automatic Data Mining for Large PDF Files

*Team 13*  
Adityaraj Kaushik, Rishabh Malviya, Kumar Harsh

---

## 📌 Project Description

This project automates the extraction, cleaning, and semantic processing of unstructured PDF documents exceeding 500 pages. 
It integrates NLP techniques, dense vector embeddings, and visualization tools to support intelligent tasks like semantic search and question answering (QA).
The project features a React frontend with a Flask RAG application that validates answers using LLM.

---
## 📌 GOOGLE COLLAB LINK FOR TESTING AND EDA
[https://colab.research.google.com/drive/1lotEGHq7wR0EFOZxB0Td2K1By-juk-wT](url)

---

## 🚀 Major Contributions

- ✅ Text and image extraction from long-form PDF documents using PyMuPDF  
- ✅ Text cleaning and preprocessing using regex, ASCII encoding, and NLTK stopwords
- ✅ EDA: Word Frequency, Basic stats, WordCloud visualization, Top keywords, Word Image count per page, Summary tables for all pdfs, NER
- ✅ Word frequency analysis and WordCloud visualization  
- ✅ Semantic chunking of large text blocks for meaningful context segmentation  
- ✅ Dense vector embeddings using SentenceTransformers  
- ✅ Chunk-wise summarization using transformer models  
- ✅ SQLite database integration for document storage and retrieval  
- ✅ End-to-end natural language query pipeline
- ✅ React Frontend with chunk and source displays of context sent to LLM for RAG.  
- ✅ Flask API with /ask and /clear-cache endpoints to ask and handle cache (for adding new pdfs and embeddings)

---

## 👥 Team & Responsibilities

| Name               | SJSU ID     | Responsibilities |
|--------------------|-------------|------------------|
| *Adityaraj Kaushik* | 017631471 | Acquired and processed PDFs, text and image extraction, data cleaning, tokenization, stopword removal, EDA, WordCloud and Seaborn visualizations |
| *Rishabh Malviya*   | 018190419 | Implemented semantic chunking, embedding with SentenceTransformers and QA pipeline |
| *Harsh Kumar*       | 017680845 | Created SQLite database connection, implemented translation processing, question/problem formulation logic and validation, Worked on RAG architecture and application development |
