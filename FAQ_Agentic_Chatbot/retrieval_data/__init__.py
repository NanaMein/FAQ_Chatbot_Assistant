# 1. General-Purpose Embeddings
# Model Name	Dim	Description
# BAAI/bge-small-en-v1.5	384	Small, efficient, English-focused (your current choice).
# BAAI/bge-base-en-v1.5	768	Larger version of bge-small, better accuracy.
# BAAI/bge-large-en-v1.5	1024	High-quality, but slower and resource-intensive.
# sentence-transformers/all-MiniLM-L6-v2	384	Lightweight, multilingual support.
# sentence-transformers/all-mpnet-base-v2	768	High accuracy, English-only.
# 2. Multilingual Models
# Model Name	Dim	Description
# sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2	384	Supports 50+ languages.
# intfloat/multilingual-e5-large	1024	State-of-the-art for multilingual tasks.
# 3. Specialized Models
# Model Name	Dim	Description
# llmrails/ember-v1	1024	Optimized for RAG (Retrieval-Augmented Generation).
# thenlper/gte-base	768	General Text Embeddings (GTE) by阿里巴巴.
# Salesforce/SFR-Embedding-Mistral	4096	Large, high-performance (requires significant GPU RAM).