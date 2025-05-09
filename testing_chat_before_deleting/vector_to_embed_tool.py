from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import VectorStoreIndex, StorageContext
from dotenv import load_dotenv
import os
from pymilvus import connections

# Step 1: Connect to Milvus

load_dotenv()

connections.connect(
    alias="default",
    uri=os.getenv('CLUSTER_URI'),  # Example: 'http://localhost:19530' or Zilliz URL
    token=os.getenv('CLUSTER_TOKEN'),
    user=os.getenv('CLUSTER_USER'),
    password=os.getenv('CLUSTER_PASS')# If using Zilliz Cloud
)


llm = Groq(
    model=os.getenv('TEST_GROQ_MODEL'),
    api_key=os.getenv('GROQ_API_KEY')
)

milvus_store = MilvusVectorStore(
    uri=os.getenv('CLUSTER_URI'),
    token=os.getenv('CLUSTER_TOKEN'),
    dim=768,
    collection_name='llama_docs',
)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en")

storage_context = StorageContext.from_defaults(vector_store=milvus_store)

index = VectorStoreIndex.from_vector_store(
    vector_store=milvus_store,
    embed_model=embed_model,
    storage_context=milvus_store
)

query_engine = index.as_query_engine(llm=llm)

while True:
    print("infinity LOOP infinity LOOP infinity LOOP infinity LOOP infinity LOOP ")
    test_msg = input("Insert Info ")
    result = query_engine.query(test_msg)
    print(str(result))
    print('WOMP WOMP WOMP WOMP WOMP WOMP WOMP WOMP WOMP WOMP WOMP WOMP WOMP WOMP WOMP WOMP ')


