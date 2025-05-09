from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.text_splitter import SentenceSplitter
from dotenv import load_dotenv
from llama_index.core import Settings
import os

from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer

load_dotenv()


def run_please():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


    llm = Groq(model=os.getenv("TEST_GROQ_SMALL")or os.getenv("GROQ_MODEL"), api_key=os.getenv('GROQ_API_KEY'))

    # Initialize the Milvus vector store
    milvus_store = MilvusVectorStore(
        uri=os.getenv('CLUSTER_URI'),
        token=os.getenv('CLUSTER_TOKEN'),
        dim=384,  # Ensure this matches your embedding model's output dimension
        collection_name='kokomi_collections'
    )
    # Load your documents
    load_docx = SimpleDirectoryReader(input_dir=os.getenv('DATA_DIR')).load_data()

    # Split your documents into nodes
    chunk = SentenceSplitter(chunk_size=1000, chunk_overlap=100)
    nodes = chunk.get_nodes_from_documents(load_docx)

    storage_context = StorageContext.from_defaults(vector_store=milvus_store)
    # Create the VectorStoreIndex
    index = VectorStoreIndex(nodes).from_vector_store(
        vector_store=milvus_store,
        embed_model=embed_model,
        storage_context=storage_context
    )

    # Now you can build your query engine from the index
    query_engine = index.as_query_engine(llm=llm)
    print("preparing query\n****************************************")
    test_response = query_engine.query("give something interesting about kokomi?")
    print(test_response)

if __name__ == '__main__':
    run_please()
# from pymilvus import connections, MilvusClient
# connections.connect(
#     alias="default",
#     uri=os.getenv('CLUSTER_URI'),  # Example: 'http://localhost:19530' or Zilliz URL
#     token=os.getenv('CLUSTER_TOKEN'),
#     user=os.getenv('CLUSTER_USER'),
#     password=os.getenv('CLUSTER_PASS')
# )# If using Zilliz Cloud
# client = MilvusClient(uri=os.getenv('CLUSTER_URI'),
#                       token=os.getenv('CLUSTER_TOKEN'),
#                       user=os.getenv('CLUSTER_USER'),
#                         password=os.getenv('CLUSTER_PASS')
#                       )
# Set the embedding model globally