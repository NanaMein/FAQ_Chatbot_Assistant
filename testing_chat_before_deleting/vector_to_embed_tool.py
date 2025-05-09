from litellm import vector_store_registry
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.text_splitter import SentenceSplitter
from dotenv import load_dotenv
from llama_index.core import Settings
import os

load_dotenv()


def run_please():
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    llm = Groq(model=os.getenv("TEST_GROQ_SMALL") or os.getenv("GROQ_MODEL"),
               api_key=os.getenv('GROQ_API_KEY'))

    # Initialize the Milvus vector store
    milvus_store = MilvusVectorStore(
        uri=os.getenv('CLUSTER_URI'),
        token=os.getenv('CLUSTER_TOKEN'),
        dim=384,  # Ensure this matches your embedding model's output dimension
        collection_name='kokomi_collections',
        # overwrite=True,
        embedding_field="embedding"
    )
    # Load your documents
    # load_docx = SimpleDirectoryReader(input_dir=os.getenv('DATA_DIR')).load_data()

    storage_context = StorageContext.from_defaults(vector_store=milvus_store)

    index = VectorStoreIndex.from_vector_store(
        storage_context=storage_context,
        vector_store=milvus_store
    )





    # index = VectorStoreIndex.from_documents(
    #     load_docx,
    #     storage_context=storage_context,
    #     vector_store=milvus_store
    # )

    query_engine = index.as_query_engine(llm=llm)
    print("preparing query\n****************************************")
    test_response = query_engine.query("give something interesting about kokomi?")
    print(test_response)


if __name__ == '__main__':
    run_please()
