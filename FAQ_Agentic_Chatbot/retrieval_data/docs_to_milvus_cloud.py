from typing import Any
from crewai.tools import BaseTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.groq import Groq
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
# from llama_index.core.text_splitter import SentenceSplitter
# from llama_index.core.query_engine import BaseQueryEngine
# from llama_index.core.memory import ChatMemoryBuffer
# from llama_index.core.llms import ChatMessage
from llama_index.core import Settings
from dotenv import load_dotenv
import os



load_dotenv()


llm=Groq(
    model=os.getenv('TEST_GROQ_SMALL')
    or os.getenv('GROQ_MODEL'),
    api_key=os.getenv('GROQ_API_KEY')
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name='BAAI/bge-base-en-v1.5'
)

def lazy_loading():
    documents = SimpleDirectoryReader(input_dir='data/').load_data()

    # chunk = SentenceSplitter()
    #
    # nodes = SentenceSplitter.get_nodes_from_documents(documents)
    milvus_store = MilvusVectorStore(
        uri=os.getenv('CLUSTER_URI'),
        token=os.getenv('CLUSTER_TOKEN'),
        dim=768,
        embedding_field="embedding",
        collection_name="baby_fionica_collections"

    )
    storage_context = StorageContext.from_defaults(vector_store=milvus_store)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, vector_stores=milvus_store
    )

    engine = index.as_query_engine(llm=llm)
    return engine

query_engine=lazy_loading()

ques = "who is fionica?"
tool = query_engine.query(ques)

print(tool)


# class QueryEngineTool():#(BaseTool):
#
#     def __init__(self, _engine):
#         # super().__init__()
#         self.query_engine: BaseQueryEngine = _engine
#         # self.name = "Retrieval of document chatbot"
#         # self.description = " For retrieving relevant information depending on context"
#
#     def _run(self, question: str) -> str:
#         response = self.query_engine.query(question)
#         return str(response)



