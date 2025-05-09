from typing import Any

from crewai.tools import BaseTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from dotenv import load_dotenv
import os

load_dotenv()


llm=Groq(
    model=os.getenv('TEST_GROQ_SMALL','GROQ_MODEL'),
    api_key=os.getenv('GROQ_API_KEY')
)
embed_model = HuggingFaceEmbedding(
    model_name='BAAI/bge-base-en'
)

def lazy_loading():
    documents = SimpleDirectoryReader(input_dir='data/').load_data()

    chunk = SentenceSplitter(chunk_size=1000, chunk_overlap=100)

    nodes = chunk.get_nodes_from_documents(documents)

    index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)

    engine = index.as_query_engine(llm=llm)
    return engine

query_engine=lazy_loading()

class QueryEngineTool(BaseTool):

    def __init__(self, _engine):
        super().__init__()
        self.query_engine: BaseQueryEngine = _engine
        self.name = "Retrieval of document chatbot"
        self.description = " For retrieving relevant information depending on context"

    def _run(self, question: str) -> str:
        response = self.query_engine.query(question)
        return str(response)



