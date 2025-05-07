from dotenv import load_dotenv
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.node_parser import NodeParser
import os

load_dotenv()


embed_model = HuggingFaceEmbedding()
llm = Groq(model=os.getenv('TEST_GROQ_SMALL'), api_key=os.getenv('GROQ_API_KEY'))

def load_background_documents() -> BaseQueryEngine:
    load_docx = SimpleDirectoryReader(input_dir=os.getenv('DATA_DIR')).load_data()

    chunking = SentenceSplitter(chunk_size=1000, chunk_overlap=100)

    nodes = chunking.get_nodes_from_documents(load_docx)
    index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)
    query_engine = index.as_query_engine(llm=llm)
    return query_engine

def run_chatbot(human_message: str) -> str:
    query_engine = load_background_documents()

    prompt_template=f"""You are a Virtual Chatbot named Kokomi. You will roleplay based on 
    how kokomi talks and acts. You could also add some expressions. You will always assist and answer
    query. You are Kokomi, so if Kokomi is in the context, you should reply as I or me instead of describing Kokomi
    in a different perspective. The question would be : {human_message}. Additionally,
    You are my virtual daughter and you will act like Kokomi.
    To answer, explain your answer like how a roleplaying virtual character is"""

    response = query_engine.query(prompt_template).response

    return response


