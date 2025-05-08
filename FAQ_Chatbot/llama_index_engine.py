from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.text_splitter import SentenceSplitter
from dotenv import load_dotenv
import os

from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer

load_dotenv()

embed_model = HuggingFaceEmbedding()
llm = Groq(model=os.getenv("TEST_GROQ_SMALL", "GROQ_MODEL"), api_key=os.getenv('GROQ_API_KEY'))


load_docx = SimpleDirectoryReader(input_dir=os.getenv('DATA_DIR')).load_data()

chunk = SentenceSplitter(chunk_size=1000, chunk_overlap=100)

nodes = chunk.get_nodes_from_documents(load_docx)

index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)

query_engine = index.as_query_engine(llm=llm)

memory = ChatMemoryBuffer.from_defaults(token_limit=5000)

def query_engine_run(question: str):
    history = memory.get()


    prompt_template = f""" Using the previous context: //{history}//
                    and the current question //{question}//, use all these information
                    to generate an answer based on the information you have"""

    _query_engine = query_engine.query(prompt_template)

    chat_history = [
        ChatMessage(role="user", content= str(question)),
        ChatMessage(role="assistant", content= _query_engine),
    ]
    memory.put_messages(chat_history)

    return _query_engine



if __name__=='__main__':
    while True:
        loop_message = input("what do you want to say? ")
        loop_message_output =query_engine_run(loop_message)

        if loop_message.lower() == "exit":

            print("Exiting the loop. Goodbye!")
            memory.reset()
            break  # Exit the loop

        print(loop_message_output)


