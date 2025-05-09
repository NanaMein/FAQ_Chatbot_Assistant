from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_groq.chat_models import ChatGroq
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.groq import Groq
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

embed_model = HuggingFaceEmbedding()

documents = SimpleDirectoryReader(input_dir="data_sources/" ).load_data()

parser=SentenceSplitter(chunk_size=1000, chunk_overlap=100)

node = parser.get_nodes_from_documents(documents)

index = VectorStoreIndex(node, embed_model=embed_model)


query_engine = index.as_query_engine(llm=llama_index_llm)


"""FUCK THIS LLAMA AND CREWAII, BETTER CHANGE THE GENAI LLAMA TO SOMETHING
        COMPATIBLE WITH CREWAI
        """




def test_tool(query: str) -> str:
    """this is a tool used to be called always to return a string"""
    return f"the query:( {query} )has successfully passed in test tool"
testing=test_tool()
chat_agent = Agent(
    role="ChatBot",
    goal="""Your task is to always reply based on query ({query})
            Do not reply in a very long prompt except when explicitly
            asked to describe or explain
            """,
    backstory="""You are a youtube live streamer, you keep your viewer entertained.
                You also keep answering them regarding their query ({query}).
                And calmy reply to them one by one
                """,
    llm=ai_agent_llm,
    verbose=True,
    tools=[testing]
)
task = Task(
    description="""the agent will use the tool given to it""",
    expected_output="""use the tool and retrieve the string value given by the tool""",
    agent=chat_agent
)


# chat_agent = Agent(
#     role="ChatBot",
#     goal="""Your task is to always reply based on query ({query})
#             Do not reply in a very long prompt except when explicitly
#             asked to describe or explain
#             """,
#     backstory="""You are a youtube live streamer, you keep your viewer entertained.
#                 You also keep answering them regarding their query ({query}).
#                 And calmy reply to them one by one
#                 """,
#     llm=ai_agent_llm,
#     verbose=True,
#     tools=[test_tool]
#
# )






#
# def test_tool(question: str) -> str:
#         """This tool will test if there would be a return of string"""
#         return "tool is working. please configure this code"

# task = Task(
#     description="""You will roleplay as a virtual chatbot,
#                 you will reply based like how humans talked as an everyday
#                 conversation. Unless when asked to describe or explained based
#                 on query ({query})""",
#     expected_output="""reply based on query ({query}) as an everyday conversation
#                 sized sentences""",
#     agent=chat_agent
# )

crew = Crew(
    agents=[chat_agent],
    tasks=[task],
    process=Process.sequential,
    verbose=True

)






chat = query_engine.query("who is fionica?")


print("Howdie, motherfucker : do you wanna know who is fiona?) ")

_inputs={
    'query': "hello my friend, i want to ansk a question"
}

output=crew.kickoff(inputs=_inputs)
print(f"answer: {output}"
      f"who is fionica? the answer is :{chat}")