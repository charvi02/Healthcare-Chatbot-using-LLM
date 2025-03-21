from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

local_model = "llama3.1"
llm = ChatOllama(model=local_model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the usefr overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

persist_directory = "chroma_db_5_tb"
vector_db = Chroma(
    persist_directory=persist_directory,
    embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="local-rag"
)

retriever = MultiQueryRetriever.from_llm(
    retriever=vector_db.as_retriever(search_kwargs={"k": 2}),
    llm=llm,
    prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

choice = input("What do you want to do? 1 to diagnose and 2 to quit")
while choice != '2':
    question = input("Enter the symptoms")
    print(chain.invoke(question))
    choice = input("What do you want to do? 1 to diagnose and 2 to quit")