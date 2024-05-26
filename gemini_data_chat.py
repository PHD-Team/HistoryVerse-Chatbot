import os
import logging
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.chroma import Chroma

from chromadb.config import Settings

import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

PERSIST_DIRECTORY = "DB/"

CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

def retrieval_qa_pipline():
   
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # load the vectorstore
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS
    )
    retriever = db.as_retriever()

    # get the prompt template and memory if set by the user.
    prompt_template = """
    your name is Mora. You are a helpful assistant, you will use the provided context to answer user questions.
    Read the given context before answering questions and think step by step. If you can not answer a user question based on 
    the provided context just say, "answer is not available in the context", don't provide the wrong answer. Provide a detailed answer to the question.\n\n.
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])

    # load the llm pipeline
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, LOGGING=logging)
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
        retriever=retriever,
        return_source_documents=True,  # verbose=True,
        callbacks=callback_manager,
        chain_type_kwargs={
            "prompt": prompt,
        },
    )

    return qa

# def main():

#     qa = retrieval_qa_pipline()
#     # Interactive questions and answers
#     while True:
#         query = input("\nEnter a query: ")
#         if query == "exit":
#             break
        
#         # Get the answer from the chain
#         res = qa.invoke({"query": query})  # Pass the query as a dictionary
#         answer, docs = res["result"], res["source_documents"]

#         # Print the result
#         print("\n\n> Question:")
#         print(query)
#         print("\n> Answer:")
#         print(answer)

# if __name__ == "__main__":
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
#     )
#     main()