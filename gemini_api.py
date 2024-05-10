import logging
import os
import argparse
import sys

from flask import Flask, request, jsonify
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from gemini import GOOGLE_API_KEY, gemini_text_model, gemini_vision_model, recognize_speech, voice, identify_url_content
from gemini_data_chat import retrieval_qa_pipline

from PIL import Image
import io
import requests
from PIL import Image

import tempfile
from langchain_community.document_loaders import PyPDFLoader

app = Flask(__name__)

@app.route("/text_convo", methods=["GET", "POST"])
def text_conversation():
    data = request.get_json()
    query = data.get("query")
    speak_response = data.get("speak", False)  # Get 'speak' flag from request

    if query.startswith('http'):
        content_type = identify_url_content(query)
        if content_type == 'image':
            response = image_conversation()
        elif content_type == 'pdf':
            response = pdf_conversation()
    elif isinstance(query, str):
        qa = retrieval_qa_pipline()
        res = qa.invoke({"query": query})
        answer, docs = res["result"], res["source_documents"]

        if answer == 'answer is not available in the context':
            convo = gemini_text_model(query)
            response_text = convo
        else:
            response_text = answer  

        response = jsonify({"response": response_text})

    
    if speak_response: 
        jsonify(response_text)
        voice(response_text)
    return response

@app.route("/voice_convo", methods=["GET", "POST"])
def voice_conversation():
    data = request.get_json()
    speak_response = data.get("speak", False)

    query = recognize_speech()
    if query:
        qa = retrieval_qa_pipline()
        res = qa.invoke({"query": query})
        answer, docs = res["result"], res["source_documents"]
        if answer == 'answer is not available in the context':
            convo = gemini_text_model(query)
            response_text = convo 
        else:
            response_text = answer

        response = jsonify({"response": response_text})
    else:
        response = jsonify({"error": "Speech not recognized."})

    if speak_response:
        voice(response_text)
    return response
    
@app.route("/image_convo", methods=["GET", "POST"])
def image_conversation():
    data = request.get_json()
    image_source = data.get("query")
    speak_response = data.get("speak", False)

    if image_source.startswith('http'):
        img_response = requests.get(image_source)
        if img_response.status_code == 200:
            img = Image.open(io.BytesIO(img_response.content))
        else:
            return jsonify({"Error downloading image": img_response.status_code})
    else:
        if not os.path.isfile(image_source):
            return jsonify({"error": "Invalid image path"})
        img = Image.open(image_source)
        
    question = question = input("\nEnter your question about this image: ")
    convo = gemini_vision_model(question, img)
    response_text = convo
    response = jsonify({"response": response_text})
    
    if speak_response:
        voice(response)
    return response

@app.route("/pdf_convo", methods=["GET", "POST"])
def pdf_conversation():
    data = request.get_json()
    file_source = data.get("query")
    speak_response = data.get("speak", False)

    if os.path.isfile(file_source):
        pdf_loader = PyPDFLoader(file_source)
        pages = pdf_loader.load_and_split()
    else:
        pdf_response = requests.get(file_source, stream=True)
        if pdf_response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
                temp_pdf.write(pdf_response.content)
                pdf_loader = PyPDFLoader(temp_pdf.name)
                pages = pdf_loader.load_and_split()
        else:
            return jsonify({"Error downloading PDF": pdf_response.status_code})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_text(context)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 5})

    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(
        model, retriever=vector_index, return_source_documents=False
    )

    prompt_template = """
    Your name is Mora. Answer the question as detailed as possible from the provided context.
    Make sure to provide all the details. If the answer is not in the provided context,
    simply say so and inform the user. Do not provide the wrong answer.
    Context:
    {context}
    Question:
    {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    stuff_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    question = input("\nEnter your question about this PDF: ")
    result = stuff_chain.invoke({"input_documents": pages, "question": question}, return_only_outputs=True)
    response_text = result['output_text']
    response = jsonify({"response": response_text})
    
    if speak_response:
        voice(response)
    return response

if __name__ == "__main__":
    app.run(debug=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5110, help="Port to run the API on. Defaults to 5110.")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to run the UI on. Defaults to 127.0.0.1. "
        "Set to 0.0.0.0 to make the UI externally "
        "accessible from other devices.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    app.run(debug=False, host=args.host, port=args.port)