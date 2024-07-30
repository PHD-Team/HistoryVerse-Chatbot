import logging
import os
import argparse
import uuid

from flask import Flask, request, jsonify
# for vision model and images visulization
import io
import requests
from PIL import Image
# for chat with docs
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from gemini import GOOGLE_API_KEY, gemini_chat_model, identify_url_content
from SpeechRecognition import recognize_speech, process_user_voice, voice, speek, text_to_speech, process_audio_path
from gemini_data_chat import retrieval_qa_pipline
from upload_audio_file import get_firebase_url, upload_to_firebase

from gemini import memory

app = Flask(__name__)

def process_response(query, convo, speak_response):
    response = jsonify({"Question": query, "Answer": convo})
    if speak_response:
        audio_filename = f"{uuid.uuid4()}.mp3" 
        audio_path = f"speak_audio_file/{audio_filename}"

        # Generate the audio file
        audio_filename = voice(convo, audio_filename)

        # Upload the audio to Firebase Storage
        upload_to_firebase(audio_filename, audio_path) 

        # Get the Firebase URL
        firebase_url = get_firebase_url(audio_path)

        # Delete the local audio file after upload
        os.remove(audio_filename)

        # Update the response to include the Firebase URL
        response = jsonify({"Question": query}, {"Answer": convo}, {"audio_url": firebase_url})

    return response

@app.route("/text_convo", methods=["GET", "POST"])
def text_conversation():
    data = request.get_json()
    query = data.get("query")
    speak_response = data.get("speak", False)

    if query.startswith('http'):
        content_type = identify_url_content(query)
        if content_type == 'image':
            return image_conversation()
        elif content_type == 'pdf':
            return pdf_conversation()
    
    elif isinstance(query, str):
        qa = retrieval_qa_pipline()
        res = qa.invoke({"query": query})
        answer, docs = res["result"], res["source_documents"]
        memory.append(f"answer: {answer}")

        if answer.lower() != 'answer is not available in the context':
             response = process_response(query, answer, speak_response)
        else:
            convo = gemini_chat_model(query)
            response = process_response(query, convo, speak_response)          
    return response

@app.route("/voice_convo", methods=["GET", "POST"])
def voice_conversation():
    data = request.get_json()
    audio_file_path = data.get("audio_file_path")
    speak_response = data.get("speak", False)

    query = process_audio_path(audio_file_path)
    if not query:
        return jsonify("No input detected. Please try again.")
    else:
        qa = retrieval_qa_pipline()
        res = qa.invoke({"query": query})
        answer = res["result"]
        memory.append(f"answer: {answer}")  
        
        if answer.lower() != 'answer is not available in the context':
             response = process_response(query, answer, speak_response)
        else:
            convo = gemini_chat_model(query)
            response = process_response(query, convo, speak_response)           
    return response

@app.route("/image_convo", methods=["GET", "POST"])
def image_conversation():
    data = request.get_json()
    image_source = data.get("image_source")
    mode = data.get("mode", "text")  # Default to text mode
    question = data.get("question")
    audio_file_path = data.get("audio_file_path")
    speak_response = data.get("speak", False)

    if image_source.startswith('http'):
        img_response = requests.get(image_source)
        if img_response.status_code == 200:
            img = Image.open(io.BytesIO(img_response.content))
        else:
            return jsonify({"error": "Error downloading image", "status_code": img_response.status_code}),500
    else:
        if not os.path.isfile(image_source):
            return jsonify({"error": "Invalid image path"})
        img = Image.open(image_source)

    if mode == "text":
        if question:
            response_text = gemini_chat_model(question, img)
        else:
            return jsonify({"error": "Please provide a question in text mode."})

    elif mode == "voice":
        question = process_audio_path(audio_file_path)
        if question:
            response_text = gemini_chat_model(question, img)
        else:
            return jsonify({"error": "Speech not recognized."})
    else:
        return jsonify({"error": "Invalid mode. Choose 'text' or 'voice'."})

    response = process_response(question, response_text, speak_response)
    return response

@app.route("/pdf_convo", methods=["GET", "POST"])
def pdf_conversation():
    data = request.get_json()
    file_source = data.get("file_source")
    mode = data.get("mode", "text")
    audio_file_path = data.get("audio_file_path")
    question = data.get("question")
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
    you can summrize the all context too. Make sure to provide all the details.
    If the answer is not in the provided context, simply say so and inform the user.
    Do not provide the wrong answer. Please answer in the same language as the question.
    Make sure the answers are in the same language as their question.
    Context:
    {context}
    Question:
    {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    full_prompt = prompt + "\n" + "\n".join(memory)

    stuff_chain = load_qa_chain(model, chain_type="stuff", prompt= full_prompt)
    
    if mode == "text":
        if not question:
            return jsonify({"error": "Please provide a question in text mode."})
    elif mode == "voice":
        question = process_audio_path(audio_file_path)
        if not question:
            return jsonify({"error": "Speech not recognized."})
    else:
        return jsonify({"error": "Invalid mode. Choose 'text' or 'voice'."})
    
    memory.append(f"User: {question}")
    
    result = stuff_chain.invoke({"input_documents": pages, "question": question}, return_only_outputs=True)
    memory.append(f"answer: {result['output_text']}")
    response_text = result['output_text']
    response = process_response(question, response_text, speak_response)
    return response
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5110, help="Port to run the API on. Defaults to 5110.")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",  # Change this to 0.0.0.0
        help="Host to run the UI on. Defaults to 0.0.0.0 to make the UI externally accessible from other devices.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    app.run(debug=False, host=args.host, port=args.port)