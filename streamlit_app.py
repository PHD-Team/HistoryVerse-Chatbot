import os
import tempfile
import requests
from urllib.parse import urlparse
import google.generativeai as genai
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from chromadb.config import Settings
import speech_recognition as sr
import langid
from gtts import gTTS
from faster_whisper import WhisperModel
from PIL import Image
import logging
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
import pygame
import constants

load_dotenv()
GOOGLE_API_KEY = genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

recognizer = sr.Recognizer()
source = sr.Microphone()

whisper_size = 'small'
num_cores = os.cpu_count()
whisper_model = WhisperModel(whisper_size,
                             device='cpu',
                             compute_type='int8',
                             cpu_threads=num_cores,
                             num_workers=num_cores)

import warnings
warnings.filterwarnings(
    "ignore", message=r"torch.utils._pytree._register_pytree_node is deprecated"
)

memory = [] # global conversation_memory
PERSIST_DIRECTORY = "DB/"
CHROMA_SETTINGS = Settings(anonymized_telemetry=False, is_persistent=True)
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


def process_user_voice():
    with source as s:
        recognizer.adjust_for_ambient_noise(s, duration=2)
        st.write("\nListening ... \n")
        audio = recognizer.listen(source)
    try:
        prompt_audio_path = 'prompt.wav'
        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())
        prompt_text = wav_to_text(prompt_audio_path)
        if len(prompt_text.strip()) < 3:
            st.error("No input detected. Please try again.")
            return None
        else:
            return prompt_text
    except Exception as e:
        st.error(f'Prompt error: {e}')
        return None


def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text


def voice(text, filename='output.mp3'):
    spoken_response = text.replace('*', '').replace('\n', '').replace(
        '#', '')
    language_code, _ = langid.classify(spoken_response)
    tts = gTTS(text=spoken_response,
               lang=language_code,
               slow=False,
               tld='co.uk')
    tts.save(filename)
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    return filename


generation_config = {
    "temperature": 1,
    "top_p": 1,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]


def retrieval_qa_pipline():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings,
                client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    prompt_template = """
    your name is Mora. You are a helpful assistant, you will use the provided context to answer user questions.
    Read the given context before answering questions and think step by step. If you can not answer a user question based on 
    the provided context just say, "answer is not available in the context", don't provide the wrong answer. Provide a detailed answer to the question.
    Make sure the answers are in the same language as their question.\n\n.
    Context: \n{history} \n{context}?\n
    Question: \n{question}\n
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template,
                           input_variables=["history", "context", "question"])
    memory = ConversationBufferMemory(input_key="question", memory_key="history")
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, verbose=True)
    
    qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True, 
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
        
    return qa


def gemini_chat_model(query, img=None):
    historical_prompt = """
    Your name is Mora. You are a large language model specialized in analyzing history, historical questions, historical images, and historical places in all the world, esbacaily those related to Egypt and Alexandria. 
    You have extensive knowledge of history and can answer questions about historical events, figures, and periods. you can not answer about other fildes.
    
    If the query is related to an image,Please examine the image and provide insights into its historical context , significance, and any relevant details about the depicted objects, people, or places.
    If the image does not appear to be historical or relevant to Egypt/Alexandria, please state that you cannot analyze it.
    
    If the query is about history but not related to an image, answer only about history and histracil places and torism in all the world, esbacaily in egypt and alexandrai.
    You have extensive knowledge of history and can answer questions about historical events, figures, and periods. if the question not about it please state that you cannot answer.
    Please answer the following question with historical accuracy and provide relevant details, keeping in mind the previous conversation:
    
    If the query is not about history or the image is not related to Egypt/Alexandria, please state that you cannot analyze it.
    """
    memory.append(f"User: {query}")
    full_query = historical_prompt + "\n" + "\n".join(memory)
    gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                                             generation_config=generation_config,
                                             safety_settings=safety_settings)
    if img:
        response = gemini_model.generate_content([full_query, img])
    else:
        response = gemini_model.generate_content(full_query)

    memory.append(f"answer: {response.text}")
    return response.text

def identify_url_content(url):
    try:
        response = requests.head(url, allow_redirects=True)
        content_type = response.headers.get('content-type')
        if 'image' in content_type:
            return "image"
        elif 'pdf' in content_type:
            return "pdf"
        else:
            parsed_url = urlparse(url)
            file_extension = parsed_url.path.split('.')[-1].lower()
            if file_extension == 'pdf':
                return "pdf"
            else:
                return "unknown"
    except Exception as e:
        st.error(f"Error processing URL: {e}")
        return "unknown"

def handle_answer(query, answer):
    memory.append(f"User: {query}")
    memory.append(f"Answer: {answer}")
    st.write(f"**Question:** {query}")
    st.write(f"**Answer:** {answer}")
    if st.button("Hear the response"):
        voice(answer)

def text_conversation():
    query = st.text_input("Enter a query:")
    if query:
        # Handle URL input
        if query.startswith('http'):
            content_type = identify_url_content(query)
            if content_type == 'image':
                image_conversation(query)
            elif content_type == 'pdf':
                pdf_conversation(query)
        else:
            # Handle text input
            qa = retrieval_qa_pipline()
            res = qa.invoke({"query": query})
            answer = res["result"]
            
            # First check if the database returned an answer 
            if answer.lower() != 'answer is not available in the context':
                handle_answer(query, answer)
            else:
                # If no answer from the database, use the conversational model
                convo = gemini_chat_model(query)
                handle_answer(query, convo)

def voice_conversation():
    query = process_user_voice()
    if query:
        qa = retrieval_qa_pipline()
        res = qa.invoke({"query": query})
        answer = res["result"]
        
        # First check if the database returned an answer 
        if answer.lower() != 'answer is not available in the context':
            handle_answer(query, answer)
        else:
            # If no answer from the database, use the conversational model
            convo = gemini_chat_model(query)
            handle_answer(query, convo)  

def image_conversation(image):
    conv_img_type = st.radio("Select conversation type for the image:",
                            ('Text', 'Voice'),
                            key="image_conv_type")
    
    if conv_img_type == 'Text':
        question = st.text_input("Enter your question about this image:",
                                key="image_question_text")
    elif conv_img_type == 'Voice':
        question = process_user_voice()

    if question: 
        convo = gemini_chat_model(question, image)
        handle_answer(question, convo)

def pdf_conversation(uploaded_file):
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.getvalue())
            pdf_path = temp_pdf.name

        pdf_loader = PyPDFLoader(pdf_path)
        pages = pdf_loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                      chunk_overlap=200)
        context = "\n\n".join(str(p.page_content) for p in pages)
        texts = text_splitter.split_text(context)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        vector_index = Chroma.from_texts(texts, embeddings).as_retriever(
            search_kwargs={"k": 5})

        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                     google_api_key=GOOGLE_API_KEY,
                                     temperature=1)
        qa_chain = RetrievalQA.from_chain_type(
            model, retriever=vector_index, return_source_documents=False)

        prompt_template = """
        Your name is Mora. Answer the question as detailed as possible from the provided context.
        You can summarize the entire context too. Make sure to provide all the details.
        If the answer is not in the provided context, simply say so and inform the user.
        Do not provide the wrong answer. Make sure the answers are in the same language as their question.
        Context:
        {context}
        Question:
        {question}
        Answer:
        """
        prompt = PromptTemplate(template=prompt_template,
                               input_variables=["context", "question"])
        full_prompt = prompt + "\n" + "\n".join(memory)

        stuff_chain = load_qa_chain(model,
                                   chain_type="stuff",
                                   prompt=full_prompt)  

        conv_pdf_type = st.radio(
            "Select conversation type for the PDF:",
            ('Text', 'Voice'),
            key="pdf_conv_type"
        )

        question = None  

        if conv_pdf_type == 'Text':
            question = st.text_input(
                "Enter your question about this PDF:", 
                key="pdf_question_text" 
            )
        elif conv_pdf_type == 'Voice':
            question = process_user_voice()
        else: 
            st.error("Invalid input. Please select 'Text' or 'Voice'.")
            return 

        if question:
            memory.append(f"User: {question}")
            result = stuff_chain.invoke(
                {"input_documents": pages, "question": question},
                return_only_outputs=True
            )
            handle_answer(question, result['output_text'])


def main():
    
    st.title("HistoryVerse Chatbot")
    st.write(
        "Welcome to the HistoryVerse Chatbot. I'm MORA. You can ask any questions about history, historical images, and documents."
    )

    conv_type = st.selectbox("Select conversation type:",
                             ('Text', 'Voice', 'Image', 'PDF'))

    if conv_type == 'Text':
        text_conversation()

    elif conv_type == 'Voice':
        voice_conversation()

    elif conv_type == 'Image':
        img_file = st.file_uploader("Upload an image:",
                                    type=['jpg', 'jpeg', 'png'])
        if img_file is not None:
            image = Image.open(img_file)
            image_conversation(image)  # Pass the image to the image conversation function

    elif conv_type == 'PDF':
        pdf_file = st.file_uploader("Upload a PDF:", type=['pdf'])
        if pdf_file:
            pdf_conversation(pdf_file)  # Pass the pdf file to the pdf conversation function

if __name__ == "__main__":
    main()