import os
import sys
# for speech recognition
import speech_recognition as sr
import pyttsx3
# for vision model and images visulization
import io
import requests
from PIL import Image
# for chat with docs
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
# for identifythe url content type
from urllib.parse import urlparse
# for load gemini models api keys
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up the model
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

# use it if you want to chat with text only
def gemini_text_model(query):
  text_model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                generation_config=generation_config,
                                safety_settings=safety_settings)
  text_response = text_model.generate_content(query)
  return text_response.text

# use it if you want to chat with image too
def gemini_vision_model(query, img):
   vision_model = genai.GenerativeModel(model_name="gemini-pro-vision",
                                generation_config=generation_config,
                                safety_settings=safety_settings)
   vision_response = vision_model.generate_content([query, img])
   return vision_response.text

# take the voice from the user and convert it to text
def recognize_speech():
    reconizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Listening ...")
        audio = reconizer.listen(source)
    
    try:
        user_prompt = reconizer.recognize_google(audio)
        print(f"You said: {user_prompt}")
        return user_prompt
    
    except sr.UnknownValueError:
        print("I couldn't understand what you said.")
        exit()
    
    except sr.RequestError as e:
        print("Could not connect to Google Speech.")
        exit()

# convert the model answer to voice
def voice(text):
    spoken_response = text.replace('*', '')
    tts_engine = pyttsx3.init()
    voices = tts_engine.getProperty('voices')
    tts_engine.setProperty('voice', voices[1].id) #changing index changes voices but ony 0(male) and 1(female) are working here
    tts_engine.say(spoken_response)
    tts_engine.runAndWait()

def identify_url_content(url):
    
    try:
        response = requests.head(url, allow_redirects=True)
        content_type = response.headers.get('content-type')

        if 'image' in content_type:
            return "image"
        elif 'pdf' in content_type:
            return "pdf"
        else:
            # Analyze file extension as a fallback
            parsed_url = urlparse(url)
            file_extension = parsed_url.path.split('.')[-1].lower()
            if file_extension == 'pdf':
                return "pdf"
            else:
                return "unknown"  # Return "unknown" for other types
    except Exception as e:
        print(f"Error processing URL: {e}")
        return "unknown"

def image_conversation(image_source):
    
    if image_source.startswith('http'):
        response = requests.get(image_source)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
        else:
            print(f"Error downloading image: {response.status_code}")
            return
    else:
        if not os.path.isfile(image_source):
            raise SystemExit("Invalid image path")
        img = Image.open(image_source)

    while True:
        conv_img_type = input("\nPlease enter 't' for text conversation about this image or enter 'v' for voice conversation about this image: ").lower()
        if conv_img_type == 't':
            question = input("\nEnter your question about this image: ")
            if question.lower() in ['quit', 'q', 'exit']:
                break
            else:
                convo = gemini_vision_model(question, img)
                print('\n', convo)
                speak_response = input("\nDo you want to hear the response? (yes or no): ").lower()
                if speak_response in ['yes', 'y']:
                    voice(convo)
                  
        elif conv_img_type == 'v':
            print("\nEnter your question about this image: ")
            question = recognize_speech()  
            if question.lower() in ['quit', 'q', 'exit']:
                break
            else:
                convo = gemini_vision_model(question, img)
                print('\n', convo)
                speak_response = input("\nDo you want to hear the response? (yes or no): ").lower()
                if speak_response in ['yes', 'y']:
                    voice(convo)      

def pdf_conversation(file_source):
    # Load PDF content
    if os.path.isfile(file_source):
        pdf_loader = PyPDFLoader(file_source)
        pages = pdf_loader.load_and_split()
    else:
        response = requests.get(file_source, stream=True)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
                temp_pdf.write(response.content)
                pdf_loader = PyPDFLoader(temp_pdf.name)
                pages = pdf_loader.load_and_split()
        else:
            raise SystemExit(f"Error downloading PDF: {response.status_code}")

    # Prepare text for embedding and retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_text(context)

    # Create embeddings and vector index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 5})

    # Set up model and chain for question answering
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(
        model, retriever=vector_index, return_source_documents=False
    )

    # Define prompt template
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    Make sure to provide all the details. 
    If the answer is not in the provided context, simply say so and inform the user. 
    Do not provide the wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create Stuff chain
    stuff_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    while True:
        # Get user input type
        conv_pdf_type = input("\nPlease enter 't' for text conversation about this pdf or 'v' for voice conversation about this pdf: ")

        if conv_pdf_type == 't':
            question = input("\nEnter your question about this PDF: ")
        elif conv_pdf_type == 'v':
            print("\nEnter your question about this PDF: ")
            question = recognize_speech()
        else:
            print("Invalid input. Please enter 't' or 'v'.")
            continue

        # Exit if requested
        if question.lower() in ['quit', 'q', 'exit']:
            break
        result = stuff_chain.invoke({"input_documents": pages, "question": question}, return_only_outputs=True)
        print("\n> Question:")
        print(question)
        print("\n> Answer:")
        print(result['output_text'])
        speak_response = input("\nDo you want to hear the response? (yes or no): ").lower()
        if speak_response in ['yes', 'y']:
           voice(result['output_text'])

def main():
    
    while True:
      conv_type = input("\nPlease enter 't' for text conversation or enter 'v' for voice conversation or enter 'i' for images conversation or enter 'p' for pdf conversation: ").lower()
      
      if conv_type == 't':
        
        query = input("\nEnter a query: ")
        
        if query.lower() in ['quit', 'q', 'exit']:
          sys.exit()

        elif query.startswith('http'):
           content_type = identify_url_content(query)
           if content_type == 'image':
              image_conversation(query)
           
           elif content_type == 'pdf':
              pdf_conversation(query)

        elif isinstance(query, str):
          convo = gemini_text_model(query)
          print(convo)
          speak_response = input("\nDo you want to hear the response? (yes or no): ").lower()
          if speak_response in ['yes', 'y']:
             voice(convo)
         
      
      elif conv_type == 'v':
        
        print("\nEnter a query: ")
        query = recognize_speech()
        
        if query.lower() in ['quit', 'q', 'exit']:
          sys.exit()
        
        elif isinstance(query, str):
          convo = gemini_text_model(query)
          print('\n',convo)
          speak_response = input("\nDo you want to hear the response? (yes or no): ").lower()
          if speak_response in ['yes', 'y']:
             voice(convo)
          
      elif conv_type == 'i':
         
         image_path = input("\nplease enter yout image path: ")
         if not os.path.isfile(image_path):
            raise SystemExit("invaild image path")
         img = Image.open(image_path)
         
         while True:
            conv_img_type = input("\nPlease enter 't' for text conversation about this image or enter 'v' for voice conversation about this image: ").lower()
            
            if conv_img_type == 't':

              question = input("\nenter your question about this image: ")
              if question.lower() in ['quit', 'q', 'exit']:
                break
              else:
                convo = gemini_vision_model(question, img)
                print('\n',convo)
                speak_response = input("\nDo you want to hear the response? (yes or no): ").lower()
                if speak_response in ['yes', 'y']:
                    voice(convo)
                    
            elif conv_img_type == 'v':
              
              print("\nenter your question about this image: ")
              question = recognize_speech()
              if question.lower() in ['quit', 'q', 'exit']:
                 break
              else:
                convo = gemini_vision_model(question, img)
                print('\n',convo)
                speak_response = input("\nDo you want to hear the response? (yes or no): ").lower()
                if speak_response in ['yes', 'y']:
                    voice(convo)
                
      elif conv_type == 'p':
            
            pdf_path = input("\nEnter the path to your PDF file: ")
            if not os.path.isfile(pdf_path):
                raise SystemExit("Invalid PDF path")
            pdf_conversation(pdf_path)

      else:
        print("Invalid input. Please enter 't', 'v', 'i', or 'p'.")               


if __name__ == "__main__":
  main()