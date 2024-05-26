import os
import sys
# for vision model and images visulization
import io
import requests
from PIL import Image
# for chat with docs
import tempfile
from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
# for identifythe url content type
from urllib.parse import urlparse
# for speech recognition
from SpeechRecognition import recognize_speech, process_user_voice, voice, speek
# for chating with our database
from gemini_data_chat import retrieval_qa_pipline
# for load gemini models api keys
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up the model
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
memory = [] # global conversation_memory
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
    gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                                             generation_config=generation_config,
                                             safety_settings=safety_settings)
    # Determine which model to use based on the presence of an image
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
                convo = gemini_chat_model(question, img)
                print(f"\n> Question:\n {question}")
                print(f"\n> Answer:\n {convo}")
                speak_response = input("\nDo you want to hear the response? (yes or no): ").lower()
                if speak_response in ['yes', 'y']:
                    voice(convo)
                  
        elif conv_img_type == 'v':
            print("\nEnter your question about this image: ")
            question = recognize_speech()
            # question = process_user_voice()  
            if question.lower() in ['quit', 'q', 'exit']:
                break
            else:
                convo = gemini_chat_model(question, img)
                print(f"\n> Question:\n {question}")
                print(f"\n> Answer:\n {convo}")
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
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY, temperature=1)
    qa_chain = RetrievalQA.from_chain_type(
        model, retriever=vector_index, return_source_documents=False
    )

    # Define prompt template
    prompt_template = """
    Your name is Mora. Answer the question as detailed as possible from the provided context.
    you can summrize the all context too. Make sure to provide all the details.
    If the answer is not in the provided context, simply say so and inform the user.
    Do not provide the wrong answer.
    Context:
    {context}
    Question:
    {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    full_prompt = prompt + "\n" + "\n".join(memory)

    # Create Stuff chain
    stuff_chain = load_qa_chain(model, chain_type="stuff", prompt=full_prompt)

    while True:
        # Get user input type
        conv_pdf_type = input("\nPlease enter 't' for text conversation about this pdf or 'v' for voice conversation about this pdf: ")

        if conv_pdf_type == 't':
            question = input("\nEnter your question about this PDF: ")
        elif conv_pdf_type == 'v':
            print("\nEnter your question about this PDF: ")
            question = recognize_speech()
            # question = process_user_voice()
        else:
            print("Invalid input. Please enter 't' or 'v'.")
            continue

        memory.append(f"User: {question}")

        # Exit if requested
        if question.lower() in ['quit', 'q', 'exit']:
            break
        result = stuff_chain.invoke({"input_documents": pages, "question": question}, return_only_outputs=True)
        memory.append(f"answer: {result['output_text']}")
        print(f"\n> Question:\n {question}")
        print(f"\n> Answer:\n {result['output_text']}")
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
            qa = retrieval_qa_pipline()
            res = qa.invoke({"query": query})  # Check for answer in context
            answer, docs = res["result"], res["source_documents"]
            memory.append(f"answer: {answer}") 
            
            if answer == 'answer is not available in the context':
                convo = gemini_chat_model(query)
                print(f"\n> Question:\n {query}")
                print(f"\n> Answer:\n {convo}")
                speak_response = input("\nDo you want to hear the response? (yes or no): ").lower()
                if speak_response in ['yes', 'y']:
                    voice(convo)
            
            else:
                print(f"\n> Question:\n {query}")
                print(f"\n> Answer:\n {answer}")
                speak_response = input("\nDo you want to hear the response? (yes or no): ").lower()
                if speak_response in ['yes', 'y']:
                    voice(answer)  
         
      
      elif conv_type == 'v':
        
        query = recognize_speech()
        # query = process_user_voice()
        
        if query.lower() in ['quit', 'q', 'exit']:
          sys.exit()
        
        elif isinstance(query, str):
            qa = retrieval_qa_pipline()
            res = qa.invoke({"query": query})  # Check for answer in context
            answer, docs = res["result"], res["source_documents"]
            memory.append(f"answer: {answer}")
            
            if answer == 'answer is not available in the context':
                convo = gemini_chat_model(query)
                print(f"\n> Question:\n {query}")
                print(f"\n> Answer:\n {convo}")
                speak_response = input("\nDo you want to hear the response? (yes or no): ").lower()
                if speak_response in ['yes', 'y']:
                    voice(convo)
            else:
                print(f"\n> Question:\n {query}")
                print(f"\n> Answer:\n {answer}")
                speak_response = input("\nDo you want to hear the response? (yes or no): ").lower()
                if speak_response in ['yes', 'y']:
                    voice(answer)  

      elif conv_type == 'i':
         
         image_path = input("\nplease enter yout image path: ")
         image_conversation(image_path)
                
      elif conv_type == 'p':
            
            pdf_path = input("\nEnter the path to your PDF file: ")
            pdf_conversation(pdf_path)

      else:
        print("Invalid input. Please enter 't', 'v', 'i', or 'p'.")               


if __name__ == "__main__":
  main()