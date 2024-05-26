import os
import sys
# for speech recognition
from SpeechRecognition import recognize_speech, process_user_voice, voice, speek
# for load gemini models api keys
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
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

def gemini_ar_chat_model(query, statue_name):
    historical_prompt = f"""
    Your name is Mora. You are a large language model specialized in analyzing historical statues. 
    You have extensive knowledge about the historical statue '{statue_name}' and can answer questions about its history, significance, location, creator, and any other relevant information.
    """
    memory.append(f"User: {query}")
    full_query = historical_prompt + "\n" + "\n".join(memory)
    gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                                             generation_config=generation_config,
                                             safety_settings=safety_settings)
    response = gemini_model.generate_content(full_query)
    memory.append(f"answer: {response.text}")
    return response.text

import os
import sys
# for speech recognition
from SpeechRecognition import recognize_speech, process_user_voice, voice, speek
# for load gemini models api keys
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
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

def gemini_ar_chat_model(query, lable):
    historical_prompt = f"""
    Your name is Mora. You are a large language model specialized in analyzing history, historical questions, historical images, and historical places in the world, which related to object of '{lable}'. 
    You have extensive knowledge of history related to '{lable}' and can answer questions about historical events, figures, and periods within this context. you can not answer about other fildes or unrelated to '{lable}'.
    if the user questions is not about '{lable}', please state that you cannot analyze it.
    """
    memory.append(f"User: {query}")
    full_query = historical_prompt + "\n" + "\n".join(memory)
    gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                                             generation_config=generation_config,
                                             safety_settings=safety_settings)
    response = gemini_model.generate_content(full_query)
    memory.append(f"answer: {response.text}")
    return response.text

def main():
    while True:
        label = input("\nEnter the name of the historical statue: ")
        while True:
            query = f"Explain this {label} as much as you can, but in 10 lines maximum"
            convo = gemini_ar_chat_model(query, label)
            print(f"\n{convo}")
            speak_response = input("\nDo you want to hear the response? (yes or no): ").lower()
            if speak_response in ['yes', 'y']:
                voice(convo)
                
            while True:
                conv_type = input("\nPlease enter 't' for text conversation or enter 'v' for voice conversation: ").lower()

                if conv_type == 't':
                    query = input("\nEnter a query (or type 'exit' to quit): ")

                    if query.lower() in ['quit', 'q', 'exit']:
                        break

                    elif isinstance(query, str):
                        convo = gemini_ar_chat_model(query, label)
                        print(f"\n> Question:\n {query}")
                        print(f"\n> Answer:\n {convo}")
                        speak_response = input("\nDo you want to hear the response? (yes or no): ").lower()
                        if speak_response in ['yes', 'y']:
                            voice(convo)

                elif conv_type == 'v':
                    query = recognize_speech()

                    if query.lower() in ['quit', 'q', 'exit']:
                        break

                    elif isinstance(query, str):
                        convo = gemini_ar_chat_model(query, label)
                        print(f"\n> Question:\n {query}")
                        print(f"\n> Answer:\n {convo}")
                        speak_response = input("\nDo you want to hear the response? (yes or no): ").lower()
                        if speak_response in ['yes', 'y']:
                            voice(convo)
                else:
                    print("Invalid input. Please enter 't' or 'v'.")

            if query.lower() in ['quit', 'q', 'exit']:
                break

if __name__ == "__main__":
    main()