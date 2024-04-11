import os
import sys
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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
def model():
  text_model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                generation_config=generation_config,
                                safety_settings=safety_settings)
  return text_model


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


def voice(text):
    spoken_response = text.replace('*', '')
    tts_engine = pyttsx3.init()
    voices = tts_engine.getProperty('voices')
    tts_engine.setProperty('voice', voices[1].id) #changing index changes voices but ony 0(male) and 1(female) are working here
    tts_engine.say(spoken_response)
    tts_engine.runAndWait()


def main():
    
    while True:
      conv_type = input("Please enter 't' for text conversation or enter 'v' for voice conversation: ").lower()
      
      if conv_type == 't':
        
        query = input("\nEnter a query: ")
        
        if query.lower() in ['quit', 'q', 'exit']:
          sys.exit()
            
        elif isinstance(query, str):
          convo = model().start_chat(history=[])
          convo.send_message(query)
          print(convo.last.text)
          voice(convo.last.text)
      elif conv_type == 'v':
        query = recognize_speech()
        
        if query.lower() in ['quit', 'q', 'exit']:
          sys.exit()
        
        elif isinstance(query, str):
          convo = model().start_chat(history=[])
          convo.send_message(query)
          print(convo.last.text)
          voice(convo.last.text)  

if __name__ == "__main__":
  main()