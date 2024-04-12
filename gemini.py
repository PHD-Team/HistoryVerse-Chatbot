import os
import sys
# for speech recognition
import speech_recognition as sr
import pyttsx3
# for vision model and images visulization
import io
import requests
from PIL import Image
# for load gemini models api keys
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
def gemini_text_model(query):
  text_model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                generation_config=generation_config,
                                safety_settings=safety_settings)
  text_response = text_model.generate_content(query)
  return text_response

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


def main():
    
    while True:
      conv_type = input("\nPlease enter 't' for text conversation or enter 'v' for voice conversation or enter 'i' for images conversation: ").lower()
      
      if conv_type == 't':
        
        query = input("\nEnter a query: ")
        
        if query.lower() in ['quit', 'q', 'exit']:
          sys.exit()

        elif query.startswith('http'):
          
          image_path = query
          response = requests.get(image_path)
          if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
         
          while True:
            conv_img_type = input("\nPlease enter 't' for text conversation about this image or enter 'v' for voice conversation about this image: ").lower()
            
            if conv_img_type == 't':

              question = input("\nenter your question about this image: ")
              if question.lower() in ['quit', 'q', 'exit']:
                break
              else:
                convo = gemini_vision_model(question, img)
                print('\n',convo)
                voice(convo)

            elif conv_img_type == 'v':
              
              print("\nenter your question about this image: ")
              question = recognize_speech()
              if question.lower() in ['quit', 'q', 'exit']:
                 break
              else:
                convo = gemini_vision_model(question, img)
                print('\n',convo)
                voice(convo)    
            
        elif isinstance(query, str):
          convo = gemini_text_model(query)
          print('\n',convo.text)
          voice(convo.text)
      
      elif conv_type == 'v':
        
        print("\nEnter a query: ")
        query = recognize_speech()
        
        if query.lower() in ['quit', 'q', 'exit']:
          sys.exit()
        
        elif isinstance(query, str):
          convo = gemini_text_model(query)
          print('\n',convo.text)
          voice(convo.text)

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
                voice(convo)

            elif conv_img_type == 'v':
              
              print("\nenter your question about this image: ")
              question = recognize_speech()
              if question.lower() in ['quit', 'q', 'exit']:
                 break
              else:
                convo = gemini_vision_model(question, img)
                print('\n',convo)
                voice(convo)    


if __name__ == "__main__":
  main()