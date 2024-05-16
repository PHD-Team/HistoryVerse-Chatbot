import os
import speech_recognition as sr
import pyttsx3
import pyaudio
import time
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import google.generativeai as genai
from openai import OpenAI
import constants

import warnings
warnings.filterwarnings("ignore", message=r"torch.utils._pytree._register_pytree_node is deprecated")

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
    tts_engine.setProperty('voice', voices[0].id) #changing index changes voices but ony 0(male) and 1(female) are working here
    tts_engine.say(spoken_response)
    tts_engine.runAndWait()

load_dotenv()
GOOGLE_API_KEY = genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

reconizer = sr.Recognizer()
source = sr.Microphone()

whisper_size = 'small'
num_cores = os.cpu_count()
whisper_model = WhisperModel(whisper_size, device='cpu', compute_type='int8', cpu_threads=num_cores, num_workers=num_cores)

def process_user_voice():
    with source as s:
        reconizer.adjust_for_ambient_noise(s, duration=2)
        print("\nListening ... \n")
        audio = reconizer.listen(source)
    try:
        prompt_audio_path = 'prompt.wav'
        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())
        prompt_text = wav_to_text(prompt_audio_path)
        if len(prompt_text.strip()) < 3:  # Check for very short input
            print("No input detected. Please try again.")
            return None
        else:
            print('user: ' + prompt_text)
            return prompt_text
    except Exception as e:
        print('prompt error: ', e)
        return None
    
def speek(text):
    player_stream = pyaudio.PyAudio().open(format = pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream_start = False
    with client.audio.speech.with_streaming_response.create(
        model = "tts-1",
        voice="alloy",
        response_format= "pcm",
        input= text,
    ) as response:
        silence_threshold = 0.01
        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_start:
                player_stream.write(chunk)
            elif max(chunk) > silence_threshold:
                player_stream.write(chunk)
                stream_start = True   

def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text

# # for load gemini models api keys
# import google.generativeai as genai
# from dotenv import load_dotenv

# load_dotenv()

# GOOGLE_API_KEY = genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Set up the model
# generation_config = {
#   "temperature": 0.9,
#   "top_p": 1,
#   "top_k": 1,
#   "max_output_tokens": 2048,
# }

# safety_settings = [
#   {
#     "category": "HARM_CATEGORY_HARASSMENT",
#     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#   },
#   {
#     "category": "HARM_CATEGORY_HATE_SPEECH",
#     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#   },
#   {
#     "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#   },
#   {
#     "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
#     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#   },
# ]

# # use it if you want to chat with text only
# def gemini_text_model(query):
#     text_model = genai.GenerativeModel(model_name="gemini-1.0-pro",
#                                       generation_config=generation_config,
#                                       safety_settings=safety_settings)
#     text_response = text_model.generate_content(query)
#     return text_response.text

# while True:
#     query = process_user_voice()
#     if query.lower() in ['quit', 'q', 'exit']:
#         sys.exit()
#     elif isinstance(query, str):
#         convo = gemini_text_model(query)
#         print("\n\n> Question:")
#         print(query)
#         print("\n> Answer:")
#         print(convo)
#         speak_response = input("\nDo you want to hear the response? (yes or no): ").lower()
#         if speak_response in ['yes', 'y']:
#             speek(convo)