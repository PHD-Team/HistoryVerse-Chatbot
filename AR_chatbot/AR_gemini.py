import os
import sys
# for speech recognition
import pygame
import speech_recognition as sr
import langid
from gtts import gTTS
from faster_whisper import WhisperModel
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
    if the user questions is not about '{lable}', please state that you cannot analyze it. Make sure the answers are in the same language as their question.
    """
    memory.append(f"User: {query}")
    full_query = historical_prompt + "\n" + "\n".join(memory)
    gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                                             generation_config=generation_config,
                                             safety_settings=safety_settings)
    response = gemini_model.generate_content(full_query)
    memory.append(f"answer: {response.text}")
    return response.text

def handle_answer(query, answer):
    memory.append(f"User: {query}")
    memory.append(f"Answer: {answer}")
    print(f"\nQuestion: {query}")
    print(f"\nAnswer: {answer}")  
    voice_response = input("\nDo you want to hear the response? (yes or no): ").lower()
    if voice_response in ['yes', 'y']:
        hear_response(answer)  


recognizer = sr.Recognizer()
source = sr.Microphone()
whisper_size = 'small'
num_cores = os.cpu_count()
whisper_model = WhisperModel(whisper_size,
                             device='cpu',
                             compute_type='int8',
                             cpu_threads=num_cores,
                             num_workers=num_cores)

def process_user_voice():
    with source as s:
        recognizer.adjust_for_ambient_noise(s, duration=2)
        print("\nListening ... \n")
        audio = recognizer.listen(source)
    try:
        prompt_audio_path = 'prompt.wav'
        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())
        prompt_text = wav_to_text(prompt_audio_path)
        if len(prompt_text.strip()) < 3:
            print("No input detected. Please try again.")
            return None
        else:
            return prompt_text
    except Exception as e:
        print(f'Prompt error: {e}')
        return None

def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text

def hear_response(text, filename='output.mp3'):
    # Remove any special characters (e.g., '*' ) from the text
    spoken_response = text.replace('*', '').replace('\n', '').replace('#', '')
    # Identify the language of the text
    language_code, _ = langid.classify(spoken_response)
    # Create a gTTS object
    tts = gTTS(text=spoken_response, lang=language_code, slow=False, tld='co.uk')
    # Save the audio file
    tts.save(filename)
    # Play the audio file using Pygame
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    # Wait for playback to finish
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    # Stop and quit Pygame
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    # Delete the audio file
    return filename    

def text_conversation(label):
    while True:
        query = input("\nEnter a query: ")

        if query.lower() in ['quit', 'q', 'exit']:
            return
        
        if query:
            convo = gemini_ar_chat_model(query, label)
            handle_answer(query, convo)

def voice_conversation(label):
    while True:
        query = process_user_voice()

        if query.lower() in ['quit', 'q', 'exit']:
            return

        if query:
          convo = gemini_ar_chat_model(query, label)
          handle_answer(query, convo)    

def main():
    while True:
        label = input("\nEnter the name of the historical statue, or 'exit' to end the program: ")

        if label.lower() in ['quit', 'q', 'exit']:
            sys.exit()

        else:
            # Initial explanation of the statue
            query = f"Tell me everything you can about {label}, but try to keep it brief (10 lines or less)."
            convo = gemini_ar_chat_model(query, label)
            handle_answer(query, convo)    

        # Conversation loop
        while True:
            conv_type = input("\nPlease enter 't' for text conversation or enter 'v' for voice conversation, or 'exit' to add a nwe name of the historical statue: ").lower()

            if conv_type.lower() in ['quit', 'q', 'exit']:
                break  # Exit the inner loop
            elif conv_type == 't':
                text_conversation(label)
            elif conv_type == 'v':
                voice_conversation(label)
            else:
                print("Invalid input. Please enter 't' or 'v'.")

if __name__ == "__main__":
    main()