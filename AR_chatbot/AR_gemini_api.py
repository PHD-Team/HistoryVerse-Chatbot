import os
import logging
import argparse
import uuid
import pyrebase
from faster_whisper import WhisperModel
import langid

from gtts import gTTS
from dotenv import load_dotenv


from flask import Flask, request, jsonify

# for load gemini models api keys
import google.generativeai as genai
from dotenv import load_dotenv

app = Flask(__name__)

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

memory = []
current_statue_name = None


def gemini_ar_chat_model(query, lable):
    historical_prompt = f"""
    Your name is Mora. You are a large language model specialized in analyzing history, historical questions, historical images, and historical places in the world, which related to object of '{lable}'. 
    You have extensive knowledge of history related to '{lable}' and can answer questions about historical events, figures, and periods within this context. you can not answer about other fildes or unrelated to '{lable}'.
    if the user questions is not about '{lable}', please state that you cannot analyze it. Make sure the answers are in the same language as their question.
    """
    memory.append(f"User: {query}")
    full_query = historical_prompt + "\n" + "\n".join(memory)
    gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                                             generation_config=generation_config,
                                             safety_settings=safety_settings)
    response = gemini_model.generate_content(full_query)
    memory.append(f"answer: {response.text}")
    return response.text

whisper_size = 'small'
num_cores = os.cpu_count()
whisper_model = WhisperModel(whisper_size, device='cpu', compute_type='int8', cpu_threads=num_cores, num_workers=num_cores)

def process_audio_path(audio_file_path):
    try:
        prompt_text = wav_to_text(audio_file_path)
        if len(prompt_text.strip()) < 3:  # Check for very short input
            print("No input detected. Please try again.")
            return None
        else:
            print('user: ' + prompt_text)
            return prompt_text
    except Exception as e:
        print('prompt error: ', e)
        return None
    
def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text

def voice(text, filename='output.mp3'):
    # Remove any special characters (e.g., '*' ) from the text
    spoken_response = text.replace('*', '').replace('\n', '').replace('#', '')
    # Identify the language of the text
    language_code, _ = langid.classify(spoken_response)
    # Create a gTTS object
    tts = gTTS(text=spoken_response, lang=language_code, slow=False, tld='co.uk')
    # Save the audio file
    tts.save(filename)
    return filename  # Return the filename, no need to play or delete

config = {
  "apiKey": "AIzaSyA_uHpMvKLevHnBjFxTe3pX3S0G7SeCbBo",
  "authDomain": "historyversechatbot.firebaseapp.com",
  "databaseURL": "https://historyversechatbot-default-rtdb.firebaseio.com",
  "projectId": "historyversechatbot",
  "storageBucket": "historyversechatbot.appspot.com",
  "messagingSenderId": "879881281124",
  "appId": "1:879881281124:web:52d4ea7c1c384157f40971",
  "measurementId": "G-SK4S7D8W4W",
  "serviceAcount": "serviceAcount.json",
  "databaseURL": "https://historyversechatbot-default-rtdb.firebaseio.com/"
}

firebase = pyrebase.initialize_app(config)
localpath = "output.mp3" 
cloudpath = "speak_audio_file/output.mp3"

#for uploading 
def upload_to_firebase(localpath, cloudpath):
  storage = firebase.storage()
  storage.child(cloudpath).put(localpath)
  print(f"File uploaded to: {cloudpath}")  # Optional: Print confirmation

# Getting the file URL
def get_firebase_url(cloudpath):
  """Retrieves the download URL for a file in Firebase Storage."""
  storage = firebase.storage()
  file_url = storage.child(cloudpath).get_url(None)
  return file_url

def process_response(query, convo, speak_response):
    response = jsonify({"Question": query, "Answer": convo})
    if speak_response:
        audio_filename = f"{uuid.uuid4()}.mp3" 
        audio_path = f"speak_audio_file/{audio_filename}"
        audio_filename = voice(convo, audio_filename)
        upload_to_firebase(audio_filename, audio_path) 
        firebase_url = get_firebase_url(audio_path)
        os.remove(audio_filename)
        response = jsonify({"Question": query}, {"Answer": convo}, {"audio_url": firebase_url})

    return response


@app.route("/start_convo", methods=["GET", "POST"])
def start_conversation():
    global current_statue_name
    data = request.get_json()
    statue_name = data.get("statue_name")
    speak_response = data.get("speak", False)
    if not statue_name:
        return jsonify({"error": "Missing statue_name"}), 400

    current_statue_name = statue_name  # Store the statue name

    intro_query = f"Tell me everything you can about {statue_name}, but try to keep it brief (10 lines or less)."
    intro_convo = gemini_ar_chat_model(intro_query, statue_name)

    # Process the response (potentially with audio)
    response = process_response(intro_query, intro_convo, speak_response)

    return response

@app.route("/text_convo", methods=["GET", "POST"])
def text_conversation():
    global current_statue_name
    data = request.get_json()
    query = data.get("query")
    speak_response = data.get("speak", False)

    if not current_statue_name or not query:
        return jsonify({"error": "Missing statue_name or query"}), 400

    if isinstance(query, str):
        answer = gemini_ar_chat_model(query, current_statue_name)

        # Process the response (potentially with audio)
        response = process_response(query, answer, speak_response)

    return response

@app.route("/voice_convo", methods=["GET", "POST"])
def voice_conversation():
    global current_statue_name
    data = request.get_json()
    speak_response = data.get("speak", False)
    audio_file_path = data.get("audio_file_path") 

    if not current_statue_name:
        return jsonify({"error": "Missing statue_name"}), 400

    query = process_audio_path(audio_file_path)
    if not query:
        return jsonify("No input detected. Please try again.")
    else:
        answer = gemini_ar_chat_model(query, current_statue_name)

        # Process the response (potentially with audio)
        response = process_response(query, answer, speak_response)

    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000, help="Port to run the API on. Defaults to 5000.")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",  # Change to 0.0.0.0 to make it externally accessible
        help="Host to run the UI on. Defaults to 0.0.0.0 to make the UI externally accessible from other devices.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    app.run(debug=False, host=args.host, port=args.port)