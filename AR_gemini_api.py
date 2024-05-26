import logging
import argparse

from flask import Flask, request, jsonify

from AR_gemini import gemini_ar_chat_model
from SpeechRecognition import recognize_speech, process_user_voice, voice, speek, text_to_speech

app = Flask(__name__)

@app.route("/text_convo", methods=["GET", "POST"])
def text_conversation():
    data = request.get_json()
    lable = data.get("statue_name")
    query = data.get("query")
    speak_response = data.get("speak", False)
 
    if query.lower() in ['quit', 'q', 'exit']:
        return jsonify({"message": "Session ended."})

    if isinstance(query, str):
        convo = gemini_ar_chat_model(query, lable)
        response_text = convo
        response = jsonify({"Question": query}, {"Answer": convo})
        if speak_response:
            voice(response_text)         
    return response

@app.route("/voice_convo", methods=["GET", "POST"])
def voice_conversation():
    data = request.get_json()
    lable = data.get("lable")
    speak_response = data.get("speak", False)

    query = process_user_voice()
    if not query:
        response = jsonify("No input detected. Please try again.")
    elif query.lower() in ['quit', 'q', 'exit']:
        return jsonify({"message": "Session ended."})
    else:
        convo = gemini_ar_chat_model(query, lable)
        response_text = convo
        response = jsonify({"Question": query}, {"Answer": convo})
        if speak_response:
            voice(response_text)        
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000, help="Port to run the API on. Defaults to 5000.")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.2",
        help="Host to run the UI on. Defaults to 127.0.0.2. "
        "Set to 0.0.0.1 to make the UI externally "
        "accessible from other devices.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    app.run(debug=False, host=args.host, port=args.port)