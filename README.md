# HistoryVerse-Chatbot

## It's a powerful AI chatbot assistant using google gemini api

### chatbot features:

1) chat with your data
2) chat with imge
3) voice recognition

### To run this project:

1) Clone the repo using git: git clone https://github.com/PHD-Team/HistoryVerse-Chatbot.git

2) Create and activate a new virtual environment with python=3.10.0.

3) To set up your environment to run the code, first install all requirements using pip: python -m pip install -r requirements.txt.

4) Put your google api key in .env. you can get your api key from here https://ai.google.dev/

4) Put you data you want to chat with it in the SOURCE_DOCUMENTS folder. You can put multiple folders within the SOURCE_DOCUMENTS folder and the code will recursively read your files.

5) Run the following command to ingest all the data: (python ingest.py)

6) In order to chat with your documents, run the following command: (python gemini.py)
