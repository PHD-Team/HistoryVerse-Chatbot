## HistoryVerse-Chatbot: Your Powerful AI Chat Assistant

**Harnessing the power of the Google Gemini API, HistoryVerse-Chatbot brings you a versatile and customizable AI assistant.**

### Features:

1. **Chat with Your Data:** Interact directly with your own documents and files. 
2. **Chat with Images:** Ask questions about images you provide.
3. **Voice Recognition:**  Speak your questions and receive spoken responses.
4. **Generate Answers from the Internet:** Get answers to your questions by tapping into the vast knowledge of the web. 
   - **Customization:** Modify the prompt used for internet searches to tailor the type of responses you receive. Control the scope and style of the answers by providing specific instructions.
5. **Chat with Memory:** The chatbot can remember past interactions, making conversations more natural and engaging.

### Running the Project:

1. **Clone the Repository:** Use git to clone the repository: `git clone https://github.com/PHD-Team/HistoryVerse-Chatbot.git`

2. **Virtual Environment:** Create and activate a new virtual environment with Python 3.10.0 (or a compatible version).

3. **Install Dependencies:** Install all required packages using pip: `python -m pip install -r requirements.txt`

4. **Google API Key:**  
   - Obtain your Google API key from [https://ai.google.dev/].
   - Store the key in a `.env` file within the project directory.

5. **OpenAI API Key (for Voice Recognition):**
   - Obtain your OpenAI API key from [https://platform.openai.com/account/api-keys].
   - Store the key in a `constants.py` file within the project directory.
   - Inside `constants.py`, replacing `put your openai api key` with your actual key.
   
6. **Data Ingestion:**
   - Place your data files (documents, text, etc.) in the `SOURCE_DOCUMENTS` folder. 
   - You can organize multiple folders within `SOURCE_DOCUMENTS`, and the code will recursively read all files.
   - To ingest your data, run the command: `python ingest.py`

7. **Chatting with Your Data:**  To interact with your uploaded data only, run: `python gemini_data_chat.py`

8. **Chatting with the Internet (with Customization):**
    - To use the internet for answers and customize the type of questions it can handle, run: `python gemini.py`
    - **Customization:** The prompt used to query the internet can be adjusted in the `gemini.py` file.
    - By modifying the `prompt` string, you can provide context, instructions, or limitations to the AI, ensuring it answers questions within your desired scope.