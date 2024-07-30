import pyrebase

config = {
  "apiKey": "YOUR_API_KEY", 
  "authDomain": "YOUR_PROJECT_ID.firebaseapp.com", 
  "databaseURL": "https://YOUR_PROJECT_ID-default-rtdb.firebaseio.com/",
  "projectId": "YOUR_PROJECT_ID",
  "storageBucket": "YOUR_PROJECT_ID.appspot.com",
  "messagingSenderId": "YOUR_MESSAGING_SENDER_ID",
  "appId": "1:YOUR_MESSAGING_SENDER_ID:web:YOUR_APP_ID_SUFFIX", 
  "measurementId": "YOUR_MEASUREMENT_ID",
  "serviceAccount": "path/to/your/serviceAccountKey.json", 
  "databaseURL": "https://YOUR_PROJECT_ID-default-rtdb.firebaseio.com/"
}

firebase = pyrebase.initialize_app(config)
localpath = "output.mp3" 
cloudpath = "speak_audio_file/output.mp3"

#for uploading 
def upload_to_firebase(localpath, cloudpath):
  storage = firebase.storage()
  storage.child(cloudpath).put(localpath)
  print(f"File uploaded to: {cloudpath}")  # Optional: Print confirmation

# for downloading
def download_from_firebase(cloudpath, localpath):
  """Downloads a file from Firebase Storage."""
  storage = firebase.storage()
  storage.child(cloudpath).download(localpath)
  print(f"File downloaded to: {localpath}")  # Optional: Print confirmation  

# Getting the file URL
def get_firebase_url(cloudpath):
  """Retrieves the download URL for a file in Firebase Storage."""
  storage = firebase.storage()
  file_url = storage.child(cloudpath).get_url(None)
  return file_url

# upload_audio = upload_to_firebase(localpath, cloudpath) 
# firebase_url = get_firebase_url(cloudpath)
# print(f"Firebase URL: {firebase_url}")