🌐 Language Detection using FastAPI

A Machine Learning-based web API that detects the language of a given text input. This application uses Naive Bayes, CountVectorizer, and is deployed using FastAPI.


---

✅ 1. Project Overview

This project identifies the language of a given sentence or text input using NLP and Machine Learning.
A trained classification model is loaded in a FastAPI backend and provides real-time language prediction via API.


---

📂 2. Project Structure

📁 Language-Detection-FastAPI
│
├── ldmodel.pkl                # Trained ML model
├── ldvector.pkl           # CountVectorizer object
├── lang.py                  # FastAPI application (main backend file)
├── ld.csv                 # (Optional) Dataset used for training
└── README.md                # Project documentation


---

💻 3. Technologies Used

Technology	Purpose

Python	Programming Language
FastAPI	Web Framework for API
scikit-learn	Machine Learning Model
CountVectorizer	Text Feature Extraction
Naive Bayes	Classification Algorithm
Uvicorn	ASGI Server to run FastAPI
Pickle	Model & Vectorizer Serialization



---

⚙ 4. Installation & Setup

✅ Step 1: Clone or download the project

cd project-folder-name

✅ Step 2: Install dependencies

pip install fastapi uvicorn scikit-learn numpy pickle-mixin

✅ Step 3: Place the trained files

Make sure these files are available in the same folder as your FastAPI code: ✔ ldmodel.pkl
✔ ldvector.pkl


---

🚀 5. Run the FastAPI App

uvicorn lang:app --reload

(Replace lang with your file name if different)


---

✅ 6. FastAPI Code (Include in README under Code Section)

from fastapi import FastAPI
import pickle
import numpy as np
import uvicorn

app = FastAPI()

model = pickle.load(open('ldmodel.pkl', 'rb'))
vectorizer = pickle.load(open('ldvector.pkl', 'rb'))

@app.get("/")
def home():
    return {"message": "Welcome to Language Detection API"}

@app.post("/predict")
def predict_language(text: str):
    x = vectorizer.transform([text]).toarray()
    prediction = model.predict(x)[0]

    if(prediction == 0):
        output = 'Arabic'
    if(prediction == 1):
        output = 'Danish'
    if(prediction == 2):
        output = 'Dutch'
    if(prediction == 3):
        output = 'English'
    if(prediction == 4):
        output = 'French'
    if(prediction == 5):
        output = 'German'
    if(prediction == 6):
        output = 'Greek'
    if(prediction == 7):
        output = 'Hindi'
    if(prediction == 8):
        output = 'Italian'
    if(prediction == 9):
        output = 'Kannada'
    if(prediction == 10):
        output = 'Portuguese'
    if(prediction == 12):
        output = 'Russian'
    if(prediction == 13):
        output = 'Spanish'
    if(prediction == 14):
        output = 'Swedish'
    if(prediction == 15):
        output = 'Tamil'
    if(prediction == 16):
        output = 'Turkish'

    return {"predicted_language": output}

if _name_ == "_main_":
    uvicorn.run(app, host="127.0.0.1", port=8000)


---

🧪 7. Test the API

✅ Option 1: Browser Homepage

Open:

http://127.0.0.1:8000/

✅ Option 2: Swagger UI (API Testing)

FastAPI provides a built-in UI for testing:

http://127.0.0.1:8000/docs

✅ Example Input:

{
  "text": "Hola, como estas?"
}

✅ Example Output:

{
  "predicted_language": "Spanish"
}


---

📊 8. Model Overview

Algorithm Used: Multinomial Naive Bayes

Feature Extraction: CountVectorizer (BoW Model)

Supported Languages:
English     
French        
Spanish        
Portugeese     
Italian        
Russian        
Sweedish       
Malayalam      
Dutch          
Arabic         
Turkish        
German         
Tamil          
Danish        
Kannada   
Greek        
Hindi           



---

📌 9. Future Improvements

✔ Deploy on cloud (Render / AWS / Railway)
✔ Add more languages
✔ Use a deep learning model for better accuracy
✔ Create a frontend using Streamlit or React

 ---

🎥 Demo Video

You can watch the project in action here:

▶ [Click to watch LanguageDetection.mp4](./LanguageDetection.mp4)

Or download directly from this repository.

✍ Author

Kousik Chakraborty

👤 Kousik Chakraborty
📧 Email: www.kousik.c.in@gmail.com
🔗 GitHub Profile: https://github.com/iamkousikc-create18
🔗 Project Repository: https://github.com/iamkousikc-create18/Language-Detection

