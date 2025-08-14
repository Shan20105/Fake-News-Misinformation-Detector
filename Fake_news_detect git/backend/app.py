# app.py

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Part 1: Model Training (Run this part once) ---

def train_and_save_model():
    """
    This function trains the model on the dataset and saves the
    trained model and the vectorizer to disk.
    """
    # 1. Load Data
    # Download from Kaggle: "Fake and Real News Dataset"
    # Place Fake.csv and True.csv in the same directory as this script.
    try:
        df_fake = pd.read_csv('data/Fake.csv')
        df_true = pd.read_csv('data/True.csv')
    except FileNotFoundError:
        print("Error: Make sure Fake.csv and True.csv are in the directory.")
        return

    # 2. Label the data
    df_fake['label'] = 0  # 0 for Fake
    df_true['label'] = 1  # 1 for True

    # Combine text and title for better feature extraction
    df_fake['full_text'] = df_fake['title'] + ' ' + df_fake['text']
    df_true['full_text'] = df_true['title'] + ' ' + df_true['text']
    
    # Combine the dataframes
    df = pd.concat([df_fake, df_true], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle the data

    # 3. Clean the text
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove punctuation and numbers
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['full_text'] = df['full_text'].apply(clean_text)

    # 4. Split data and vectorize
    X_train, X_test, y_train, y_test = train_test_split(df['full_text'], df['label'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000) # Use top 5000 words
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 5. Train the model
    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(X_train_vec, y_train)

    # 6. Evaluate
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained with Accuracy: {accuracy*100:.2f}%")

    # 7. Save the model and vectorizer
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    with open('vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
        
    print("Model and vectorizer saved successfully.")


# --- Part 2: Flask API ---

app = Flask(__name__)
CORS(app)  # This is crucial to allow requests from the browser extension

# Load the saved model and vectorizer
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    model = None
    vectorizer = None
    print("Model/Vectorizer not found. Please run the training function first.")

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return jsonify({'error': 'Model not loaded. Train the model first.'}), 500

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    text = data['text']
    
    # Clean and vectorize the input text
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text.lower()).strip()
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(vectorized_text)
    
    result = 'Real' if prediction[0] == 1 else 'Fake'
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    # Important: Run this block once to train and save the model.
    # After the files are saved, you can comment out the next line.
    #train_and_save_model() 
    
    # Start the Flask server
    app.run(port=5000, debug=True)