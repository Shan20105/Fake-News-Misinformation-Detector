# Fake-News-Misinformation-Detector
Fake News & Misinformation Detector
A browser-based tool that leverages a machine learning model to analyze the text of online news articles and predict whether they are real or potential misinformation.

In an era of rampant online misinformation, this project provides a simple yet powerful tool for users to quickly assess the credibility of news content directly in their browser.

(Pro Tip: Create a short screen recording or GIF of your project working and place it here. It's a great way to impress anyone viewing your repository.)

Features
Real-Time Analysis: Get predictions for any news article with a single click.

Machine Learning Backend: Utilizes a Python backend with a trained NLP model to classify text.

Simple UI: An intuitive Chrome Extension that seamlessly integrates with your Browse experience.

RESTful API: The frontend and backend communicate via a lightweight Flask API.

Tech Stack
Backend
Python: The core programming language.

Scikit-learn: Used for the TfidfVectorizer and PassiveAggressiveClassifier model.

Pandas: For data manipulation and loading the training dataset.

Flask & Flask-CORS: To create and serve the backend API.

Frontend (Chrome Extension)
JavaScript: For handling user interactions and API calls.

HTML & CSS: To structure and style the extension's popup window.

Setup and Installation

true.csv and fake.csv dataset from kaggle:https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=True.csv

Follow these steps to get the project running on your local machine.

1. Clone the Repository
Bash

git clone https://github.com/Shan_20105/Fake_News_Misinformation_Detector.git
cd Fake_News_Detector
2. Set Up the Backend
Navigate to the backend directory:

Bash

cd backend
Install the required Python packages:

Bash

pip install -r requirements.txt
3. Train the Model (One-Time Step)
Open backend/app.py and uncomment the train_and_save_model() line at the bottom.

Run the script to train the model and generate the .pkl files:

Bash

python app.py
Once the model is trained and the server starts, stop the server (CTRL+C).

Re-comment the train_and_save_model() line in app.py.

4. Start the Server
With the model trained, start the Flask server:

Bash

python app.py
Leave this terminal running. The server will be live at http://127.0.0.1:5000.

5. Load the Chrome Extension
Open Google Chrome and navigate to chrome://extensions.

Enable "Developer mode" in the top-right corner.

Click "Load unpacked" and select the extension folder from this project.

The extension will be loaded. Pin it to your toolbar for easy access.

How to Use
Make sure the Flask server is running in your terminal.

Navigate to any online news article in Chrome.

Click the extension's icon on your toolbar.

In the popup that appears, click the "Analyze Page" button.

The prediction ("Real" or "Fake") will be displayed in the popup.

Project Structure
Fake_News_Detector/

│

├── backend/

│   ├── data/

│   │   ├── Fake.csv

│   │   └── True.csv

│   ├── app.py

│   ├── model.pkl        (Generated after training)

│   ├── vectorizer.pkl   (Generated after training)

│   └── requirements.txt

│

└── extension/

    ├── images/
    
    │   ├── icon16.png
    
    │   ├── icon48.png
    
    │   └── icon128.png
    
    ├── manifest.json

    ├── popup.html
    
    ├── popup.css
    
    └── popup.js
