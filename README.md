# 🌱 Plant Disease Prediction Framework
This project leverages Convolutional Neural Networks (CNN) to predict plant diseases using images of leaves. It includes a dashboard for managing datasets, predicting diseases, and showcasing results with a confusion matrix for model evaluation.

# 🚀 Features
Home Page:
Displays detailed plant-related data.

About Page:
Information about datasets, accuracy, and validation.

Plant Disease Page:
Upload a photo of a leaf, and the model predicts the disease with high accuracy.

Plant Village Page:
A virtual plant nursery showcasing various plants for users.

# 🛠️ Technologies Used
Frontend: HTML, CSS
Backend: Python (Flask/Django)
Model: TensorFlow/Keras for CNN
Database: SQLite/MySQL (as needed)

# 📂 Project Structure
Plant-Disease-Prediction/
│
├── static/                 # CSS, JS, and Images  
├── templates/              # HTML files for UI  
├── models/                 # Pre-trained CNN model  
├── dataset/                # Plant images for training and testing  
├── app.py                  # Main backend logic  
├── requirements.txt        # Python dependencies  
├── README.md               # Project documentation (this file)  
└── LICENSE                 # Project license (if applicable)  

# ⚡ Getting Started
## 1. Clone the Repository
git clone https://github.com/yourusername/plant-disease-prediction.git
cd plant-disease-prediction
## 2. Set Up the Virtual Environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
## 3. Install Dependencies
pip install -r requirements.txt
## 4. Download the Pre-trained Model
 * Place the model file (plant_disease_model.h5) into the models/ directory.
 * If the model isn't included, retrain it using the dataset provided.

## 5. Run the Application
streamlite run main.py

## 6. Access the Dashboard
Open your browser and navigate to:
http://127.0.0.1:5000

   
