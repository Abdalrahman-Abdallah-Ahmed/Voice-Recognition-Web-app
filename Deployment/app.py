from flask import Flask, request, render_template
import torch
from torchvision import transforms
from PIL import Image
import librosa 
import numpy as np
from torch.nn import functional as F
import torch.nn as nn 
import os
from werkzeug.utils import secure_filename

class modell(nn.Module):
    def __init__(self):
        super(modell, self).__init__()
        self.fc1 = nn.Linear(100,1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 7)
    def forward(self, X):
        X = torch.relu(self.fc1(X))
        X = torch.relu(self.fc2(X))
        X = torch.relu(self.fc3(X))
        X = torch.relu(self.fc4(X))
        return F.softmax(self.fc5(X), dim=1)

model = modell()

app = Flask(__name__)

# Load the model
model.load_state_dict(torch.load('C:\\Users\\YAMAMA\\OneDrive\\سطح المكتب\\deploy\\models\\Sound_Classefier_model.pth'))
model.eval()
labelsss = ['7masa', 'Abdelrahman', 'Geda', 'Haridy', 'Hossam', 'Khaled', 'Zain']

# Define transformation
def feature_extractor(file):
    audio, sr = librosa.load(file)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=100)
    scaled = np.mean(mfccs_features.T, axis=0)
    return scaled

@app.route('/')
def home():
    return render_template("index.html")

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get record from request
        record_file = request.files['record']
        # Preprocess record
        record = feature_extractor(record_file)
        record = torch.tensor([record], dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            predictedd = model(record)
            output = predictedd.numpy()[0]
        
        # Return prediction
        return render_template("index.html", prediction_text=f"its {labelsss[output.argmax()]}")
    
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
