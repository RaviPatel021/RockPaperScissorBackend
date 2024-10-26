from flask import Flask, jsonify, request
import random
import os
import numpy as np
import torch
import torch.nn as nn
from flask_cors import CORS, cross_origin
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Disable GPU (if you want to use only CPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)
CORS(app, resources={r"/play": {"origins": "*"}})

# Define a mapping for rock-paper-scissors values
rps_mapping = {
    'rock': 0,
    'paper': 1,
    'scissors': 2
}

# Load the trained PyTorch model
class RPSLSTM(nn.Module):
    def __init__(self):
        super(RPSLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=2, hidden_size=128, batch_first=True)
        self.batchnorm1 = nn.BatchNorm1d(20)  # Update the sequence length here
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.batchnorm1(out)
        out = self.dropout1(out)
        out, (hn, cn) = self.lstm2(out)
        out = self.batchnorm2(hn[-1])
        out = self.dropout2(out)
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# Instantiate the model and load the saved state_dict
model = RPSLSTM()
model.load_state_dict(torch.load('rock_paper_scissors_model.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

choices = ['paper', 'scissors', 'rock']
sequence_length = 20  # Keep the last 20 pairs of choices
past_data = [[random.randint(0, 2) for _ in range(2)] for _ in range(sequence_length)]

@app.route('/play', methods=['POST'])
@cross_origin(supports_credentials=True)
def play():
    data = request.get_json()
    user_choice = data.get('choice')

    # For now, computer's choice is random
    computer_choice = random.choice(choices)

    # Prepare input data for the model
    input_data = np.array(past_data).reshape(1, sequence_length, 2)
    input_data_tensor = torch.tensor(input_data, dtype=torch.float32)

    # Log input data shape
    logging.info(f"Input data shape: {input_data_tensor.shape}")

    # Predict computer choice
    try:
        logging.info("Predicting computer choice...")
        with torch.no_grad():
            outputs = model(input_data_tensor)
            predicted_move_index = torch.argmax(outputs).item()
            computer_choice = choices[predicted_move_index]
        logging.info(f"Predicted computer choice: {computer_choice}")
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed!'}), 500

    # Update past data
    past_data.append([rps_mapping[user_choice], rps_mapping[computer_choice]])
    past_data = past_data[1:]

    # Determine the winner
    result = determine_winner(user_choice, computer_choice)
    
    # Log the result
    logging.info(f"Result: {result}")

    # Return the response
    return jsonify({
        'user_choice': user_choice,
        'computer_choice': computer_choice,
        'result': determine_winner(user_choice, computer_choice)
    })

@app.route('/')
def home():
    return "Welcome to the Rock Paper Scissors API! Please use the /play endpoint to play."


def determine_winner(user, computer):
    if user == computer:
        return "It's a tie!"
    if (user == 'rock' and computer == 'scissors') or \
       (user == 'scissors' and computer == 'paper') or \
       (user == 'paper' and computer == 'rock'):
        return "You win!"
    return "Computer wins!"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if not specified by Render
    app.run(host='0.0.0.0', port=port)

