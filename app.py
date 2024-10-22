from flask import Flask, jsonify, request
import random
import os
import numpy as np
import tensorflow as tf
from flask_cors import CORS

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU


app = Flask(__name__)
CORS(app)  # This will allow all origins by default
CORS(app, support_credentials=True)

# Define a mapping for rock-paper-scissors values
rps_mapping = {
    'rock': 0,
    'paper': 1,
    'scissors': 2
}

# Load the trained LSTM model
model = tf.keras.models.load_model('rock_paper_scissors_model.h5')

choices = ['paper', 'scissors', 'rock']

sequence_length = 5  # We will only keep the last 5 pairs of choices
past_data = [[random.randint(0, 2) for _ in range(2)] for _ in range(sequence_length)]


@app.route('/play', methods=['POST'])
def play():
    global past_data  # Declare past_data as global to modify it
    data = request.get_json()
    user_choice = data.get('choice')

    input_data = np.array(past_data).reshape(1, sequence_length, 2)


    # Predict computer choice
    predicted_move_index = np.argmax(model.predict(input_data))
    computer_choice = choices[predicted_move_index]


    past_data.append([rps_mapping[user_choice], rps_mapping[computer_choice]])
    past_data = past_data[1:]

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
