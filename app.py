from flask import Flask, jsonify, request
import random
import os
import numpy as np
import tensorflow as tf
from flask_cors import CORS

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'  # Prevent TensorFlow from allocating GPU memory

# Ensure TensorFlow uses CPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_visible_devices([], 'GPU')  # This will hide GPU devices

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
model = tf.keras.models.load_model('rock_paper_scissors_model.keras')

choices = ['paper', 'scissors', 'rock']

sequence_length = 5  # We will only keep the last 5 pairs of choices
past_data = [[random.randint(0, 2) for _ in range(2)] for _ in range(sequence_length)]


@app.route('/play', methods=['POST'])
def play():
    global past_data  # Declare past_data as global to modify it
    
    # Log the start of the request
    logging.info("Received request to play.")
    
    data = request.get_json()
    user_choice = data.get('choice')
    
    # Log the user's choice
    logging.info(f"User choice: {user_choice}")

    # Validate user choice
    if user_choice not in rps_mapping:
        logging.error(f"Invalid user choice: {user_choice}")
        return jsonify({'error': 'Invalid choice! Please choose rock, paper, or scissors.'}), 400

    # Prepare input data for the model
    input_data = np.array(past_data).reshape(1, sequence_length, 2)

    # Log input data shape
    logging.info(f"Input data shape: {input_data.shape}")

    # Predict computer choice
    try:
        logging.info("Predicting computer choice...")
        predicted_move_index = np.argmax(model.predict(input_data))
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
        'result': result
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
