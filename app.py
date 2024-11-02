from flask import Flask, jsonify, request
import random
import os
import numpy as np
from flask_cors import CORS
import logging
import onnxruntime as ort
from pymongo import MongoClient


# Configure logging
logging.basicConfig(level=logging.INFO)

# Disable GPU (if you want to use only CPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# MongoDB connection setup
mongodb_connection_string = os.environ.get('MONGODB_CONNECTION_STRING')
client = MongoClient(mongodb_connection_string)
db = client['rockpaperscissor']  # Replace with your database name
results_collection = db['results']  # Collection to store game results

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Define a mapping for rock-paper-scissors values
rps_mapping = {
    'rock': 0,
    'paper': 1,
    'scissors': 2
}

# Load the ONNX model
onnx_model_path = 'rock_paper_scissors_model.onnx'
onnx_session = ort.InferenceSession(onnx_model_path)  # Renamed this variable

choices = ['paper', 'scissors', 'rock']
sequence_length = 20  # Keep the last 20 pairs of choices
past_data = [[random.randint(0, 2) for _ in range(2)] for _ in range(sequence_length)]

@app.route('/play', methods=['POST'])
def play():
    global past_data
    data = request.get_json()
    user_choice = data.get('choice')

    user_id = data.get('user_id')
    isRandom = data.get('random')
    logging.info(f"User ID: {user_id}")

    # For now, computer's choice is random
    computer_choice = random.choice(choices)

    if isRandom:
        computer_choice = data.get('computer')

        # Determine the winner
        result = determine_winner(user_choice, computer_choice)

        # Store the result in MongoDB with the username
        store_result(user_id, user_choice, computer_choice, result, isRandom)

        # Return the response
        return jsonify({
            'user_choice': user_choice,
            'computer_choice': computer_choice,
            'result': determine_winner(user_choice, computer_choice)
        })

    else:
        # Prepare input data for the model
        input_data = np.array(past_data, dtype=np.float32).reshape(1, sequence_length, 2)

        # Log input data shape
        logging.info(f"Input data shape: {input_data.shape}")

        # Predict computer choice using ONNX model
        try:
            logging.info("Predicting computer choice...")
            outputs = onnx_session.run(None, {'input': input_data})
            predicted_move_index = np.argmax(outputs[0])
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

        # Store the result in MongoDB with the username
        store_result(user_id, user_choice, computer_choice, result, isRandom)

        # Return the response
        return jsonify({
            'user_choice': user_choice,
            'computer_choice': computer_choice,
            'result': determine_winner(user_choice, computer_choice)
        })
        



@app.route('/')
def home():
    return "Welcome to the Rock Paper Scissors API! Please use the /play endpoint to play."


def store_result(username, user_choice, computer_choice, result, isRandom):
    """Store game result in MongoDB."""
    game_result = {
        'user_choice': user_choice,
        'computer_choice': computer_choice,
        'result': result,
        'random': isRandom,
    }

    # Check if the user already has a results document
    existing_user = results_collection.find_one({'username': username})
    
    if existing_user:
        # If user exists, append to the results array
        results_collection.update_one(
            {'username': username},
            {'$push': {'results': game_result}}  # Append the game result to the results array
        )
    else:
        # If user does not exist, create a new document with an array
        results_collection.insert_one({
            'username': username,
            'results': [game_result]  # Create an array with the first game result
        })


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
    app.run(host='0.0.0.0', port=port, debug=True)

