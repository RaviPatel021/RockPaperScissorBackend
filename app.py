from flask import Flask, jsonify, request, session
import random
import os
import csv
import uuid
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)  # This will allow all origins by default
CORS(app, support_credentials=True)


choices = ['rock', 'paper', 'scissors']

# Path to your CSV file
csv_file_path = 'game_data.csv'

def write_to_csv(user_id, user_choice, computer_choice, result):
    # Create the CSV file if it doesn't exist
    file_exists = os.path.isfile(csv_file_path)
    
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            # Write the header if the file is new
            writer.writerow(['user_id', 'user_choice', 'computer_choice', 'result'])
        
        # Write the player's choices and result
        writer.writerow([user_id, user_choice, computer_choice, result])
        
        # Add an empty row to separate each player's data
        writer.writerow([])

@app.route('/play', methods=['POST'])
@cross_origin(supports_credentials=True)
def play():
    data = request.get_json()
    user_choice = data.get('choice')

    # Generate a random computer choice
    computer_choice = random.choice(choices)

    # Check if session already has a user_id, otherwise create one
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())  # Generate a random user ID

    user_id = session['user_id']

    # Determine the result of the game
    result = determine_winner(user_choice, computer_choice)

    # Save the result to the CSV file
    write_to_csv(user_id, user_choice, computer_choice, result)

    return jsonify({
        'user_choice': user_choice,
        'computer_choice': computer_choice,
        'result': result,
        'user_id': user_id  # Sending the user ID for reference
    })

@app.route('/')
def home():
    return "Welcome to the Rock Paper Scissors API! Use the /play endpoint to play."

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
