from flask import Flask, jsonify, request
import random
import os
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)  # This will allow all origins by default
CORS(app, support_credentials=True)

choices = ['rock', 'paper', 'scissors']

@app.route('/play', methods=['POST'])
@cross_origin(supports_credentials=True)
def play():
    data = request.get_json()
    user_choice = data.get('choice')

    # For now, computer's choice is random
    computer_choice = random.choice(choices)

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
