from flask import Flask, jsonify, request
import random

app = Flask(__name__)

choices = ['rock', 'paper', 'scissors']

@app.route('/play', methods=['POST'])
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

def determine_winner(user, computer):
    if user == computer:
        return "It's a tie!"
    if (user == 'rock' and computer == 'scissors') or \
       (user == 'scissors' and computer == 'paper') or \
       (user == 'paper' and computer == 'rock'):
        return "You win!"
    return "Computer wins!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
