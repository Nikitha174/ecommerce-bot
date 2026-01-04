from flask import Flask, Blueprint, request, jsonify, render_template
from response_handler import get_response
main_routes = Blueprint('main', __name__)
@main_routes.route('/')
def home():
    return render_template('index.html')

@main_routes.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = get_response(user_input)

    if isinstance(response, dict):
        return jsonify(response)
    else:
        return jsonify({'response': response})
