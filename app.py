from flask import Flask, send_from_directory
from routes import main_routes
import os

app = Flask(__name__, static_folder='static')

# Register your Blueprint
app.register_blueprint(main_routes)

# ✅ Route to serve images from 'imgs/' folder (outside static/)
@app.route('/imgs/<path:filename>')
def serve_image(filename):
    return send_from_directory('imgs', filename)

if __name__ == "__main__":
    app.run(debug=True)
