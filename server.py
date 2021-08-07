from PIL import Image
from flask import request, Flask

import main

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print(
        "Started to predict"
        )
        file = request.files['image']
        img = Image.open(file)
        print(
        "Received image file"
        )
        img=img.resize((180,180))
        print(
        "Resized image file"
        )
        predicted_class, score = main.predict(img)
        return predicted_class


if __name__ == '__main__':
    app.run(debug=True)

