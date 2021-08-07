from PIL import Image
from flask import request, Flask

from main import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        print(
        "Started to predict"
        )
        img = Image.open(request.form['file'])
        print(
        "Received image file"
        )
        img=img.resize((180,180))
        print(
        "Resized image file"
        )
        predicted_class, score = predict(img)
        return predicted_class


if __name__ == '__main__':
    app.run(debug=True)
