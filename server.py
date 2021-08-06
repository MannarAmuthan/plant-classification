from PIL import Image
from flask import request, Flask

from main import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def success():
    if request.method == 'POST':
        img = Image.open(request.form['file'])
        img=img.resize((180,180))
        predicted_class, score = predict(img)
        return predicted_class


if __name__ == '__main__':
    app.run(debug=True)
