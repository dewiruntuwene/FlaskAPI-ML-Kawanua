from flask import Flask, jsonify, request
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config["ALLOWED_EXTENSIONS"] = set(['png', 'jpg', 'jpeg'])
app.config["UPLOAD_FOLDER"] = "D:/MSIB - STUDI INDEPENDEN/Bangkit/api-ml-kawanua/venv/static/uploads"
app.config['MODEL_FILE'] = 'D:/MSIB - STUDI INDEPENDEN/Bangkit/api-ml-kawanua/indonesian-endangered.h5'
app.config['LABELS_FILE'] = 'D:/MSIB - STUDI INDEPENDEN/Bangkit/api-ml-kawanua/label.txt'

def allowed_file(filename):
    return "." in filename and \
        filename.split(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]

model = load_model("KawanuaModel.h5", compile=False)
with open(app.config['LABELS_FILE'], 'r') as file:
    labels = file.read().splitlines()

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        image = request.files["image"]
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            img = Image.open(image_path).convert("RGB")
            img = img.resize((120, 80))
            img_array = np.asarray(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array.astype(np.float32) / 255.0
            

            prediction = model.predict(img_array)
            index = np.argmax(prediction)
            class_names = labels[index]
            confidence_score = prediction[0][index]

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success predicting",
                },
                "data": {
                    "endangered_prediction": class_names,
                    "confidence": float(confidence_score)
                }
            }), 200
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Client side error"
                },
                "data": None
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None,
        }), 405


if __name__ == "__main__":  # There is an error on this line
    app.run(debug=True, host='0.0.0.0')

