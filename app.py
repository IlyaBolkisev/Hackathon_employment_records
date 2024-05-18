import numpy as np
import pandas as pd
import base64
import cv2
import io
import os
from PIL import Image

from flask import Flask, request, jsonify, render_template

app = Flask(__name__, static_folder='static')


def get_image(data):
    # if os.path.exists(data[:100]):
    #     with open(data, 'rb') as f:
    #         data = f.read()
    # elif # check if str:
    #     data = base64.b64decode(data)
    img_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return img
def img2b64(img):
    """Converts image to base64 encoded string
    Args:
        img: numpy.ndarray
    Returns:
        base64 encoded string
    """
    buffer = io.BytesIO()

    if isinstance(img, np.ndarray):
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = Image.fromarray(img)
    elif isinstance(img, Image.Image):
        pass

    img.save(buffer, 'PNG')
    buffer.seek(0)

    return base64.b64encode(buffer.getvalue()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def get_main():
    if request.method == 'GET':
        return render_template('main.html')
    if request.method == 'POST':
        doc_file = request.files['doc-file']
        img = get_image(doc_file.read())
        return render_template('report.html', image=img2b64(img))

    

if __name__ == '__main__':
    app.run(host='0.0.0.0')