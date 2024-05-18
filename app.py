import numpy as np
import pandas as pd
import base64
import cv2
import io
import os
import fitz
from PIL import Image

from flask import Flask, request, jsonify, render_template

import json
app = Flask(__name__, static_folder='static')

placeholder_img = cv2.imread('placeholder.jpg', cv2.IMREAD_COLOR)

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

def parse_pdf(file_data):
    images = []
    doc = fitz.open(None, file_data, 'pdf')
    for i in range(len(doc)):
        page_img = doc.load_page(i).get_pixmap(dpi=200)
        im = np.asarray(np.frombuffer(page_img.samples, dtype=np.uint8).reshape((page_img.h, page_img.w, page_img.n)))
        im = im.copy()
        im.flags.writeable = True
        images.append(im)

    return images

@app.route('/', methods=['GET', 'POST'])
def get_main():
    if request.method == 'GET':
        return render_template('main.html')
    if request.method == 'POST':
        doc_file = request.files.getlist("doc-file")
        # img = get_image(doc_file.read())
        if len(doc_file) == 1 and doc_file[0].filename.split('.')[-1].lower() == 'pdf':
            imgs = parse_pdf(doc_file[0].read())
        else:
            imgs = [get_image(file.read()) for file in doc_file]
        
        im = imgs[0]
        print(im.shape)
        with open('layout2.json', 'r', encoding='utf-8') as f:
            layout = json.load(f)
        
        preds = []
        for row in layout:
            for ent in row:
                ent_dict = row[ent]
                start = (int(ent_dict['x']), int(ent_dict['y']))
                end = (int(ent_dict['x'])+int(ent_dict['w']), int(ent_dict['y'])+int(ent_dict['h']))
                if (ent == 'id'and ent_dict['text'] == '-1') or ent_dict['text'] == "":
                    continue
                crop = img2b64(im[start[1]:end[1], start[0]:end[0]]) \
                    if (end[1]-start[1] + end[0]-start[0]) > 0 \
                    else img2b64(placeholder_img)
                preds.append({
                    'tag': ent,
                    'pred': ent_dict['text'],
                    'crop': crop
                })
                # preds.append(ent_dict['text'])
                row[ent]['crop'] = img2b64(im[start[1]:end[1], start[0]:end[0]])
                cv2.rectangle(im, start, end, (255, 0, 0), 3)
                cv2.putText(im, ent_dict['text'], (int(ent_dict['x']), max(0, int(ent_dict['y'])-5)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        imgs = [im]
        # print(preds)
        return render_template('report.html', images=[img2b64(img) for img in imgs], preds=layout)

    

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)