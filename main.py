import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from recognition_model import get_model


def get_cols_and_rows(img):
    scale = 500 / min(img.shape[:2])
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    horizontal_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    vertical_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    v_lines = vertical_mask.sum(0) / vertical_mask.shape[0] / 255
    v_lines = np.where(v_lines > 0.3)[0]
    shape = v_lines.shape[0]
    v_lines = np.array([v_lines[0]] + [v_lines[i] for i in range(1, v_lines.shape[0])
                                       if (v_lines[i] - v_lines[i - 1] != 1)])
    v_shift = np.ceil(shape / v_lines.shape[0])

    h_lines = horizontal_mask.sum(1) / horizontal_mask.shape[1] / 255
    h_lines = np.where(h_lines > 0.3)[0]
    shape = h_lines.shape[0]
    h_lines = np.array([h_lines[0]] + [h_lines[i] for i in range(1, h_lines.shape[0])
                                       if (h_lines[i] - h_lines[i - 1] != 1)])
    h_shift = np.ceil(shape / h_lines.shape[0])

    return (v_lines / scale).astype(int), (h_lines / scale).astype(int), \
           (v_shift / scale).astype(int), (h_shift / scale).astype(int)


def process_image(img, shape_to=(64, 256)):
    """
    params:
    ---
    img : np.array
    returns
    ---
    img : np.array
    """
    w, h, _ = img.shape
    new_w = shape_to[0]
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h, _ = img.shape

    img = img.astype('float32')

    new_h = shape_to[1]
    if h < new_h:
        add_zeros = np.full((w, new_h - h, 3), 255)
        img = np.concatenate((img, add_zeros), axis=1)

    if h > new_h:
        img = cv2.resize(img, (new_h, new_w))

    return img


img = cv2.imread('../../Downloads/table_sample.jpg')[:, :, ::-1]

device = 'cpu'
model = get_model(device)
model.load_state_dict(torch.load('weights/recognition_model.pt'))
model.eval()

cols, rows, v_shift, h_shift = get_cols_and_rows(img)
for i in range(1, rows.shape[0]):
    # detect row number
    segment = img[rows[i-1]-v_shift:rows[i]+v_shift, :cols[0]]
    segment = process_image(segment)

    with torch.no_grad():
        pred = model.predict(torch.tensor(segment).unsqueeze(0).to(torch.float32))

    plt.imshow(segment)
    plt.show()
