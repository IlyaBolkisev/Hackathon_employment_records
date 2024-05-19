import cv2
import numpy as np
from matplotlib import pyplot as plt


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
    if len(v_lines) == 0:
        return None, None, None, None
    shape = v_lines.shape[0]
    v_lines = np.array([v_lines[0]] + [v_lines[i] for i in range(1, v_lines.shape[0])
                                       if (v_lines[i] - v_lines[i - 1] != 1)])
    v_shift = np.ceil(shape / v_lines.shape[0])

    h_lines = horizontal_mask.sum(1) / horizontal_mask.shape[1] / 255
    h_lines = np.where(h_lines > 0.1)[0]
    if len(h_lines) == 0:
        return None, None, None, None
    shape = h_lines.shape[0]
    h_lines = np.array([h_lines[0]] + [h_lines[i] for i in range(1, h_lines.shape[0])
                                       if (h_lines[i] - h_lines[i - 1] != 1)])
    h_shift = np.ceil(shape / h_lines.shape[0])

    return (v_lines / scale).astype(int), (h_lines / scale).astype(int), \
           (v_shift / scale).astype(int), (h_shift / scale).astype(int)


def validate_rows(rows, rows_height, neighborhood=0.1):
    rows_height = (rows_height * (1 - neighborhood), rows_height * (1 + neighborhood))
    new_rows = []
    previous = rows[0]
    for i in range(1, len(rows)):
        if rows[i] - previous > rows_height[1]:
            previous = rows[i]
        elif rows_height[0] <= rows[i] - previous <= rows_height[1]:
            new_rows.append(previous)
            if i + 1 == len(rows):
                new_rows.append(rows[i])
            previous = rows[i]
    return new_rows


def get_blocks(segments, rows):
    is_full = []
    for i, segment in enumerate(segments):
        gray = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        is_full.append(sum(thresh.sum(axis=1) > 0) / thresh.shape[0] > 0.5)

    borders = []
    for b, c in zip(is_full[1:], rows[1:-1]):
        if b:
            borders.append(c)

    borders.insert(0, rows[0])
    borders.append(rows[-1])
    return [(borders[i-1], borders[i]) for i in range(1, len(borders))]


def generate_empty_dict():
    d = {}
    for category in ['id', 'date', 'info', 'doc', 'stamp']:
        d[category] = {"text": "", "x": 0, "y": 0, "w": 0, "h": 0}
    return d