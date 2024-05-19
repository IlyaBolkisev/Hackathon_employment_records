import json

import cv2
import numpy as np
import torch

from modules.localization import detect_document, get_localization_model
from modules.parse_table import get_cols_and_rows, validate_rows, get_blocks, generate_empty_dict
from modules.recognition import get_recognition_model
from modules.stamp_detection import detect_stamps


def wrapper(imgs):
    layout = []

    device = 'cpu'

    # Load models
    localization_model = get_localization_model('weights/localization_model.pt')
    recognition_model = get_recognition_model().to(device)
    recognition_model.load_state_dict(torch.load('weights/recognition_model.pt'))

    for img in imgs:
        layout.append([])
        tables, coords = detect_document(localization_model, img)
        for table, (x_t, y_t) in zip(tables, coords):
            layout[-1].append([])
            cols, rows, v_shift, h_shift = get_cols_and_rows(table)
            if cols is None:
                del layout[-1][-1]
                continue
            rows_height = np.median([rows[i] - rows[i-1] for i in range(1, len(rows))]).astype(int)
            rows = validate_rows(rows, rows_height)

            id_segments = [table[rows[i-1]:rows[i], :cols[0]] for i in range(1, len(rows))]
            block = get_blocks(id_segments, rows)

            stamps = detect_stamps(table)
            cols = [0]+list(cols)+[table.shape[1]]
            for i in range(len(block)):
                layout[-1][-1].append(generate_empty_dict())
                for j, category in enumerate(['id', 'date', 'info', 'doc']):
                    if j + 1 < len(cols):
                        layout[-1][-1][-1][category] = {"text": "", "x": int(x_t+cols[i]), "y": int(y_t+block[i][0]),
                                                        "w": int(cols[i+1] - cols[i]), "h": int(block[i][1] - block[i][0])}

                for x, y, r in stamps:
                    if block[i][0] < y <= block[i][1]:
                        layout[-1][-1][-1]['stamp'] = {"text": "", "x": int(x_t+x-r), "y": int(y_t+y-r),
                                                       "w": int(2*r), "h": int(2*r)}
    return layout
