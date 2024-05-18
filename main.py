import cv2
import torch

from modules.localization import get_localization_model, detect_document
from modules.parse_table import get_cols_and_rows
from modules.recognition import get_recognition_model, prediction, idx2char
from modules.stamp_detection import detect_stamps

device = 'cpu'

# Load models
localization_model = get_localization_model('weights/localization_model.pt')
recognition_model = get_recognition_model().to(device)
recognition_model.load_state_dict(torch.load('weights/recognition_model.pt'))


scan = cv2.imread('./test_images/scan_sample.jpg')[:, :, ::-1]

tables = detect_document(localization_model, scan)

for table in tables:
    for i, (x, y, r) in enumerate(detect_stamps(table)):
        cv2.imshow(str(i), table[y-r:y+r, x-r:x+r])
    cv2.waitKey()

    cols, rows, v_shift, h_shift = get_cols_and_rows(table)
    circles = detect_stamps(table)
    ids = []
    segments = [table[rows[i-1]+v_shift:rows[i], :cols[0]] for i in range(1, rows.shape[0])]
    pred, processed = prediction(recognition_model, segments,  idx2char, device)
    print(processed[0].shape)
    for i in range(len(pred)):
        if processed[i] is None:
            print(i, 'Skipped')
            continue
        print(i, pred[i])
        cv2.imshow(str(i), processed[i])
        cv2.waitKey()
