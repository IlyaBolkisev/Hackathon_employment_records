import cv2
import torch
import numpy as np

from recognition_model import get_model, ALPHABET


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
    w, h = img.shape
    new_w = shape_to[0]
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape

    img = img.astype('float32')

    new_h = shape_to[1]
    if h < new_h:
        add_zeros = np.full((w, new_h - h), 255)
        img = np.concatenate((img, add_zeros), axis=1)

    if h > new_h:
        img = cv2.resize(img, (new_h, new_w))

    return img


# TRANSLATE INDICIES TO TEXT
def indicies_to_text(indexes, idx2char):
    text = "".join([idx2char[i] for i in indexes])
    text = text.replace('EOS', '').replace('PAD', '').replace('SOS', '')
    return text


def prediction(model, imgs, idx2char, device):
    """
    params
    ---
    model : nn.Module
    test_dir : str
        path to directory with images
    id2char : dict
        map from indicies to chars

    returns
    ---
    preds : dict
        key : name of image in directory
        value : dict with keys ['p_value', 'predicted_label']
    """
    preds = []
    p_imgs = []
    with torch.no_grad():
        for img in imgs:
            if min(img.shape) == 0:
                preds.append('')
                p_imgs.append(None)
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.GaussianBlur(img, (3, 3), 0)
            img = 255 - cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            img = process_image(img).astype('uint8')
            img = np.expand_dims(img, -1)
            img = img / img.max()
            p_imgs.append(img)
            img = np.transpose(img, (2, 0, 1))

            src = torch.FloatTensor(img).unsqueeze(0).to(device)
            out_indexes = model.predict(src)
            pred = indicies_to_text(out_indexes[0], idx2char)
            preds.append(pred)

    return preds, p_imgs


img = cv2.imread('../../Downloads/table_sample.jpg')[:, :, ::-1]

device = 'cpu'
model = get_model().to(device)
model.load_state_dict(torch.load('weights/recognition_model.pt'))

cols, rows, v_shift, h_shift = get_cols_and_rows(img)

ids = []
segments = [img[rows[i-1]+v_shift:rows[i], :cols[0]] for i in range(1, rows.shape[0])]
idx2char = {idx: char for idx, char in enumerate(ALPHABET)}
pred, processed = prediction(model, segments,  idx2char, device)
print(processed[0].shape)
for i in range(len(pred)):
    if processed[i] is None:
        print(i, 'Skipped')
        continue
    print(i, pred[i])
    cv2.imshow(str(i), processed[i])
    cv2.waitKey()
