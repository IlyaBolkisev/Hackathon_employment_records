from ultralytics import YOLO


def get_localization_model(path):
    model = YOLO(path)
    return model


def detect_document(model, img):
    results = model(img)

    cropped_docs = []
    coordinates = []
    boxes_list = results[0].obb.xyxy.cpu().tolist()
    for i, xyxy in enumerate(boxes_list):
        cropped_docs.append(img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])])
        coordinates.append((xyxy[0], xyxy[1]))
    return cropped_docs, coordinates
