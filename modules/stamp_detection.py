import cv2
import numpy as np


def detect_stamps(img):
    stamps_coords = []  # tuples of (x, y, r)
    gray_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    blurred_gray_img = cv2.bilateralFilter(gray_img.copy(), 15, 10, 10)
    gray_final = cv2.medianBlur(blurred_gray_img, 5)
    img_rows = gray_final.shape[0]
    imgcircles = cv2.HoughCircles(gray_final, cv2.HOUGH_GRADIENT, 1, img_rows / 8, param1=100, param2=85, minRadius=150,
                                  maxRadius=300)
    if imgcircles is not None:
        imgcircles = np.uint16(np.around(imgcircles))
        for i in imgcircles[0, :]:
            stamps_coords.append((i[0], i[1], i[2]))

    return stamps_coords

