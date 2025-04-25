import numpy as np
import math
import cv2
import re


# Hiệu chỉnh độ tương phản
def changeContrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


# Xoay ảnh quanh tâm một góc bất kỳ
def rotate_image(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)


# Tính góc nghiêng của ảnh
def compute_skew(src_img, center_thres):
    h, w = src_img.shape[:2]
    img = cv2.medianBlur(src_img, 3)
    edges = cv2.Canny(img, threshold1=30, threshold2=100, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, 30, minLineLength=w / 1.5, maxLineGap=h / 3.0)

    if lines is None:
        return 0.0

    min_line_y = float('inf')
    min_line_pos = 0

    for i, line in enumerate(lines):
        for x1, y1, x2, y2 in line:
            center_y = (y1 + y2) / 2
            if center_thres and center_y < 7:
                continue
            if center_y < min_line_y:
                min_line_y = center_y
                min_line_pos = i

    angle = 0.0
    count = 0
    for x1, y1, x2, y2 in lines[min_line_pos]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        if abs(ang) <= math.radians(30):
            angle += ang
            count += 1

    return math.degrees(angle / count) if count > 0 else 0.0


# Deskew ảnh (nếu cần tăng độ tương phản)
def deskew(src_img, change_cons, center_thres):
    img = changeContrast(src_img) if change_cons else src_img
    angle = compute_skew(img, center_thres)
    return rotate_image(src_img, angle)


# Sửa ký tự dễ nhầm lẫn
def correct_common_misreads(plate_text):
    corrections = {
        'O': '0', 'I': '1', 'Z': '2',
        'S': '5', 'B': '8', 'Q': '0',
    }
    return ''.join(corrections.get(c, c) for c in plate_text)


# Kiểm tra định dạng biển số Việt Nam
def is_valid_plate(plate_text):
    if not plate_text:
        return False
    plate_text = plate_text.replace("-", "").upper()
    pattern = r'^\d{2}[A-Z]\d{3,4}\.?\d{2}$'
    return bool(re.match(pattern, plate_text))
