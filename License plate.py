import cv2
import imutils
import pytesseract

# Đường dẫn Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Đọc ảnh
image = cv2.imread('car4.jpg')
image = imutils.resize(image, width=500)

cv2.imshow('Bước 1', image)
cv2.waitKey(1)

# Chuyển sang ảnh xám
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Làm mịn ảnh để giảm nhiễu
smooth = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("Bước 2", gray)
cv2.waitKey(1)

# Phát hiện cạnh bằng Canny
corner = cv2.Canny(gray, 170, 200)
cv2.imshow("Bước 3", corner)
cv2.waitKey(1)

# Tìm contour (đường viền)
seg, new = cv2.findContours(corner.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

image1 = image.copy()
cv2.drawContours(image1, seg, -1, (0, 0, 255), 3)
cv2.imshow('Bước 4', image1)
cv2.waitKey(1)

# Chọn 30 contour lớn nhất
seg = sorted(seg, key=cv2.contourArea, reverse=True)[:30]
NoPlate = None

image2 = image.copy()
cv2.drawContours(image2, seg, -1, (0, 255, 0), 3)
cv2.imshow("Bước 5", image2)
cv2.waitKey(1)

# Tìm contour có 4 cạnh (hình chữ nhật)
for i in seg:
    perimeter = cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i, 0.02 * perimeter, True)

    if len(approx) == 4:  # Nếu contour có 4 điểm (hình chữ nhật)
        NoPlate = approx
        x, y, w, h = cv2.boundingRect(i)
        crp_image = image[y:y+h, x:x+w]

        cv2.imwrite('Biển số.png', crp_image)
        break

# Vẽ viền quanh biển số xe
cv2.drawContours(image, [NoPlate], -1, (0, 255, 0), 3)
cv2.imshow("Bước 6", image)
cv2.waitKey(1)

# OCR - Đọc biển số từ ảnh
crp_gray = cv2.cvtColor(crp_image, cv2.COLOR_BGR2GRAY)
text = pytesseract.image_to_string(crp_gray, config='--psm 8')

print("Biển số xe được nhận diện:", text.strip())

cv2.imshow('Ảnh biển số thu nhỏ', crp_gray)
cv2.waitKey(0)
