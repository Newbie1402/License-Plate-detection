import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO

# Cấu hình đường dẫn Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load mô hình YOLO (nếu có mô hình train riêng, thay thế "yolov8n.pt" bằng "best.pt")
model = YOLO("yolov8n.pt")  

# Mở webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dùng YOLO để phát hiện biển số
    results = model(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Tọa độ khung chữ nhật
            conf = box.conf[0].item()  # Độ chính xác
            cls = int(box.cls[0])  # Nhãn dự đoán

            if conf > 0.5:  # Chỉ lấy kết quả có độ chính xác cao
                plate_crop = frame[y1:y2, x1:x2]  # Cắt vùng chứa biển số

                # ---- XỬ LÝ ẢNH TRƯỚC KHI NHẬN DIỆN OCR ----
                plate_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám
                plate_crop = cv2.GaussianBlur(plate_crop, (3, 3), 0)  # Giảm nhiễu
                plate_crop = cv2.adaptiveThreshold(plate_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, 11, 2)  # Tăng độ tương phản

                # ---- NHẬN DIỆN KÝ TỰ TỪ BIỂN SỐ ----
                plate_text = pytesseract.image_to_string(plate_crop, config="--psm 7 --oem 3")
                plate_text = plate_text.strip()  # Xóa khoảng trắng thừa

                # Hiển thị biển số lên màn hình
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ khung
                cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)  # Ghi chữ lên hình
                
                print("Biển số nhận diện:", plate_text)  # In ra terminal

    # Hiển thị video
    cv2.imshow("License Plate Detection", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
