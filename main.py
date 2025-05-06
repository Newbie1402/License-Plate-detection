import sys
import os
from tkinter import *
from PIL import Image, ImageTk
import cv2
import torch
import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from ultralytics import YOLO
import function.utils_rotate as utils_rotate
import function.helper as helper
from sys import exit


def resource_path(relative_path):
    """Lấy đường dẫn đúng tới file khi chạy bằng PyInstaller (.exe)"""
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS  # đường dẫn tạm khi chạy file .exe
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
# Tải mô hình YOLO
def load_models():
    try:
        detect_model_path = resource_path('model/LicensePlate/Detect_Plate.pt')
        ocr_model_path = resource_path('model/OCR2/ocr2.pt')

        yolo_LP_detect = YOLO(detect_model_path)
        yolo_license_plate = YOLO(ocr_model_path)
        yolo_license_plate.conf = 0.60  # Ngưỡng confidence cho OCR

        print("Mô hình YOLOv8 đã được tải thành công.")
        return yolo_LP_detect, yolo_license_plate
    except Exception as e:
        print(f"Lỗi khi tải mô hình YOLOv8: {e}")
        sys.exit()

yolo_LP_detect, yolo_license_plate = load_models()

cap = None
video_running = False
detected_plates = {}

# Cập nhật TextBox với biển số đã phát hiện
def update_textbox(plate, crop_img):
    if plate not in detected_plates:
        print(f"Thêm biển số vào TextBox: {plate}")
        detected_plates[plate] = crop_img
        textbox.insert(tk.END, f"{plate}\n")
        textbox.see(tk.END)

# Xuất kết quả vào file Excel
def export_to_excel():
    if not detected_plates:
        print("Không có biển số nào để xuất.")
        return

    output_dir = "detected_plates"
    os.makedirs(output_dir, exist_ok=True)

    data = []
    for idx, (plate, crop_img) in enumerate(detected_plates.items()):
        image_path = os.path.join(output_dir, f"plate_{idx+1}.png")
        cv2.imwrite(image_path, crop_img)
        data.append([idx + 1, image_path, plate])

    file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
    if not file_path:
        print("Không chọn file để lưu.")
        return

    try:
        df = pd.DataFrame(data, columns=["Số thứ tự", "Hình ảnh", "Dạng văn bản"])
        with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Detected Plates")
            workbook = writer.book
            worksheet = writer.sheets["Detected Plates"]
            for idx, image_path in enumerate(df["Hình ảnh"]):
                worksheet.set_row(idx + 1, 80)
                worksheet.insert_image(idx + 1, 1, image_path, {"x_scale": 0.5, "y_scale": 0.5, "object_position": 1})
        print(f"Xuất file Excel thành công: {file_path}")
    except Exception as e:
        print(f"Đã xảy ra lỗi khi xuất file Excel: {e}")

def detect_image():
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh để nhận diện biển số",
        filetypes=[("Image files", "*.jpg *.png *.jpeg")]
    )
    if not file_path:
        print("Không có ảnh nào được chọn.")
        return

    image = cv2.imread(file_path)
    if image is None:
        print(f"Không thể đọc ảnh từ: {file_path}")
        return

    detect_and_display(image)
    display_image(image)


# Phát hiện biển số từ video
def detect_video():
    global cap, video_running
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    if not file_path:
        return
    cap = cv2.VideoCapture(file_path)
    video_running = True
    stop_video_button.pack(pady=5)
    stop_webcam_button.pack_forget()
    display_video()

# Phát hiện biển số từ webcam
def detect_webcam():
    global cap, video_running
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được webcam.")
        return
    video_running = True
    stop_webcam_button.pack(pady=5)
    stop_video_button.pack_forget()
    display_video()

# Hiển thị video liên tục
def display_video():
    global cap, video_running
    if cap is None or not video_running:
        return
    ret, frame = cap.read()
    if not ret or frame is None:
        video_running = False
        cap.release()
        return

    # Phát hiện và hiển thị biển số
    detect_and_display(frame)
    display_image(frame)

    # Lặp lại việc hiển thị video
    if video_running:
        video_label.after(30, display_video)

def detect_and_display(frame):
    # Phát hiện biển số từ ảnh bằng YOLOv8
    results = yolo_LP_detect(frame, imgsz=640)
    list_plates = results[0].boxes.data.tolist()

    # Duyệt qua các biển số phát hiện được
    for plate in list_plates:
        x, y, xmax, ymax = map(int, plate[:4])
        crop_img = frame[y:ymax, x:xmax]
        if crop_img.size == 0:
            continue
        cv2.rectangle(frame, (x, y), (xmax, ymax), (0, 0, 225), 2)

        # Phát hiện văn bản biển số
        detected_texts = []
        for cc in range(2):
            for ct in range(2):
                lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                if lp != "unknown":
                    detected_texts.append(lp)

        if detected_texts:
            final_plate = max(set(detected_texts), key=detected_texts.count)
            print(f"Biển số đọc được: {final_plate}")

            if final_plate not in detected_plates:
                cv2.putText(frame, final_plate, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                update_textbox(final_plate, crop_img)


# Hiển thị ảnh lên giao diện
# def display_image(image):
#     frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(frame)
#     imgtk = ImageTk.PhotoImage(image=img)
#     video_label.imgtk = imgtk
#     video_label.configure(image=imgtk)

def display_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    max_width = 640

    if w > max_width:
        scale = max_width / w
        img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)))

    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    # Sửa ở đây
    video_label.config(image=img_tk)
    video_label.image = img_tk


# Dừng video
def stop_video():
    global cap, video_running
    if video_running:
        video_running = False
        if cap is not None:
            cap.release()
        print("Video đã dừng.")
        export_to_excel()
    video_label.configure(image='')
    stop_video_button.pack_forget()

# Dừng webcam
def stop_webcam():
    global cap, video_running
    if video_running:
        video_running = False
        if cap is not None:
            cap.release()
        print("Webcam đã dừng.")
    video_label.configure(image='')
    stop_webcam_button.pack_forget()

# Giao diện người dùng Tkinter
root = tk.Tk()
root.title("License Plate Detection")

button_frame = tk.Frame(root)
button_frame.pack(side=tk.LEFT, padx=10, pady=10)

tk.Label(button_frame, text="Chọn chế độ phát hiện:", font=("Arial", 14)).pack(pady=10)
tk.Button(button_frame, text="Chọn Ảnh", command=detect_image, width=20, height=2).pack(pady=5)
tk.Button(button_frame, text="Chọn Video", command=detect_video, width=20, height=2).pack(pady=5)
tk.Button(button_frame, text="Webcam", command=detect_webcam, width=20, height=2).pack(pady=5)
stop_video_button = tk.Button(button_frame, text="Dừng Video", command=stop_video, width=20, height=2, bg="#FF5722")
stop_webcam_button = tk.Button(button_frame, text="Thoát Webcam", command=stop_webcam, width=20, height=2, bg="#FF5733", fg="white")

stop_video_button.pack_forget()
stop_webcam_button.pack_forget()

tk.Button(button_frame, text="Export to Excel", command=export_to_excel, width=20, height=2, bg="#FFC107").pack(pady=5)

video_frame = tk.Frame(root, width=640, height=480, bg="black")
video_frame.pack(side=tk.TOP, padx=10, pady=10)
video_label = tk.Label(video_frame, width=640, height=480)
video_label.pack()

text_frame = tk.Frame(root)
text_frame.pack(side=tk.BOTTOM, padx=10, pady=10, fill=tk.X)

tk.Label(text_frame, text="Biển số đã phát hiện:", font=("Arial", 12)).pack(pady=5)
textbox = tk.Text(text_frame, height=5, width=80)
textbox.pack()

root.mainloop()
