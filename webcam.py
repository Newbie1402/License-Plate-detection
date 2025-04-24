from PIL import Image, ImageTk
import cv2
import torch
import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import time
import function.utils_rotate as utils_rotate
import function.helper as helper

try:
    yolo_LP_detect = torch.hub.load('ultralytics/yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True)
    yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
    yolo_license_plate.conf = 0.60
    print("Mô hình YOLO đã được tải thành công.")
except Exception as e:
    print("Lỗi khi tải mô hình YOLO:", e)
    exit()

cap = None
video_running = False
detected_plates = {}

def update_textbox(plate, crop_img):
    if plate not in detected_plates:
        print(f"Thêm biển số vào TextBox: {plate}")
        detected_plates[plate] = crop_img
        textbox.insert(tk.END, f"{plate}\n")
        textbox.see(tk.END)

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

    df = pd.DataFrame(data, columns=["Số thứ tự", "Hình ảnh", "Dạng văn bản"])
    file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
    if file_path:
        with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Detected Plates")
            workbook = writer.book
            worksheet = writer.sheets["Detected Plates"]
            for idx, image_path in enumerate(df["Hình ảnh"]):
                worksheet.set_row(idx + 1, 80)
                worksheet.insert_image(idx + 1, 1, image_path, {"x_scale": 0.5, "y_scale": 0.5, "object_position": 1})
        print(f"Xuất file Excel thành công: {file_path}")

def detect_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return
    image = cv2.imread(file_path)
    detect_and_display(image)
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

def detect_video():
    global cap, video_running
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    if not file_path:
        return
    cap = cv2.VideoCapture(file_path)
    video_running = True
    display_video()

def detect_webcam():
    global cap, video_running
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được webcam.")
        return
    video_running = True
    display_video()

def display_video():
    global cap, video_running
    if cap is None or not video_running:
        return
    ret, frame = cap.read()
    if not ret or frame is None:
        video_running = False
        cap.release()
        return

    detect_and_display(frame)
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    if imgtk:
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    video_label.after(30, display_video)

def detect_and_display(frame):
    plates = yolo_LP_detect(frame, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    print(f"Phát hiện được {len(list_plates)} biển số.")
    for plate in list_plates:
        x, y, xmax, ymax = map(int, plate[:4])
        crop_img = frame[y:ymax, x:xmax]
        if crop_img.size == 0:
            continue
        cv2.rectangle(frame, (x, y), (xmax, ymax), (0, 0, 225), 2)
        for cc in range(2):
            for ct in range(2):
                lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                print(f"Biển số đọc được: {lp}")
                if lp != "unknown":
                    cv2.putText(frame, lp, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    update_textbox(lp, crop_img)
                    break

root = tk.Tk()
root.title("License Plate Detection")

button_frame = tk.Frame(root)
button_frame.pack(side=tk.LEFT, padx=10, pady=10)

tk.Label(button_frame, text="Chọn chế độ phát hiện:", font=("Arial", 14)).pack(pady=10)
tk.Button(button_frame, text="Chọn Ảnh", command=detect_image, width=20, height=2).pack(pady=5)
tk.Button(button_frame, text="Chọn Video", command=detect_video, width=20, height=2).pack(pady=5)
tk.Button(button_frame, text="Webcam", command=detect_webcam, width=20, height=2).pack(pady=5)
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
