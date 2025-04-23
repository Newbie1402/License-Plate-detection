from PIL import Image, ImageTk
import cv2
import torch
import os
import time
import tkinter as tk
from tkinter import filedialog
import function.utils_rotate as utils_rotate
import function.helper as helper

# Load YOLO models
try:
    yolo_LP_detect = torch.hub.load('ultralytics/yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True)
    yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
    yolo_license_plate.conf = 0.60
    print("Mô hình YOLO đã được tải thành công.")
except Exception as e:
    print("Lỗi khi tải mô hình YOLO:", e)
    exit()

# Global variables
cap = None
video_running = False
detected_plates = set()  # Set to store detected license plates

# Function to update the TextBox with detected plates
def update_textbox(plate):
    if plate not in detected_plates:
        print(f"Thêm biển số vào TextBox: {plate}")  # Debug: In biển số được thêm
        detected_plates.add(plate)
        textbox.insert(tk.END, f"{plate}\n")
        textbox.see(tk.END)  # Scroll to the bottom

# Function to detect license plates in an image
def detect_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return
    image = cv2.imread(file_path)

    # Resize image to fit the display frame (640x480)
    image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)

    detect_and_display(image)

    # Hiển thị ảnh trong video_label
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

# Function to detect license plates in a video
def detect_video():
    global cap, video_running
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    if not file_path:
        return
    cap = cv2.VideoCapture(file_path)
    video_running = True
    display_video()

# Function to detect license plates using webcam
def detect_webcam():
    global cap, video_running
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được webcam.")
        return
    video_running = True
    display_video()

# Function to display video in the Tkinter frame
def display_video():
    global cap, video_running
    if cap is None or not video_running:
        return
    ret, frame = cap.read()
    if not ret:
        video_running = False
        cap.release()
        return

    # Giảm FPS bằng cách tạm dừng
    time.sleep(0.2)  # Tạm dừng 100ms giữa các khung hình (giảm FPS)

    # Phóng to khung hình (zoom)
    height, width, _ = frame.shape
    zoom_factor = 1.5  # Tăng kích thước khung hình lên 1.5 lần
    center_x, center_y = width // 2, height // 2
    new_width, new_height = int(width / zoom_factor), int(height / zoom_factor)
    x1, y1 = center_x - new_width // 2, center_y - new_height // 2
    x2, y2 = center_x + new_width // 2, center_y + new_height // 2
    frame = frame[y1:y2, x1:x2]  # Cắt khung hình
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)  # Resize về kích thước cố định

    # Gửi khung hình vào hàm nhận diện
    detect_and_display(frame)

    # Convert frame to ImageTk format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Schedule the next frame
    video_label.after(10, display_video)

# Function to detect and display license plates
def detect_and_display(frame):
    plates = yolo_LP_detect(frame, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    print(f"Phát hiện được {len(list_plates)} biển số.")  # Debug: In số lượng biển số phát hiện được
    for plate in list_plates:
        x = int(plate[0])  # xmin
        y = int(plate[1])  # ymin
        w = int(plate[2] - plate[0])  # xmax - xmin
        h = int(plate[3] - plate[1])  # ymax - ymin
        crop_img = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), color=(0, 0, 225), thickness=2)
        for cc in range(0, 2):
            for ct in range(0, 2):
                lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                print(f"Biển số đọc được: {lp}")  # Debug: In biển số đọc được
                if lp != "unknown":
                    cv2.putText(frame, lp, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    update_textbox(lp)  # Update the TextBox with the detected plate
                    break

# Create Tkinter GUI
root = tk.Tk()
root.title("License Plate Detection")

# Main frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(side=tk.LEFT, padx=10, pady=10)

tk.Label(button_frame, text="Chọn chế độ phát hiện:", font=("Arial", 14)).pack(pady=10)
tk.Button(button_frame, text="Chọn Ảnh", command=detect_image, width=20, height=2).pack(pady=5)
tk.Button(button_frame, text="Chọn Video", command=detect_video, width=20, height=2).pack(pady=5)
tk.Button(button_frame, text="Webcam", command=detect_webcam, width=20, height=2).pack(pady=5)

# Frame for video display
video_frame = tk.Frame(root, width=640, height=480, bg="black")
video_frame.pack(side=tk.TOP, padx=10, pady=10)
video_label = tk.Label(video_frame, width=640, height=480)  # Cố định kích thước video_label
video_label.pack()

# Frame for detected plates
text_frame = tk.Frame(root)
text_frame.pack(side=tk.BOTTOM, padx=10, pady=10, fill=tk.X)

tk.Label(text_frame, text="Biển số đã phát hiện:", font=("Arial", 12)).pack(pady=5)
textbox = tk.Text(text_frame, height=5, width=80)  # Giảm chiều cao của TextBox
textbox.pack()

root.mainloop()