# from PIL import Image, ImageTk
# import cv2
# import torch
# import os
# import time
# import tkinter as tk
# from tkinter import filedialog
# import numpy as np
# import pandas as pd  # Thêm pandas để xuất file Excel
# import function.utils_rotate as utils_rotate
# import function.helper as helper

# # Load YOLO models
# try:
#     yolo_LP_detect = torch.hub.load('ultralytics/yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True)
#     yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
#     yolo_license_plate.conf = 0.60
#     print("Mô hình YOLO đã được tải thành công.")
# except Exception as e:
#     print("Lỗi khi tải mô hình YOLO:", e)
#     exit()

# # Global variables
# cap = None
# video_running = False
# detected_plates = {}  # Dictionary để lưu biển số và hình ảnh biển số

# # Function to update the TextBox with detected plates
# def update_textbox(plate, crop_img):
#     if plate not in detected_plates:
#         print(f"Thêm biển số vào TextBox: {plate}")  # Debug: In biển số được thêm
#         detected_plates[plate] = crop_img  # Lưu hình ảnh biển số vào dictionary
#         textbox.insert(tk.END, f"{plate}\n")
#         textbox.see(tk.END)  # Scroll to the bottom

# # Image processing functions
# def sharpen_image(image):
#     """Tăng độ sắc nét của ảnh."""
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5, -1],
#                        [0, -1, 0]])
#     return cv2.filter2D(image, -1, kernel)

# def enhance_contrast(image):
#     """Cải thiện độ tương phản bằng CLAHE."""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(gray)
#     return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# def denoise_image(image):
#     """Giảm nhiễu bằng bộ lọc Bilateral."""
#     return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

# # Function to export detected plates to Excel
# def export_to_excel():
#     # Lấy danh sách biển số từ TextBox
#     if not detected_plates:
#         print("Không có biển số nào để xuất.")
#         return

#     # Tạo thư mục tạm để lưu hình ảnh biển số
#     output_dir = "detected_plates"
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Tạo danh sách dữ liệu cho Excel
#     data = []
#     for idx, (plate, crop_img) in enumerate(detected_plates.items()):
#         # Lưu hình ảnh biển số vào file
#         image_path = os.path.join(output_dir, f"plate_{idx+1}.png")
#         cv2.imwrite(image_path, crop_img)

#         # Thêm dữ liệu vào danh sách
#         data.append([idx + 1, image_path, plate])

#     # Tạo DataFrame và lưu vào file Excel
#     df = pd.DataFrame(data, columns=["Số thứ tự", "Hình ảnh", "Văn bản"])
#     file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
#     if file_path:
#         with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
#             df.to_excel(writer, index=False, sheet_name="Detected Plates")

#             # Thêm hình ảnh vào file Excel
#             workbook = writer.book
#             worksheet = writer.sheets["Detected Plates"]
#             for idx, image_path in enumerate(df["Hình ảnh"]):
#                 worksheet.insert_image(idx + 1, 1, image_path, {"x_scale": 0.5, "y_scale": 0.5})

#         print(f"Xuất file Excel thành công: {file_path}")

# # Function to detect license plates in an image
# def detect_image():
#     file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
#     if not file_path:
#         return
#     image = cv2.imread(file_path)

#     # Resize image to fit the display frame (640x480)
#     image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)

#     detect_and_display(image)

#     # Hiển thị ảnh trong video_label
#     frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(frame)
#     imgtk = ImageTk.PhotoImage(image=img)
#     video_label.imgtk = imgtk
#     video_label.configure(image=imgtk)

# # Function to detect license plates in a video
# def detect_video():
#     global cap, video_running
#     file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
#     if not file_path:
#         return
#     cap = cv2.VideoCapture(file_path)
#     video_running = True
#     display_video()

# # Function to detect license plates using webcam
# def detect_webcam():
#     global cap, video_running
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Không mở được webcam.")
#         return
#     video_running = True
#     display_video()

# # Function to display video in the Tkinter frame
# def display_video():
#     global cap, video_running
#     if cap is None or not video_running:
#         return
#     ret, frame = cap.read()
#     if not ret:
#         video_running = False
#         cap.release()
#         return

#     # Giảm FPS bằng cách tạm dừng
#     time.sleep(0.1)  # Tạm dừng 100ms giữa các khung hình (giảm FPS)

#     # Phóng to khung hình (zoom)
#     height, width, _ = frame.shape
#     zoom_factor = 1.5  # Tăng kích thước khung hình lên 1.5 lần
#     center_x, center_y = width // 2, height // 2
#     new_width, new_height = int(width / zoom_factor), int(height / zoom_factor)
#     x1, y1 = center_x - new_width // 2, center_y - new_height // 2
#     x2, y2 = center_x + new_width // 2, center_y + new_height // 2
#     frame = frame[y1:y2, x1:x2]  # Cắt khung hình
#     frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)  # Resize về kích thước cố định

#     # Gửi khung hình vào hàm nhận diện
#     detect_and_display(frame)

#     # Convert frame to ImageTk format
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(frame)
#     imgtk = ImageTk.PhotoImage(image=img)
#     video_label.imgtk = imgtk
#     video_label.configure(image=imgtk)

#     # Schedule the next frame
#     video_label.after(10, display_video)

# # Function to detect and display license plates
# def detect_and_display(frame):
#     plates = yolo_LP_detect(frame, size=640)
#     list_plates = plates.pandas().xyxy[0].values.tolist()
#     print(f"Phát hiện được {len(list_plates)} biển số.")  # Debug: In số lượng biển số phát hiện được
#     for plate in list_plates:
#         x = int(plate[0])  # xmin
#         y = int(plate[1])  # ymin
#         w = int(plate[2] - plate[0])  # xmax - xmin
#         h = int(plate[3] - plate[1])  # ymax - ymin
#         crop_img = frame[y:y+h, x:x+w]

#         # Resize vùng phát hiện biển số để tăng độ phân giải
#         crop_img = cv2.resize(crop_img, (300, 100), interpolation=cv2.INTER_CUBIC)

#         # Áp dụng các kỹ thuật xử lý ảnh
#         crop_img = sharpen_image(crop_img)
#         crop_img = enhance_contrast(crop_img)
#         crop_img = denoise_image(crop_img)
#         crop_img = cv2.resize(crop_img, (600, 200), interpolation=cv2.INTER_CUBIC)

#         # Hiển thị vùng phát hiện biển số trong giao diện Tkinter
#         crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB
#         crop_img_pil = Image.fromarray(crop_img_rgb)  # Chuyển sang định dạng PIL
#         crop_img_tk = ImageTk.PhotoImage(image=crop_img_pil)  # Chuyển sang định dạng Tkinter
#         cropped_label.imgtk = crop_img_tk  # Lưu tham chiếu để tránh bị xóa
#         cropped_label.configure(image=crop_img_tk)

#         # Vẽ khung xung quanh biển số
#         cv2.rectangle(frame, (x, y), (x+w, y+h), color=(0, 0, 225), thickness=2)

#         # Đọc biển số
#         for cc in range(0, 2):
#             for ct in range(0, 2):
#                 lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
#                 print(f"Biển số đọc được: {lp}")  # Debug: In biển số đọc được
#                 if lp != "unknown":
#                     cv2.putText(frame, lp, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36, 255, 12), 3)  # Tăng fontScale và thickness
#                     update_textbox(lp, crop_img)  # Update the TextBox with the detected plate
#                     break

# # Create Tkinter GUI
# root = tk.Tk()
# root.title("License Plate Detection")

# # Main frame for buttons
# button_frame = tk.Frame(root)
# button_frame.pack(side=tk.LEFT, padx=10, pady=10)

# tk.Label(button_frame, text="Chọn chế độ phát hiện:", font=("Arial", 14)).pack(pady=10)
# tk.Button(button_frame, text="Chọn Ảnh", command=detect_image, width=20, height=2).pack(pady=5)
# tk.Button(button_frame, text="Chọn Video", command=detect_video, width=20, height=2).pack(pady=5)
# tk.Button(button_frame, text="Webcam", command=detect_webcam, width=20, height=2).pack(pady=5)
# tk.Button(button_frame, text="Export to Excel", command=export_to_excel, width=20, height=2, bg="#FFC107").pack(pady=5)

# # Frame for video display
# video_frame = tk.Frame(root, width=640, height=480, bg="black")
# video_frame.pack(side=tk.TOP, padx=10, pady=10)
# video_label = tk.Label(video_frame, width=640, height=480, bg="black")
# video_label.pack()

# # Frame for cropped plate display
# cropped_frame = tk.Frame(root, width=300, height=100, bg="white", relief=tk.SUNKEN, bd=2)
# cropped_frame.pack(side=tk.RIGHT, padx=10, pady=10)
# cropped_label = tk.Label(cropped_frame, width=300, height=100, bg="white")
# cropped_label.pack()

# # Frame for detected plates
# text_frame = tk.Frame(root)
# text_frame.pack(side=tk.BOTTOM, padx=10, pady=10, fill=tk.X)

# tk.Label(text_frame, text="Biển số đã phát hiện:", font=("Arial", 12)).pack(pady=5)
# textbox = tk.Text(text_frame, height=5, width=80)
# textbox.pack()

# root.mainloop()



from PIL import Image, ImageTk
import cv2
import torch
import os
import time
import tkinter as tk
from tkinter import filedialog
import pandas as pd  # Thêm pandas để xuất file Excel
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
detected_plates = {}  # Dictionary để lưu biển số và hình ảnh biển số

# Function to update the TextBox with detected plates
def update_textbox(plate, crop_img):
    if plate not in detected_plates:
        print(f"Thêm biển số vào TextBox: {plate}")  # Debug: In biển số được thêm
        detected_plates[plate] = crop_img  # Lưu hình ảnh biển số vào dictionary
        textbox.insert(tk.END, f"{plate}\n")
        textbox.see(tk.END)  # Scroll to the bottom

# Function to export detected plates to Excel
def export_to_excel():
    if not detected_plates:
        print("Không có biển số nào để xuất.")
        return

    # Tạo thư mục tạm để lưu hình ảnh biển số
    output_dir = "detected_plates"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Tạo danh sách dữ liệu cho Excel
    data = []
    for idx, (plate, crop_img) in enumerate(detected_plates.items()):
        # Lưu hình ảnh biển số vào file
        image_path = os.path.join(output_dir, f"plate_{idx+1}.png")
        cv2.imwrite(image_path, crop_img)

        # Thêm dữ liệu vào danh sách
        data.append([idx + 1, image_path, plate])

    # Tạo DataFrame và lưu vào file Excel
    df = pd.DataFrame(data, columns=["Số thứ tự", "Hình ảnh", "Văn bản"])
    file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
    if file_path:
        with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Detected Plates")

            # Thêm hình ảnh vào file Excel
            workbook = writer.book
            worksheet = writer.sheets["Detected Plates"]
            for idx, image_path in enumerate(df["Hình ảnh"]):
                worksheet.set_row(idx + 1, 80)  # Đặt chiều cao hàng
                worksheet.insert_image(idx + 1, 1, image_path, {"x_scale": 0.5, "y_scale": 0.5, "object_position": 1})

        print(f"Xuất file Excel thành công: {file_path}")

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
    time.sleep(0.2)  # Tạm dừng 200ms giữa các khung hình (giảm FPS)

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
                    update_textbox(lp, crop_img)  # Update the TextBox with the detected plate
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
tk.Button(button_frame, text="Export to Excel", command=export_to_excel, width=20, height=2, bg="#FFC107").pack(pady=5)

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