# 🇻🇳 Vietnamese License Plate Recognition with YOLOv8

This project leverages **YOLOv8** and **Python** to detect and recognize Vietnamese license plates from images, videos, or webcam. It includes a full GUI built with Tkinter, and supports exporting results to Excel with annotated images.

---

## 🧰 Features

- License plate detection using YOLOv8
- Character recognition using custom-trained YOLOv8m models
- GUI built with Tkinter
- Supports image, video, and webcam input
- Export results (license plate, timestamp, image) to Excel
- Preprocessing (deskew, contrast enhancement) for better OCR accuracy

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Newbie1402/License-Plate-detection.git
cd vietnamese-license-plate
code .
# install dependencies using pip 
 pip install -r ./requirement.txt
 ``` 
## 🚀 Running the Application
Navigate to the built executable
```bash
cd LicensePlate\dist
main.exe
``` 
Run the app with GUI:
```bash
- Run python main.py
``` 

## 🧪 Model Training
The project uses a 2-stage pipeline:

Stage 1 – License Plate Detection:
Model detects license plates in the image.

Stage 2 – Character Recognition:
Model detects individual characters inside the plate.

## 🏋️ Training YOLOv8 Model
```bash
- cd training
  python train.py
``` 
&rarr; Make sure to configure train.py with your dataset path and model parameters. 

## 🔗 Datasets Used
- **Character Detection:** 
  - [**Vietnamese License Plate Dataset**](https://universe.roboflow.com/cao-phong-3qbun/letter-detection-0f1lb)

- **License Plate Detection:** 
  - [**Vietnamese License Plate Dataset**](https://universe.roboflow.com/test-n0rhd/vietnamese-license-plate-tptd0-npjfu)


# 📸 Sample Result
![img.png](img.png)

## 📁 Project Structure
```
├── main.py                  # Entry point for GUI
├── dist/                    # Application.exe
├── function/                # Image processing
├── training/                # Training scripts
├── model/                   # Trained models
├── requirements.txt         # Dependencies
└── README.md
```
## Update main.spec

```bash
python -m PyInstaller main.spec 
```


