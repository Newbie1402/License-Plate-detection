from ultralytics import YOLO

def train_model():
    # Load mô hình YOLOv8
    model = YOLO("yolov8m.pt")

    # Train mô hình
    model.train(
        data="C:/Train/LETTER DETECTION.v2i.yolov8/data.yaml",
        epochs=200,
        imgsz=640,
        batch=16,
        workers=10,
        device="0",
        amp=True,
        optimizer="AdamW",
        cos_lr=True,
        cache="disk",
        dropout=0.0,
        close_mosaic=5,
        lr0=0.01,
        lrf=0.1
    )

if __name__ == "__main__":
    import torch
    print("CUDA Available:", torch.cuda.is_available())  # Debug GPU
    print("Using Device:", torch.cuda.get_device_name(0))
    
    train_model()
