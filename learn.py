from ultralytics import YOLO

# Загрузка модели YOLOv8
model = YOLO('yolov8n.pt')  # Базовая версия модели
model.train(data='datasets/data.yaml', epochs=50, imgsz=640)
