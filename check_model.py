import os
import cv2
from ultralytics import YOLO

model_path = "train4/weights/best.pt"

input_frames_dir = "check_frames"

output_frames_dir = "check_frames_output"

class_names = ["Car", "Truck", "Motorcycle", "Minivan", "Minibus", "Bus", "Lorry", "Road Train"]

os.makedirs(output_frames_dir, exist_ok=True)

model = YOLO(model_path)

frame_files = [f for f in sorted(os.listdir(input_frames_dir)) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

for frame_file in frame_files:
    frame_path = os.path.join(input_frames_dir, frame_file)

    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Не удалось загрузить файл: {frame_file}")
        continue

    # Выполняем предсказание
    results = model.predict(source=frame, conf=0.5, verbose=False)
    detections = results[0].boxes.xyxy.cpu().numpy()  # Координаты рамок
    class_ids = results[0].boxes.cls.cpu().numpy()  # Классы объектов
    confidences = results[0].boxes.conf.cpu().numpy()  # Уверенности

    # Рисуем рамки и подписи на изображении
    for detection, class_id, confidence in zip(detections, class_ids, confidences):
        xmin, ymin, xmax, ymax = map(int, detection)

        # Получаем название класса
        class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else f"Class {int(class_id)}"

        # Подпись с названием класса и уверенностью
        label = f"{class_name}: {confidence:.2f}"

        # Рисуем рамку
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Подпись класса
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Сохраняем размеченное изображение
    output_path = os.path.join(output_frames_dir, frame_file)
    cv2.imwrite(output_path, frame)
    print(f"Сохранен размеченный кадр: {output_path}")

print("Обработка завершена. Проверьте папку:", output_frames_dir)
