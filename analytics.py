import os
import cv2
import numpy as np
from ultralytics import YOLO
import json
from collections import defaultdict

# Определяем зоны
zones = [
    {"name": "Left Lane Left Side", "points": [(23, 493), (185, 423), (342, 359), (464, 298), (545, 262)]},
    {"name": "Left Lane Right Side", "points": [(495, 710), (539, 555), (574, 430), (611, 310), (625, 268)]},
    {"name": "Right Lane Left Side", "points": [(792, 714), (738, 556), (697, 431), (667, 314), (659, 266)]},
    {"name": "Right Lane Right Side", "points": [(1275, 502), (1152, 444), (1014, 385), (842, 311), (752, 262)]}
]

# Функция проверки, находится ли точка внутри полигона
def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

# Функция вычисления центра рамки
def get_bbox_center(xmin, ymin, xmax, ymax):
    return (xmin + xmax) / 2, (ymin + ymax) / 2

# Загрузка модели
model = YOLO("runs2/detect/train/weights/best.pt")

# Функция для получения списка путей ко всем изображениям в директории
def get_frame_paths(directory, extensions=(".jpg", ".png", ".jpeg")):
    return [os.path.join(directory, file) for file in sorted(os.listdir(directory)) if file.lower().endswith(extensions)]

# Функция для расчета расстояния между точками
def calculate_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Выполнение анализа на множестве кадров
def analyze_frames(frame_paths, zones, fps):
    results_summary = defaultdict(lambda: {"density": 0, "intensity": 0, "speed": []})  # Плотность, интенсивность, скорость
    object_tracks = defaultdict(list)  # Отслеживание объектов (id: [координаты])

    for frame_index, frame_path in enumerate(frame_paths):
        # Загрузка кадра
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Не удалось загрузить кадр: {frame_path}")
            continue

        # Выполняем детекцию
        results = model.predict(source=frame, conf=0.5, verbose=False)
        detections = results[0].boxes.xyxy.cpu().numpy()  # Координаты рамок: (xmin, ymin, xmax, ymax)
        class_ids = results[0].boxes.cls.cpu().numpy()  # Классы объектов
        confidences = results[0].boxes.conf.cpu().numpy()  # Уверенности детекции

        # Анализ детектированных объектов
        for detection, class_id, confidence in zip(detections, class_ids, confidences):
            xmin, ymin, xmax, ymax = detection
            center_x, center_y = get_bbox_center(xmin, ymin, xmax, ymax)

            # Отображение рамок и меток
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(frame, f"{int(class_id)} {confidence:.2f}", (int(xmin), int(ymin) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Проверяем, в какой зоне находится объект
            for zone in zones:
                if is_point_in_polygon((center_x, center_y), zone["points"]):
                    results_summary[zone["name"]]["density"] += 1
                    object_tracks[class_id].append((center_x, center_y, frame_index))  # Отслеживание по кадрам

        # Визуализация зон
        for zone in zones:
            points = np.array(zone["points"], np.int32)
            cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, zone["name"], tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Сохранение визуализированного кадра
        output_path = frame_path.replace(
            input_directory,
            "C:/Users/bekh-/PycharmProjects/AnalyticProject/processed_frames"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, frame)

    # Расчет скорости и интенсивности
    for object_id, track in object_tracks.items():
        for i in range(1, len(track)):
            prev_x, prev_y, prev_frame = track[i - 1]
            curr_x, curr_y, curr_frame = track[i]
            distance = calculate_distance((prev_x, prev_y), (curr_x, curr_y))
            time_elapsed = (curr_frame - prev_frame) / fps
            speed = distance / time_elapsed if time_elapsed > 0 else 0

            # Запись скорости в зону
            for zone in zones:
                if is_point_in_polygon((curr_x, curr_y), zone["points"]):
                    results_summary[zone["name"]]["speed"].append(speed)

        # Запись интенсивности (пересечения зон)
        for zone in zones:
            zone_crossings = sum(is_point_in_polygon((x, y), zone["points"]) for x, y, _ in track)
            results_summary[zone["name"]]["intensity"] += zone_crossings

    # Средняя скорость
    for zone_name, data in results_summary.items():
        if data["speed"]:
            data["average_speed"] = sum(data["speed"]) / len(data["speed"])
        else:
            data["average_speed"] = 0

    return results_summary

# Основной вызов функций
input_directory = "C:/Users/bekh-/PycharmProjects/AnalyticProject/check_frames"
frame_paths = get_frame_paths(input_directory)
fps = 30  # Реальная частота кадров видео

results = analyze_frames(frame_paths, zones, fps)

# Проверка и сохранение результатов аналитики
if results:
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("Результаты аналитики сохранены в 'results.json'.")
else:
    print("Анализ завершен, но результаты пусты. Проверьте обработку данных.")

# Печать итогов аналитики
for zone, data in results.items():
    print(f"Зона: {zone}")
    print(f"  Плотность: {data['density']}")
    print(f"  Интенсивность: {data['intensity']}")
    print(f"  Средняя скорость: {data['average_speed']:.2f} пикселей/сек")

