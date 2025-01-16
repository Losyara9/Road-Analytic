import os
import cv2
import numpy as np
from ultralytics import YOLO
import json
from collections import defaultdict
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum as _sum
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, ArrayType

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

# Функция для расчета расстояния между точками
def calculate_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Функция для обработки одного кадра
def process_frame(frame_path, zones, model):
    frame = cv2.imread(frame_path)
    if frame is None:
        return []

    results = model.predict(source=frame, conf=0.5, verbose=False)
    detections = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()

    frame_results = []
    for detection, class_id, confidence in zip(detections, class_ids, confidences):
        xmin, ymin, xmax, ymax = detection
        center_x, center_y = get_bbox_center(xmin, ymin, xmax, ymax)

        for zone in zones:
            if is_point_in_polygon((center_x, center_y), zone["points"]):
                frame_results.append((zone["name"], class_id, center_x, center_y))

    return frame_results

# Основной вызов функций
input_directory = "C:/Users/User/PycharmProjects/Road-Analytic-main/check_frames"
fps = 30  # Реальная частота кадров видео

# Создаем SparkSession
spark = SparkSession.builder.appName("TrafficFlowAnalysis").getOrCreate()

# Получаем список путей к кадрам
frame_paths = [os.path.join(input_directory, file) for file in sorted(os.listdir(input_directory)) if file.lower().endswith((".jpg", ".png", ".jpeg"))]

# Загрузка модели
model = YOLO("runs2/detect/train/weights/best.pt")

# Распределяем обработку кадров по узлам кластера
frame_rdd = spark.sparkContext.parallelize(frame_paths)
results_rdd = frame_rdd.flatMap(lambda path: process_frame(path, zones, model))

# Схема для DataFrame
schema = StructType([
    StructField("zone_name", StringType(), True),
    StructField("class_id", IntegerType(), True),
    StructField("center_x", FloatType(), True),
    StructField("center_y", FloatType(), True)
])

# Преобразуем RDD в DataFrame
results_df = spark.createDataFrame(results_rdd, schema)

# Агрегируем результаты по зонам
density_df = results_df.groupBy("zone_name").agg(_sum(lit(1)).alias("density"))
intensity_df = results_df.groupBy("zone_name").agg(_sum(lit(1)).alias("intensity"))

# Расчет скорости (пример, требует доработки)
window_spec = Window.partitionBy("class_id").orderBy("frame_index")
speed_df = results_df.withColumn("prev_x", F.lag("center_x").over(window_spec)) \
                     .withColumn("prev_y", F.lag("center_y").over(window_spec)) \
                     .withColumn("prev_frame", F.lag("frame_index").over(window_spec)) \
                     .withColumn("distance", F.sqrt((col("center_x") - col("prev_x"))**2 + (col("center_y") - col("prev_y"))**2)) \
                     .withColumn("time_diff", (col("frame_index") - col("prev_frame")) / fps) \
                     .withColumn("speed", col("distance") / col("time_diff")) \
                     .groupBy("zone_name").agg(avg("speed").alias("average_speed"))

# Объединяем все метрики в один DataFrame
final_df = density_df.join(intensity_df, "zone_name").join(speed_df, "zone_name")

# Сохраняем результаты в JSON
final_df.write.json("results.json")

# Останавливаем SparkSession
spark.stop()
