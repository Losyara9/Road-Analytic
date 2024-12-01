from ultralytics import YOLO

# Загрузка обученной модели
model = YOLO('runs/detect/train4/weights/best.pt')  # Используйте путь к лучшей сохранённой модели

# Прогон модели на тестовых изображениях
results = model('C:/Users/bekh-/PycharmProjects/AnalyticProject/data/images/train/frames2', save=True, save_txt=True)

# Результаты сохранятся в 'runs/detect/predict'
