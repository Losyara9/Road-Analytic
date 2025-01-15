import os
import yaml

# Объединенный список классов (все названия в нижнем регистре)
all_classes = ['bus', 'car', 'jeep', 'lorry', 'minibus', 'minivan', 'motorcycle', 'road train', 'truck']

# Пути к исходным датасетам
datasets = [
    {"path": "roboflow1", "yaml": "E:/los1/uechebnoe/kurs3/data_robo/roboflow1/data.yaml"},
    {"path": "roboflow2", "yaml": "E:/los1/uechebnoe/kurs3/data_robo/roboflow2/data.yaml"},
    {"path": "roboflow3", "yaml": "E:/los1/uechebnoe/kurs3/data_robo/roboflow3/data.yaml"},
    {"path": "dataset1", "yaml": "E:/los1/uechebnoe/kurs3/data_robo/dataset1/data.yaml"}
]

# Обработка каждого датасета
for dataset in datasets:
    dataset_path = dataset["path"]

    # Читаем оригинальный YAML
    with open(dataset["yaml"], "r") as f:
        original_data = yaml.safe_load(f)
        # Приводим все классы из YAML к нижнему регистру
        original_classes = [cls.lower() for cls in original_data["names"]]

    # Создаем маппинг старого индекса на новый
    class_mapping = {i: all_classes.index(cls) for i, cls in enumerate(original_classes)}

    # Обрабатываем файлы разметки
    label_dirs = [os.path.join(dataset_path, "train/labels"),
                  os.path.join(dataset_path, "valid/labels"),
                  os.path.join(dataset_path, "test/labels")]

    for label_dir in label_dirs:
        if not os.path.exists(label_dir):
            continue

        for file in os.listdir(label_dir):
            if file.endswith(".txt"):
                label_path = os.path.join(label_dir, file)

                # Читаем файл разметки и заменяем номера классов
                with open(label_path, "r") as f:
                    lines = f.readlines()
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    old_class_id = int(parts[0])
                    new_class_id = class_mapping[old_class_id]
                    parts[0] = str(new_class_id)
                    new_lines.append(" ".join(parts))

                # Сохраняем обновленный файл
                with open(label_path, "w") as f:
                    f.write("\n".join(new_lines))

print("Обновление разметки завершено.")
