import os
import shutil

# Пути к исходным датасетам
datasets = [
    {"path": "E:/los1/uechebnoe/kurs3/data_robo/roboflow1"},
    {"path": "E:/los1/uechebnoe/kurs3/data_robo/roboflow2"},
    {"path": "E:/los1/uechebnoe/kurs3/data_robo/roboflow3"},
    {"path": "E:/los1/uechebnoe/kurs3/data_robo/dataset1"}
]

# Путь к объединенному датасету
merged_path = "E:/los1/uechebnoe/kurs3/merged_dataset"

# Поддиректории объединенного датасета
subdirs = ["train/images", "train/labels", "valid/images", "valid/labels", "test/images", "test/labels"]

# Создаем структуру директорий
for subdir in subdirs:
    os.makedirs(os.path.join(merged_path, subdir), exist_ok=True)

# Функция для копирования файлов
def copy_files(src_dir, dest_dir, prefix):
    if not os.path.exists(src_dir):
        return
    for file in os.listdir(src_dir):
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".txt"):
            # Добавляем уникальный префикс к имени файла
            new_name = f"{prefix}_{file}"
            shutil.copy(os.path.join(src_dir, file), os.path.join(dest_dir, new_name))

# Обрабатываем каждый датасет
for i, dataset in enumerate(datasets):
    dataset_path = dataset["path"]
    prefix = f"ds{i+1}"  # Уникальный префикс для файлов каждого датасета

    # Копируем файлы для train
    copy_files(os.path.join(dataset_path, "train/images"), os.path.join(merged_path, "train/images"), prefix)
    copy_files(os.path.join(dataset_path, "train/labels"), os.path.join(merged_path, "train/labels"), prefix)

    # Копируем файлы для valid
    copy_files(os.path.join(dataset_path, "valid/images"), os.path.join(merged_path, "valid/images"), prefix)
    copy_files(os.path.join(dataset_path, "valid/labels"), os.path.join(merged_path, "valid/labels"), prefix)

    # Копируем файлы для test
    copy_files(os.path.join(dataset_path, "test/images"), os.path.join(merged_path, "test/images"), prefix)
    copy_files(os.path.join(dataset_path, "test/labels"), os.path.join(merged_path, "test/labels"), prefix)

print("Датасеты успешно объединены.")
