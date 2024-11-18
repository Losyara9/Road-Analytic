import cv2
import os

def extract_frames(video_path, output_dir, frame_rate=1):

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Частота кадров видео
    frame_interval = int(fps / frame_rate)  # Интервал между кадрами

    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_name = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Извлечено и сохранено {saved_count} кадров в '{output_dir}'.")


extract_frames("E:\\los1\\uechebnoe\\kurs3\\detection\\video\\road.mp4",
               "E:\\los1\\uechebnoe\\kurs3\\detection\\frames",
               frame_rate=10)
