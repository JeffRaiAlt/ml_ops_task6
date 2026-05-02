import cv2
import numpy as np
import os
import urllib.request
from tqdm import tqdm
import gdown
from IPython.display import Video, Image

gdrive_url = 'https://drive.google.com/uc?export=download&id=1OjSrdAS4zVfa1u7vtyh48PKEoVW20vPN'
gdown.download(gdrive_url, quiet=True); Video("/content/HW5_Woman_Happy.mp4",embed=True, width=640, height=360)

# 1. Скачиваем файл каскада, если его нет локально
cascade_file = 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_file):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    print("Загрузка каскада Хаара...")
    urllib.request.urlretrieve(url, cascade_file)

def apply_mosaic_effect(face_roi, pixel_size=10):
    if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
        return face_roi
    h, w, _ = face_roi.shape
    small_h = max(1, h // pixel_size)
    small_w = max(1, w // pixel_size)
    downscaled = cv2.resize(face_roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mosaiced_face = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_NEAREST)
    return mosaiced_face

# Пути к файлам (убрали /content/)
video_file_path = 'HW5_Woman_Happy.mp4'
output_video_path = 'HW5_Woman_Happy_blurred.avi' # XVID лучше сохранять в .avi

video_capture = cv2.VideoCapture(video_file_path)
face_cascade = cv2.CascadeClassifier(cascade_file)

if not video_capture.isOpened():
    print(f"Ошибка: Не удалось открыть видео файл {video_file_path}")
else:
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Кодек XVID
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Обработка: FPS={fps}, Res={width}x{height}")

    with tqdm(total=frame_count, desc="Прогресс") as pbar:
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                mosaiced_face = apply_mosaic_effect(face_roi, pixel_size=30) # pixel_size 100 может быть слишком крупным
                frame[y:y+h, x:x+w] = mosaiced_face

            video_writer.write(frame)

            # Показываем окно (опционально)
            cv2.imshow('Processing...', cv2.resize(frame, (width//2, height//2)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            pbar.update(1)

    video_capture.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"\nГотово! Видео сохранено в: {output_video_path}")