import os
import time
import shutil

import cv2
from tqdm import tqdm
from task5_celery import RESULT_DIR, convert_to_gray, encode_frame

VIDEO_IN  = "HW5_Woman_Happy.mp4"
VIDEO_OUT = "HW5_Woman_Happy_blurred.avi"

if os.path.exists(RESULT_DIR):
    shutil.rmtree(RESULT_DIR)
os.makedirs(RESULT_DIR)

cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise RuntimeError(f"Не удалось открыть {VIDEO_IN}")

fps    = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Обработка: FPS={fps}, Res={width}x{height}, Кадров={total}")
start = time.perf_counter()
# Отправляем кадры в очередь
with tqdm(total=total, desc="Отправка") as pbar:
    for idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        convert_to_gray.apply_async(
            args=[{"frame_id": f"frame_{idx:06d}", "frame_b64": encode_frame(frame)}],
            queue="frame.received",
        )
        pbar.update(1)

cap.release()

# Ждём результатов
with tqdm(total=total, desc="Ожидание") as pbar:
    done = 0
    while done < total:
        count = len([f for f in os.listdir(RESULT_DIR) if f.endswith(".jpg")])
        if count > done:
            pbar.update(count - done)
            done = count
        time.sleep(0.5)

# Собираем видео
writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

with tqdm(total=total, desc="Запись") as pbar:
    for i in range(total):
        frame = cv2.imread(os.path.join(RESULT_DIR, f"frame_{i:06d}.jpg"))
        if frame is None:
            raise RuntimeError(f"Не найден кадр frame_{i:06d}.jpg")
        writer.write(frame)
        pbar.update(1)

writer.release()
print(f"Готово → {VIDEO_OUT}")
elapsed = time.perf_counter() - start
print(f"Общее время: {elapsed:.1f} сек ({elapsed / 60:.1f} мин)")