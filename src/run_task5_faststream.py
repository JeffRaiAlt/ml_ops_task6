import os
import time
import shutil
import asyncio

import cv2
from tqdm import tqdm
from task5_faststream import broker, RESULT_DIR, FrameReceivedEvent, encode

VIDEO_IN  = "HW5_Woman_Happy.mp4"
VIDEO_OUT = "HW5_Woman_Happy_blurred_faststream.avi"


async def publish_frames():
    if os.path.exists(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)
    os.makedirs(RESULT_DIR)

    cap = cv2.VideoCapture(VIDEO_IN)
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Обработка: FPS={fps}, Res={width}x{height}, Кадров={total}")

    await broker.connect()

    with tqdm(total=total, desc="Отправка") as pbar:
        for idx in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            await broker.publish(
                FrameReceivedEvent(frame_id=f"frame_{idx:06d}", frame_b64=encode(frame)),
                queue="frame.received",
            )
            pbar.update(1)

    cap.release()
    # НЕ останавливаем broker здесь — пусть воркер дообрабатывает
    return fps, width, height, total


def wait_and_build(fps, width, height, total):
    tqdm.write(f"Ожидаем {total} кадров в {RESULT_DIR}...")
    with tqdm(total=total, desc="Ожидание") as pbar:
        done = 0
        no_progress_count = 0
        while done < total:
            count = len([f for f in os.listdir(RESULT_DIR) if f.endswith(".jpg")])
            if count > done:
                pbar.update(count - done)
                done = count
                no_progress_count = 0
            else:
                no_progress_count += 1
                if no_progress_count > 120:
                    tqdm.write(f"\nВремя ожидания истекло! Получено {done}/{total} кадров")
                    break
            time.sleep(0.5)

    tqdm.write(f"Получено кадров: {done}/{total}")

    writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))
    with tqdm(total=done, desc="Запись") as pbar:
        for i in range(total):
            path = os.path.join(RESULT_DIR, f"frame_{i:06d}.jpg")
            if not os.path.exists(path):
                tqdm.write(f"Пропущен кадр {i}")
                continue
            frame = cv2.imread(path)
            if frame is not None:
                writer.write(frame)
            pbar.update(1)

    writer.release()
    tqdm.write(f"Готово → {VIDEO_OUT}")

async def main():
    start = time.perf_counter()

    fps, width, height, total = await publish_frames()
    wait_and_build(fps, width, height, total)

    elapsed = time.perf_counter() - start
    print(f"Общее время: {elapsed:.1f} сек ({elapsed / 60:.1f} мин)")


if __name__ == "__main__":
    asyncio.run(main())