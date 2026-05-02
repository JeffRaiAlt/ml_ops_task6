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
                "frame.received",
            )
            pbar.update(1)

    cap.release()
    await broker.close()
    return fps, width, height, total


def wait_and_build(fps, width, height, total):
    # Ждём пока все кадры обработаются
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
            writer.write(frame)
            pbar.update(1)

    writer.release()
    print(f"Готово → {VIDEO_OUT}")


async def main():
    fps, width, height, total = await publish_frames()
    wait_and_build(fps, width, height, total)

if __name__ == "__main__":
    asyncio.run(main())