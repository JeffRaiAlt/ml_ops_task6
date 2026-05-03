import base64
import os
import asyncio
from typing import List, Tuple

import cv2
import numpy as np
from pydantic import BaseModel
from faststream import FastStream
from faststream.rabbit import RabbitBroker, RabbitQueue

broker = RabbitBroker("amqp://guest:guest@localhost:5672/")
broker.config.prefetch_count = 7

publisher_broker = RabbitBroker("amqp://guest:guest@localhost:5672/")
app = FastStream(broker)

RESULT_DIR = "processed_frames_faststream"
os.makedirs(RESULT_DIR, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
face_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml"))

# Проверка сразу при загрузке
if face_cascade.empty():
    raise RuntimeError(f"Не удалось загрузить каскад из {BASE_DIR}")

# --- Схемы ---
class FrameReceivedEvent(BaseModel):
    frame_id: str
    frame_b64: str

class GrayscaleFrameCreatedEvent(BaseModel):
    frame_id: str
    original_frame_b64: str
    gray_frame_b64: str

class FacesDetectedEvent(BaseModel):
    frame_id: str
    original_frame_b64: str
    faces: List[Tuple[int, int, int, int]]

class FrameMosaicedEvent(BaseModel):
    frame_id: str
    mosaiced_frame_b64: str

# --- Сериализация ---
def encode(image: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")

def decode(b64: str, grayscale: bool = False) -> np.ndarray:
    data = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)

def apply_mosaic(roi: np.ndarray, pixel_size: int = 30) -> np.ndarray:
    h, w = roi.shape[:2]
    if h == 0 or w == 0:
        return roi
    small = cv2.resize(roi, (max(1, w // pixel_size), max(1, h // pixel_size)))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

# --- Очереди ---
queue_frame_received = RabbitQueue("frame.received", durable=True)
queue_grayscale_frame_created = RabbitQueue("grayscale.frame.created", durable=True)
queue_faces_detected = RabbitQueue("faces.detected", durable=True)
queue_frame_mosaiced = RabbitQueue("frame.mosaiced", durable=True)

# --- Lifecycle ---
@app.on_startup
async def startup():
    await publisher_broker.connect()
    print("Publisher broker connected")

@app.on_shutdown
async def shutdown():
    await publisher_broker.stop()
    print("Publisher broker stopped")

# --- Конвейер ---
"""@broker.subscriber(queue_frame_received)
async def convert_frame_to_gray(event: FrameReceivedEvent):
    loop = asyncio.get_event_loop()

    def process():
        frame = decode(event.frame_b64)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return event.frame_b64, encode(gray)

    original_b64, gray_b64 = await loop.run_in_executor(None, process)
    print("Done gray", event.frame_id)

    await publisher_broker.publish(
        GrayscaleFrameCreatedEvent(
            frame_id=event.frame_id,
            original_frame_b64=original_b64,
            gray_frame_b64=gray_b64,
        ),
        queue="grayscale.frame.created",
    )"""

@broker.subscriber(queue_frame_received)
async def convert_frame_to_gray(event: FrameReceivedEvent):
    frame = decode(event.frame_b64)  # np.ndarray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # np.ndarray

    await publisher_broker.publish(
        GrayscaleFrameCreatedEvent(
            frame_id=event.frame_id,
            original_frame_b64=event.frame_b64,  # оригинал уже есть в event!
            gray_frame_b64=encode(gray),  # np.ndarray → str
        ),
        queue="grayscale.frame.created",
    )
    print("Done gray", event.frame_id)



@broker.subscriber(queue_grayscale_frame_created)
async def detect_faces(event: GrayscaleFrameCreatedEvent):
    print("detect_faces!!!", event.frame_id)
    loop = asyncio.get_event_loop()

    def process():
        gray = decode(event.gray_frame_b64, grayscale=True)
        return [
            (int(x), int(y), int(w), int(h))
            for (x, y, w, h) in face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        ]

    faces = await loop.run_in_executor(None, process)
    print("Faces found", event.frame_id, len(faces))

    await publisher_broker.publish(
        FacesDetectedEvent(
            frame_id=event.frame_id,
            original_frame_b64=event.original_frame_b64,
            faces=faces,
        ),
        queue="faces.detected",
    )

@broker.subscriber(queue_faces_detected)
async def apply_mosaic_to_faces(event: FacesDetectedEvent):
    print("apply_mosaic!!!", event.frame_id)
    loop = asyncio.get_event_loop()

    def process():
        frame = decode(event.original_frame_b64)
        for x, y, w, h in event.faces:
            frame[y:y+h, x:x+w] = apply_mosaic(frame[y:y+h, x:x+w])
        return encode(frame)

    mosaiced_b64 = await loop.run_in_executor(None, process)

    await publisher_broker.publish(
        FrameMosaicedEvent(frame_id=event.frame_id, mosaiced_frame_b64=mosaiced_b64),
        queue="frame.mosaiced",
    )

@broker.subscriber(queue_frame_mosaiced)
async def save_mosaiced_frame(event: FrameMosaicedEvent):
    print("save!!!", event.frame_id)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: cv2.imwrite(
            os.path.join(RESULT_DIR, f"{event.frame_id}.jpg"),
            decode(event.mosaiced_frame_b64),
        ),
    )
    print("Saved", event.frame_id)