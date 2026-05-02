import base64
import os
from typing import List, Tuple

import cv2
import numpy as np
from pydantic import BaseModel
from faststream import FastStream
from faststream.rabbit import RabbitBroker

broker = RabbitBroker("amqp://guest:guest@localhost:5672/")
app = FastStream(broker)

RESULT_DIR = "processed_frames_faststream"
os.makedirs(RESULT_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# --- Схемы сообщений ---

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


# --- Конвейер ---

@broker.subscriber("frame.received")
@broker.publisher("grayscale.frame.created")
async def convert_frame_to_gray(event: FrameReceivedEvent) -> GrayscaleFrameCreatedEvent:
    frame = decode(event.frame_b64)
    return GrayscaleFrameCreatedEvent(
        frame_id=event.frame_id,
        original_frame_b64=event.frame_b64,
        gray_frame_b64=encode(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)),
    )

@broker.subscriber("grayscale.frame.created")
@broker.publisher("faces.detected")
async def detect_faces(event: GrayscaleFrameCreatedEvent) -> FacesDetectedEvent:
    gray = decode(event.gray_frame_b64, grayscale=True)
    faces = [
        (int(x), int(y), int(w), int(h))
        for (x, y, w, h) in face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    ]
    return FacesDetectedEvent(
        frame_id=event.frame_id,
        original_frame_b64=event.original_frame_b64,
        faces=faces,
    )

@broker.subscriber("faces.detected")
@broker.publisher("frame.mosaiced")
async def apply_mosaic_to_faces(event: FacesDetectedEvent) -> FrameMosaicedEvent:
    frame = decode(event.original_frame_b64)
    for x, y, w, h in event.faces:
        frame[y:y+h, x:x+w] = apply_mosaic(frame[y:y+h, x:x+w])
    return FrameMosaicedEvent(frame_id=event.frame_id, mosaiced_frame_b64=encode(frame))

@broker.subscriber("frame.mosaiced")
async def save_mosaiced_frame(event: FrameMosaicedEvent):
    cv2.imwrite(os.path.join(RESULT_DIR, f"{event.frame_id}.jpg"), decode(event.mosaiced_frame_b64))