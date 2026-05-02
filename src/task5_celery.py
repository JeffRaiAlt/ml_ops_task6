import base64
import os
import urllib.request

import cv2
import numpy as np
from celery import Celery
from kombu import Queue



# Каждая стадия явно читает из своей очереди и публикует
# в следующую — без celery.chain, чтобы маршрутизация
# соответствовала AsyncAPI-спецификации:
#   frame.received -> grayscale.frame.created -> faces.detected -> frame.mosaiced


celery_app = Celery(
    "ml_video_pipeline",
    broker="amqp://guest:guest@localhost:5672//",
    backend="rpc://",
)

celery_app.conf.task_queues = (
    Queue("frame.received",           routing_key="frame.received"),
    Queue("grayscale.frame.created",  routing_key="grayscale.frame.created"),
    Queue("faces.detected",           routing_key="faces.detected"),
    Queue("frame.mosaiced",           routing_key="frame.mosaiced"),
)

celery_app.conf.task_routes = {
    "tasks.convert_to_gray":   {"queue": "frame.received",          "routing_key": "frame.received"},
    "tasks.detect_faces":      {"queue": "grayscale.frame.created", "routing_key": "grayscale.frame.created"},
    "tasks.apply_mosaic":      {"queue": "faces.detected",          "routing_key": "faces.detected"},
    "tasks.save_final_result": {"queue": "frame.mosaiced",          "routing_key": "frame.mosaiced"},
}



# Haar Cascade — Warm Loading
CASCADE_FILE = "haarcascade_frontalface_default.xml"
CASCADE_URL  = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "data/haarcascades/haarcascade_frontalface_default.xml"
)

if not os.path.exists(CASCADE_FILE):
    print("Загрузка каскада Хаара...")
    urllib.request.urlretrieve(CASCADE_URL, CASCADE_FILE)

face_cascade = cv2.CascadeClassifier(CASCADE_FILE)

if face_cascade.empty():
    raise RuntimeError("Не удалось загрузить Haar Cascade")



# Временное хранилище результатов

# Celery-задача не пишет напрямую в VideoWriter, потому что:
# - кадры приходят асинхронно;
# - порядок выполнения задач не гарантирован.
# Финальная задача сохраняет кадры на диск,
# затем собирем их по frame_id в правильном порядке.

RESULT_DIR = "processed_frames"
os.makedirs(RESULT_DIR, exist_ok=True)



# Сериализация

def encode_frame(frame: np.ndarray) -> str:
    """OpenCV image -> JPEG -> base64 string."""
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise ValueError("Не удалось закодировать кадр")
    return base64.b64encode(buffer).decode("utf-8")


def decode_frame(frame_b64: str, grayscale: bool = False) -> np.ndarray:
    """base64 string -> JPEG -> OpenCV image."""
    data = np.frombuffer(base64.b64decode(frame_b64), dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    frame = cv2.imdecode(data, flag)
    if frame is None:
        raise ValueError("Не удалось декодировать кадр")
    return frame


def apply_mosaic_effect(roi: np.ndarray, pixel_size: int = 30) -> np.ndarray:
    """Уменьшаем ROI и увеличиваем через INTER_NEAREST — получаем крупные пиксели."""
    h, w = roi.shape[:2]
    if h == 0 or w == 0:
        return roi
    small = cv2.resize(roi, (max(1, w // pixel_size), max(1, h // pixel_size)))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


# Stage 1: frame.received -> grayscale.frame.created
@celery_app.task(name="tasks.convert_to_gray")
def convert_to_gray(payload: dict) -> None:
    frame = decode_frame(payload["frame_b64"])
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    next_payload = {
        "frame_id":           payload["frame_id"],
        "original_frame_b64": payload["frame_b64"],
        "gray_frame_b64":     encode_frame(gray),
    }

    detect_faces.apply_async(args=[next_payload], queue="grayscale.frame.created",
                             routing_key="grayscale.frame.created")


# Stage 2: grayscale.frame.created -> faces.detected
@celery_app.task(name="tasks.detect_faces")
def detect_faces(payload: dict) -> None:
    gray  = decode_frame(payload["gray_frame_b64"], grayscale=True)
    faces = [
        (int(x), int(y), int(w), int(h))
        for x, y, w, h in face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    ]

    next_payload = {
        "frame_id":           payload["frame_id"],
        "original_frame_b64": payload["original_frame_b64"],
        "faces":              faces,
    }

    apply_mosaic.apply_async(args=[next_payload], queue="faces.detected",
                             routing_key="faces.detected")


# Stage 3: faces.detected -> frame.mosaiced
@celery_app.task(name="tasks.apply_mosaic")
def apply_mosaic(payload: dict) -> None:
    frame = decode_frame(payload["original_frame_b64"])

    for x, y, w, h in payload["faces"]:
        frame[y:y+h, x:x+w] = apply_mosaic_effect(frame[y:y+h, x:x+w])

    next_payload = {
        "frame_id":           payload["frame_id"],
        "mosaiced_frame_b64": encode_frame(frame),
    }

    save_final_result.apply_async(args=[next_payload], queue="frame.mosaiced",
                                  routing_key="frame.mosaiced")


# Stage 4: frame.mosaiced -> сохранение на диск
@celery_app.task(name="tasks.save_final_result")
def save_final_result(payload: dict) -> None:
    frame = decode_frame(payload["mosaiced_frame_b64"])
    cv2.imwrite(os.path.join(RESULT_DIR, f"{payload['frame_id']}.jpg"), frame)