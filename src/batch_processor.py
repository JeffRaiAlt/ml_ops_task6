import time
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


class BatchProcessor:
    """
    Кадры копятся в буфере и обрабатываются пачкой —
    когда буфер заполнен или истёк таймаут ожидания.
    """

    def __init__(self, batch_size=64, max_wait_seconds=30):
        self.batch_size = batch_size
        self.max_wait_seconds = max_wait_seconds
        self.buffer = []
        self.last_flush = time.time()

    def add(self, frame):
        self.buffer.append(frame)

        buffer_full = len(self.buffer) >= self.batch_size
        timeout = time.time() - self.last_flush >= self.max_wait_seconds

        if buffer_full or timeout:
            return self.flush()

    def flush(self):
        batch, self.buffer = self.buffer, []
        self.last_flush = time.time()

        # Haar Cascade не поддерживает batch-inference, поэтому
        # detectMultiScale принимает только один кадр :(
        results = []
        for frame in batch:
            results.append(
                face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)
            )
        return results


# Пример использования
processor = BatchProcessor(batch_size=64, max_wait_seconds=30)

for _ in range(100):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processor.add(gray)

# Остаток буфера
processor.flush()