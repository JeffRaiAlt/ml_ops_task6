import os
import time
import urllib.request
from abc import ABC, abstractmethod

import cv2
import numpy as np


MODEL_NAME = "haarcascade_frontalface_default.xml"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"


def download_model_if_needed():
    """
    Проверка наличие модели локально.
    Если файла нет — скачивает его.
    """
    if not os.path.exists(MODEL_NAME):
        print("[Init] Скачиваем Haar Cascade...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_NAME)


class LoadingStrategy(ABC):
    """
    Интерфейс стратегии загрузки модели.
    Позволяет менять способ загрузки (Lazy/Warm),
    не меняя код детекции.
    """

    @abstractmethod
    def get_model(self):
        pass


class LazyLoadingStrategy(LoadingStrategy):
    """
    Модель загружается только при первом вызове.
    """

    def __init__(self):
        self.model = None

    def get_model(self):
        if self.model is None:
            print("[Lazy] Загрузка модели при первом вызове")
            download_model_if_needed()

            self.model = cv2.CascadeClassifier(MODEL_NAME)

            if self.model.empty():
                raise RuntimeError("Ошибка загрузки модели")

        return self.model


class WarmLoadingStrategy(LoadingStrategy):
    """
    Модель загружается сразу при создании стратегии.
    """

    def __init__(self):
        print("[Warm] Предварительная загрузка модели")
        download_model_if_needed()

        self.model = cv2.CascadeClassifier(MODEL_NAME)

        if self.model.empty():
            raise RuntimeError("Ошибка загрузки модели")

    def get_model(self):
        return self.model


class FaceDetector:
    """
    Класс использует стратегию загрузки
    """

    def __init__(self, strategy: LoadingStrategy):
        self.strategy = strategy

    def detect(self, gray_frame):
        """
        Выполняет детекцию лиц на grayscale-кадре.
        """
        model = self.strategy.get_model()

        return model.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=4,
        )



def run_test(name, detector, gray):
    """
    Запускает детекцию и измеряет время.
    Используем для сравнения стратегий
    """
    start = time.time()
    faces = detector.detect(gray)
    t = time.time() - start

    print(f"[{name}] faces={len(faces)} time={t:.4f}s")


# Тестовый кадр (чёрное изображение)
frame = np.zeros((480, 640, 3), dtype=np.uint8)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


print("--- Lazy loading ---")
lazy_detector = FaceDetector(LazyLoadingStrategy())
run_test("Lazy (1-й вызов)", lazy_detector, gray)
run_test("Lazy (2-й вызов)", lazy_detector, gray)


print("--- Warm loading ---")
warm_detector = FaceDetector(WarmLoadingStrategy())
run_test("Warm (1-й вызов)", warm_detector, gray)
run_test("Warm (2-й вызов)", warm_detector, gray)