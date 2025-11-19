import os
from PIL import Image
from orientation_detector import OrientationDetector


class ImageProcessor:
    def __init__(self):
        # Инициализируем детектор ориентации
        # Модель загружается один раз при создании объекта
        try:
            self.orientation_detector = OrientationDetector()
        except Exception as e:
            print(f"Warning: Could not initialize orientation detector: {str(e)}")
            self.orientation_detector = None

    def ensure_correct_orientation(self, image_path: str) -> bool:
        """
        Коррекция ориентации изображения с использованием deep-image-orientation-detection.
        Возвращает True если изображение было повернуто, False если ориентация уже правильная.
        """
        try:
            # Проверяем существование файла
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found at {image_path}")
                return False

            # Если детектор не инициализирован, используем fallback
            if self.orientation_detector is None:
                print(
                    "Warning: Orientation detector not available, using fallback method"
                )
                return self._fallback_orientation(image_path)

            # Получаем угол поворота напрямую от детектора
            rotation_angle = self.orientation_detector.predict_orientation(image_path)

            if rotation_angle != 0:
                print(f"Rotating image by {rotation_angle}°")
                return self._rotate_image(image_path, rotation_angle)
            else:
                print("Image orientation is correct, no rotation needed")
                return False

        except Exception as e:
            print(f"Unexpected error in orientation detection: {str(e)}")
            return self._fallback_orientation(image_path)

    def _rotate_image(self, image_path: str, angle: int) -> bool:
        """
        Поворачивает изображение на заданный угол и сохраняет его.
        """
        try:
            with Image.open(image_path) as img:
                # Поворачиваем изображение с expand=True чтобы избежать обрезки
                rotated_img = img.rotate(angle, expand=True)
                rotated_img.save(image_path)
                return True

        except Exception as e:
            print(f"Error rotating image: {str(e)}")
            return False

    def _fallback_orientation(self, image_path: str) -> bool:
        """
        Резервный метод определения ориентации по размерам изображения.
        Используется если нейронная сеть не сработала.
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                print(f"Fallback: Image dimensions: {width}x{height}")

                # если ширина > высоты, поворачиваем на 90° по часовой стрелке
                if width > height:
                    print("Image appears horizontal, rotating 90° clockwise...")
                    rotated_img = img.rotate(-90, expand=True)
                    rotated_img.save(image_path)
                    print(f"Image rotated and saved")
                    return True
                else:
                    print("Image orientation appears correct")
                    return False

        except Exception as e:
            print(f"Error in fallback orientation: {str(e)}")
            return False
