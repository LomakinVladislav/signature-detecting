import os
import subprocess
import sys
from pathlib import Path
from PIL import Image
import cv2
import re

class ImageProcessor:
    def __init__(self):
        self.predict_script_path = os.path.join("deep-image-orientation-detection", "predict.py")
        self.orientation_detection_dir = "deep-image-orientation-detection"
    
    def ensure_correct_orientation(self, image_path: str) -> bool:
        """
        Коррекция ориентации изображения с использованием deep-image-orientation-detection.
        Возвращает True если изображение было повернуто, False если ориентация уже правильная.
        """
        try:
            # Проверяем существование скрипта и файла
            if not os.path.exists(self.predict_script_path):
                print(f"Warning: Orientation detection script not found at {self.predict_script_path}")
                return False
            
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found at {image_path}")
                return False
            
            # Получаем абсолютные пути
            abs_script_path = os.path.abspath(self.predict_script_path)
            abs_image_path = os.path.abspath(image_path)
            detection_dir = os.path.abspath(self.orientation_detection_dir)
            
            # print(f"Running orientation detection on: {abs_image_path}")
            # print(f"Working directory: {detection_dir}")
            
            # Запускаем скрипт из директории deep-image-orientation-detection
            result = subprocess.run([
                sys.executable,
                "predict.py",  # используем относительный путь
                "--input_path", abs_image_path
            ], capture_output=True, text=True, cwd=detection_dir, timeout=30)
            
            # # Логируем вывод для отладки
            # if result.stdout:
            #     print("="*60)
            #     print(result.stdout)
            #     print("="*60)   
            
            # Анализируем вывод для определения угла поворота
            rotation_angle = self._parse_rotation_angle(result.stdout)
            
            if rotation_angle is not None and rotation_angle != 0:
                print(f"Rotating image by {rotation_angle}°")
                return self._rotate_image(image_path, rotation_angle)
            else:
                print("Image orientation is correct, no rotation needed")
                return False
                
        except subprocess.TimeoutExpired:
            print("Orientation detection timed out")
            return self._fallback_orientation(image_path)
        except Exception as e:
            print(f"Unexpected error in orientation detection: {str(e)}")
            return self._fallback_orientation(image_path)
    
    def _parse_rotation_angle(self, output: str) -> int:
        """
        Парсит вывод нейронной сети для определения угла поворота.
        Согласно документации:
        Class 0: 0° (правильная ориентация)
        Class 1: 90° по часовой стрелке
        Class 2: 180°
        Class 3: 90° против часовой стрелки
        """
        try:
            # Парсим по классам
            if "Prediction: Image is correctly oriented (0°)." in output:
                return 0
            elif "Prediction: Image needs to be rotated 90° Clockwise to be correct." in output:
                return -90 
            elif "Prediction: Image needs to be rotated 180° to be correct." in output:
                return 180
            elif "Prediction: Image needs to be rotated 90° Counter-Clockwise to be correct." in output:
                return 90
            
            return 0  # По умолчанию считаем что ориентация правильная
                
        except Exception as e:
            print(f"Error parsing rotation angle: {str(e)}")
            return None
    
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