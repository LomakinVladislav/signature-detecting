from PIL import Image
import os

class ImageProcessor:
    def __init__(self):
        pass
    
    def ensure_vertical_orientation(self, image_path: str) -> bool:
        """
        Проверяет ориентацию изображения и поворачивает на 90° по часовой стрелке если нужно.
        Работает с исходным файлом (перезаписывает его).
        Возвращает True если изображение было повернуто, False если не требовалось
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
                if width > height:
                    rotated_img = img.rotate(-90, expand=True)
                    
                    rotated_img.save(image_path)
                    return True
                else:
                    return False
                    
        except Exception as e:
            print(f"Error processing image orientation: {str(e)}")
            return False