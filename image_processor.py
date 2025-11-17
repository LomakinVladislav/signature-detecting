import cv2
from PIL import Image
import os
from pytesseract import image_to_osd, Output

class ImageProcessor:
    def __init__(self):
        pass
    
    def ensure_correct_orientation(self, image_path: str) -> bool:
        """
        Умная коррекция ориентации документа с помощью Tesseract OSD.
        Возвращает True если изображение было повернуто, False если не требовалось
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Could not load image from {image_path}")
                return False
            
            osd = image_to_osd(img, output_type=Output.DICT, config='--psm 0')
            
            
            required_rotation = osd.get("rotate")
            if required_rotation is None:
                print("Tesseract failed to detect rotation angle")
                return self._fallback_orientation(image_path)
            
            confidence = osd.get("orient_conf") or osd.get("orientation_conf") or 0
            
            print(f"Tesseract OSD: required rotation = {required_rotation}°, confidence = {confidence:.2f}%")
            
            if confidence > 0 and confidence < 10:
                return self._fallback_orientation(image_path)
            
            if required_rotation == 0:
                print("Document orientation is correct")
                return False
            
            with Image.open(image_path) as pil_img:
                rotated_img = pil_img.rotate(required_rotation, expand=True)
                rotated_img.save(image_path)
                return True
                
        except Exception as e:
            print(f"Error in orientation correction: {str(e)}")
            print("Using fallback orientation method...")
            return self._fallback_orientation(image_path)
    
    def _fallback_orientation(self, image_path: str) -> bool:
        """
        Резервный метод определения ориентации по размерам изображения.
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                print(f"Fallback: Image dimensions: {width}x{height}")
                
                # если ширина > высоты, поворачиваем на 90°
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