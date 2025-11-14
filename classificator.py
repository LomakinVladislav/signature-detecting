from ultralytics import YOLO
import os

class DocumentClassificator:
    def __init__(self, model_path: str = "models/classificator.pt"):
        self.model = YOLO(model_path)
        self.class_names = {0: "handwritten", 1: "printed"}
    
    def classify_document(self, image_path: str) -> str:
        """Классифицирует документ и возвращает его тип."""
        results = self.model(image_path, verbose=False)
        
        for r in results:
            if r.probs is not None:
                # Получаем индекс класса с наибольшей вероятностью
                class_id = r.probs.top1
                return self.class_names.get(class_id, "uknown")
        
        return "uknown"