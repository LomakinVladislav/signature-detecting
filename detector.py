from ultralytics import YOLO

class SignatureDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
    
    def count_signatures(self, image_path: str) -> int:

        results = self.model(image_path, verbose=False)
        
        signature_count = 0
        for r in results:
            if r.boxes is not None:
                signature_count = len(r.boxes)
        
        return signature_count