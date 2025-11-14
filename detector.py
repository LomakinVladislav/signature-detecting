from ultralytics import YOLO
import numpy as np

class SignatureDetector:
    def __init__(self, model_path: str, iou_threshold: float = 0.7): 
        # Для изменения жесткости отсеивания нужно изменять iou_threshold
        self.model = YOLO(model_path)
        self.iou_threshold = iou_threshold
    
    def _calculate_iou(self, box1, box2):
        """Вычисляет IoU между двумя bounding boxes и возвращает значение."""
        x1_min = max(box1[0], box2[0])
        y1_min = max(box1[1], box2[1])
        x2_max = min(box1[2], box2[2])
        y2_max = min(box1[3], box2[3])
        
        intersection_area = max(0, x2_max - x1_min) * max(0, y2_max - y1_min)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0
    
    def _non_max_suppression(self, boxes, confidences):
        """Применяет NMS для удаления дублирующих bounding boxes и возвращает индексы."""
        if len(boxes) == 0:
            return []
        
        indices = np.argsort(confidences)[::-1]
        keep = []
        
        print("Все обнаруженные боксы:")
        for i, idx in enumerate(indices):
            print(f"  Box{i+1}: [{boxes[idx][0]:.1f}, {boxes[idx][1]:.1f}, {boxes[idx][2]:.1f}, {boxes[idx][3]:.1f}] - confidence: {confidences[idx]:.3f}")
        
        while len(indices) > 0:
            current_idx = indices[0]
            keep.append(current_idx)
            current_box = boxes[current_idx]
            remaining_indices = indices[1:]
            
            non_overlapping_indices = []
            
            for idx in remaining_indices:
                iou = self._calculate_iou(current_box, boxes[idx])
                print(f"IoU = {iou:.3f}")
                
                if iou < self.iou_threshold:
                    non_overlapping_indices.append(idx)
            
            indices = non_overlapping_indices
        
        print(f"Сохранилось боксов: {len(keep)}")
        for i, idx in enumerate(keep):
            print(f"  Box{i+1}: [{boxes[idx][0]:.1f}, {boxes[idx][1]:.1f}, {boxes[idx][2]:.1f}, {boxes[idx][3]:.1f}] - confidence: {confidences[idx]:.3f}")
        
        return keep
    
    def count_signatures(self, image_path: str) -> int:
        """Определяет количество подписей на изображении и возвращает число."""
        results = self.model(image_path, verbose=False)
        signature_count = 0
        
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = []
                confidences = []
                
                for i, box in enumerate(r.boxes):
                    bbox = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    boxes.append(bbox)
                    confidences.append(confidence)
                
                keep_indices = self._non_max_suppression(boxes, confidences)
                signature_count = len(keep_indices)
                print(f"ИТОГ: {signature_count} уникальных подписей")
        
        return signature_count