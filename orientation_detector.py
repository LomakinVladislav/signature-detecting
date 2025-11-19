import os
import sys
import torch

# Добавляем путь к deep-image-orientation-detection в sys.path
DETECTION_DIR = os.path.join(
    os.path.dirname(__file__), "deep-image-orientation-detection"
)
if DETECTION_DIR not in sys.path:
    sys.path.insert(0, DETECTION_DIR)

# Теперь можем импортировать модули
import config
from src.model import get_orientation_model
from src.utils import get_device, get_data_transforms, load_image_safely


class OrientationDetector:
    """
    Класс для определения ориентации изображений с использованием нейронной сети.
    Инкапсулирует логику из deep-image-orientation-detection.
    """

    def __init__(self, model_path: str = None):
        """
        Инициализирует детектор ориентации.

        Args:
            model_path: Путь к файлу модели. Если None, используется путь по умолчанию.
        """
        if model_path is None:
            model_path = os.path.join(
                DETECTION_DIR, config.MODEL_SAVE_DIR, "best_model.pth"
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                f"Please ensure the model is trained and available."
            )

        self.model_path = model_path
        self.device = get_device()
        self.transforms = get_data_transforms()["val"]

        # Загружаем модель
        self.model = get_orientation_model(pretrained=False)
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict_orientation(self, image_path: str) -> int:
        """
        Предсказывает ориентацию изображения и возвращает угол поворота.

        Args:
            image_path: Путь к изображению

        Returns:
            int: Угол поворота в градусах:
                0 - правильная ориентация
                -90 - нужно повернуть на 90° по часовой стрелке
                180 - нужно повернуть на 180°
                90 - нужно повернуть на 90° против часовой стрелки
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            image = load_image_safely(image_path)
        except Exception as e:
            raise ValueError(f"Error opening image {image_path}: {e}")

        # Преобразуем изображение в тензор
        input_tensor = self.transforms(image).unsqueeze(0).to(self.device)

        # Предсказываем
        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted_idx = torch.max(output, 1)

        predicted_class = predicted_idx.item()

        # Преобразуем класс в угол поворота согласно CLASS_MAP
        # Class 0: 0° (правильная ориентация)
        # Class 1: 90° по часовой стрелке -> -90
        # Class 2: 180°
        # Class 3: 90° против часовой стрелки -> 90
        angle_map = {0: 0, 1: -90, 2: 180, 3: 90}
        return angle_map[predicted_class]

    def get_orientation_message(self, image_path: str) -> str:
        """
        Возвращает текстовое сообщение о необходимом повороте изображения.

        Args:
            image_path: Путь к изображению

        Returns:
            str: Сообщение о необходимом повороте
        """
        predicted_class = self._get_predicted_class(image_path)
        return config.CLASS_MAP[predicted_class]

    def _get_predicted_class(self, image_path: str) -> int:
        """Внутренний метод для получения предсказанного класса."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            image = load_image_safely(image_path)
        except Exception as e:
            raise ValueError(f"Error opening image {image_path}: {e}")

        input_tensor = self.transforms(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted_idx = torch.max(output, 1)

        return predicted_idx.item()
