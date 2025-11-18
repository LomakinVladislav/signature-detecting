## Запуск приложения

1. Клонировать репозиторий
```git clone https://github.com/LomakinVladislav/signature-detecting.git```
2. Создать виртуальное окружение
```python -m venv venv```
3. Перейти в витруальное окружение
```.\venv\Scripts\activate```
4. Установить зависимости
```pip install -r requirements.txt```
5. Загрузить модели signature.pt и classificator.pt в папку models/
6. Клонировать в корень проекта репозиторий deep-image-orientation-detection
7. Добавить модель orientation_model_v2_0.9882.pth, скачанную c https://github.com/duartebarbosadev/deep-image-orientation-detection/releases и назвать ее best_model.pth
8. Запустить файл main.py
```python main.py```
или 
```uvicorn main:app --reload```