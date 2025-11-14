## Запуск приложения

1. Клонировать репозиторий
```git clone https://github.com/LomakinVladislav/signature-detecting.git```
2. Создать виртуальное окружение
```python -m venv venv```
3. Перейти в витруальное окружение
```.\venv\Scripts\activate```
4. Установить зависимости
```pip install -r requirements.txt```
5. Загрузить модель signature.pt в папку models/
6. Запустить файл main.py
```python main.py```
или 
```uvicorn main:app --reload```