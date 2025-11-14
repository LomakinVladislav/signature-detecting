from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
import os
import uuid
from detector import SignatureDetector
from classificator import DocumentClassificator
import uvicorn

SIGNATURE_MODEL_PATH = "models/signature.pt"
CLASSIFICATOR_MODEL_PATH = "models/classificator.pt"

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(SIGNATURE_MODEL_PATH):
        raise Exception(f"Модель подписей не найдена по пути: {SIGNATURE_MODEL_PATH}")
    if not os.path.exists(CLASSIFICATOR_MODEL_PATH):
        raise Exception(f"Модель классификатора не найдена по пути: {CLASSIFICATOR_MODEL_PATH}")
    
    app.state.detector = SignatureDetector(SIGNATURE_MODEL_PATH)
    app.state.classificator = DocumentClassificator(CLASSIFICATOR_MODEL_PATH)
    yield

app = FastAPI(
    title="Signature Detection API",
    lifespan=lifespan
)

@app.post("/detect-signatures")
async def detect_signatures(file: UploadFile = File(...)):
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Неподдерживаемый формат файла"
        )
    
    temp_filename = f"temp_uploads/{uuid.uuid4()}{file_extension}"
    os.makedirs("temp_uploads", exist_ok=True)
    
    try:
        with open(temp_filename, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Классифицируем документ
        classificator = app.state.classificator
        doc_type = classificator.classify_document(temp_filename)
        
        # Если документ рукописный - не обрабатываем
        if doc_type == "рукописный":
            return {
                "document_type": doc_type,
                "number_of_signatures": 0,
                "message": "Рукописные документы не обрабатываются"
            }
        
        # Если документ печатный - подсчитываем подписи
        detector = app.state.detector
        signature_count = detector.count_signatures(temp_filename)
        
        return {
            "document_type": doc_type,
            "number_of_signatures": signature_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")
    
    finally:
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        except:
            pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)