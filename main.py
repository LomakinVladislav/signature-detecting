from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
import os
import uuid
from detector import SignatureDetector
from classificator import DocumentClassificator
from image_processor import ImageProcessor
import uvicorn

SIGNATURE_MODEL_PATH = "models/signature.pt"
CLASSIFICATOR_MODEL_PATH = "models/classificator.pt"

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(SIGNATURE_MODEL_PATH):
        raise Exception(f"The signature model was not found on the way: {SIGNATURE_MODEL_PATH}")
    if not os.path.exists(CLASSIFICATOR_MODEL_PATH):
        raise Exception(f"The classifier model was not found on the way: {CLASSIFICATOR_MODEL_PATH}")
    
    app.state.detector = SignatureDetector(SIGNATURE_MODEL_PATH)
    app.state.classificator = DocumentClassificator(CLASSIFICATOR_MODEL_PATH)
    app.state.image_processor = ImageProcessor()
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
            detail=f"Unsupported file format"
        )
    
    temp_filename = f"temp_uploads/{uuid.uuid4()}{file_extension}"
    os.makedirs("temp_uploads", exist_ok=True)
    
    try:
        with open(temp_filename, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Проверяем и корректируем ориентацию изображения (работает с исходным файлом)
        image_processor = app.state.image_processor 
        was_rotated = image_processor.ensure_correct_orientation(temp_filename)
        
        if was_rotated:
            print("Image was rotated successfully")

        # Классифицируем документ
        classificator = app.state.classificator
        doc_type = classificator.classify_document(temp_filename)
        
        # Если документ рукописный - не обрабатываем
        if doc_type == "handwritten":
            return {
                "document_type": doc_type,
                "number_of_signatures": 0,
                "message": "Handwritten documents are not processed"
            }
        
        # Если документ печатный - подсчитываем подписи
        detector = app.state.detector
        signature_count = detector.count_signatures(temp_filename)
        
        return {
            "document_type": doc_type,
            "number_of_signatures": signature_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        except:
            pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)