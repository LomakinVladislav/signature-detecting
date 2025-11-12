from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
import os
import uuid
from detector import SignatureDetector
import uvicorn

MODEL_PATH = "models/signature.pt"

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise Exception(f"Модель не найдена по пути: {MODEL_PATH}")
    
    app.state.detector = SignatureDetector(MODEL_PATH)
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
        
        detector = app.state.detector
        signature_count = detector.count_signatures(temp_filename)
        
        return {"number_of_signatures": signature_count}
        
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