from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import shutil
import os

# Load trained YOLO model
MODEL_PATH = r"C:\helmet\best.pt"
model = YOLO(MODEL_PATH)

# Initialize FastAPI app
app = FastAPI()

# Ensure temp folder exists
UPLOAD_FOLDER = "temp_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def home():
    return {"message": "Helmet Detection API is running!"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Endpoint to detect helmets in an uploaded image."""
    try:
        # Save the uploaded file
        file_path = f"{UPLOAD_FOLDER}/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read image
        img = cv2.imread(file_path)

        # Run YOLO inference
        results = model(img)

        # Process results
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                conf = float(box.conf[0])  # Confidence score
                cls = int(box.cls[0])  # Class index

                detections.append({
                    "class": model.names[cls],  # "With Helmet" or "Without Helmet"
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2]
                })

        # Remove saved file to save space
        os.remove(file_path)

        return JSONResponse(content={"detections": detections})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
