from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import json
import cv2
from ultralytics import YOLO
from fastapi.responses import JSONResponse, FileResponse

app = FastAPI()

# Directorio para guardar imágenes subidas
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Cargar el modelo YOLOv8
model = YOLO('yolov8n.pt')

# Definir el archivo JSON donde se guardarán las predicciones
file_name = 'predicciones.JSON'

# Función para guardar predicciones en un archivo JSON
def save_prediction(filename, results_list):
    try:
        with open(file_name, 'r') as file:
            predictions = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        predictions = []

    # Estructura de la predicción que vamos a guardar
    prediction_data = {
        "filename": filename,
        "detections": results_list
    }

    predictions.append(prediction_data)

    with open(file_name, 'w') as file:
        json.dump(predictions, file, indent=4)


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Procesar la imagen con YOLOv8
    image = cv2.imread(str(file_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)

    # Extraer las detecciones
    detections = results[0].boxes.data.cpu().numpy()
    class_names = model.names
    results_list = []

    # Procesar cada detección
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        class_name = class_names[int(cls)]
        results_list.append({
            "class": class_name,
            "confianza": float(conf),
            "coordenadas": {
                "Superior izquierda": [float(x1), float(y1)],
                "Inferior derecha": [float(x2), float(y2)]
            }
        })

    # Guardar las predicciones en el archivo JSON
    save_prediction(file.filename, results_list)
        
    return JSONResponse({"filename": file.filename, "detections": results_list})


@app.get("/get-image/{filename}")
async def get_image(filename: str):
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)