import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image

# ---------------------------------------------------------------------------
# Startup: load the model once when the server starts, not on every request.
# This is critical for performance — loading a model takes ~1 second.
# ---------------------------------------------------------------------------
app = FastAPI(title="PPE Detector API", version="1.0")
model = YOLO("best.pt")

# ---------------------------------------------------------------------------
# Health-check endpoint — useful for Docker and cloud platforms to verify
# the service is alive.
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Main prediction endpoint
# ---------------------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), confidence: float = 0.25):
    """
    Receives an image file and returns bounding box detections.

    - file: image uploaded by the client (jpg, png, etc.)
    - confidence: minimum confidence threshold (default 0.25)
    """

    # 1. Read the raw bytes from the uploaded file
    image_bytes = await file.read()

    # 2. Convert bytes → numpy array → OpenCV image (BGR format)
    np_array = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image_bgr is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Could not decode image. Make sure it is a valid jpg or png."}
        )

    # 3. Run inference
    results = model(image_bgr, conf=confidence)[0]

    # 4. Build the response: one dict per detected bounding box
    detections = []
    for box in results.boxes:
        detections.append({
            "class_id":    int(box.cls),
            "class_name":  model.names[int(box.cls)],
            "confidence":  round(float(box.conf), 4),
            "bbox": {          # pixel coordinates
                "x1": round(float(box.xyxy[0][0])),
                "y1": round(float(box.xyxy[0][1])),
                "x2": round(float(box.xyxy[0][2])),
                "y2": round(float(box.xyxy[0][3])),
            }
        })

    return {
        "model":      "YOLOv8s-PPE",
        "confidence_threshold": confidence,
        "total_detections":     len(detections),
        "detections":           detections,
    }