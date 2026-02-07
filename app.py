import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI(title="Cats vs Dogs Classifier API")

# Load trained model
MODEL_PATH = "models/model.h5"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (150, 150)
CLASS_NAMES = ["cat", "dog"]


def preprocess_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.get("/")
def root():
    return {"message": "Cats vs Dogs API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = preprocess_image(image_bytes)

        preds = model.predict(image)
        confidence = float(preds[0][0])

        label = CLASS_NAMES[int(confidence > 0.5)]

        return JSONResponse(
            content={
                "prediction": label,
                "confidence": confidence
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# IMPORTANT: allows running via `python app.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
