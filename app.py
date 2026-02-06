from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import logging

# -------------------
# Logging
# -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------
# App
# -------------------
app = FastAPI(title="Cats vs Dogs Classifier")

MODEL_PATH = "models/model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 128

# -------------------
# Utils
# -------------------
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------
# Routes
# -------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logger.info("Received prediction request")

    image_bytes = await file.read()
    img = preprocess_image(image_bytes)

    prediction = model.predict(img)[0][0]

    label = "dog" if prediction > 0.5 else "cat"
    confidence = float(prediction)

    return {
        "prediction": label,
        "confidence": confidence
    }
