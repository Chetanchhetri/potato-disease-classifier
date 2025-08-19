# from fastapi import FastAPI , File , UploadFile
# import uvicorn
# import numpy as np
# import tensorflow as tf 
# from PIL import Image
# from io import BytesIO
# Model = tf.keras.models.load_model('C:\\Users\\chhet\\OneDrive\\Desktop\\Machine_Learning\\ml\\my_potato_model.keras')
# class_name = ["Early Blight", "Late Blight", "Healthy"]

# app=FastAPI()

# @app.get("/")

# async def ping():
#     return {"message": "Hello, World!"}

# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     Image = read_file_as_image(await file.read())
#     img_batch = np.expand_dims(Image, axis=0)
#     predictions = Model.predict(img_batch)
#     predicted_class = class_name[np.argmax(predictions[0])]
#     confidence=np.max(predictions[0])
#     pass
#     return {
#         "class": predicted_class,
#         "confidence": float(confidence)
#     }


# if __name__ == "__main__":
#     uvicorn.run(app, port=8000, host="localhost")






from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

# Load Model
Model = tf.keras.models.load_model(
    r"C:\Users\chhet\OneDrive\Desktop\Machine_Learning\ml\my_potato_model.keras"
)
class_names = ["Early Blight", "Late Blight", "Healthy"]

# FastAPI app
app = FastAPI()

# Allow frontend requests (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def ping():
    return {"message": "Hello, World!"}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)).resize((256, 256)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)

    predictions = Model.predict(img_batch)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="localhost")
