from fastapi import FastAPI, HTTPException
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import base64
from io import BytesIO
from pydantic import BaseModel
import requests
from pymongo import MongoClient
from fastapi.encoders import jsonable_encoder

app = FastAPI()


class ImageData(BaseModel):
    imagedata: str


mongo_uri = "mongodb+srv://forestwatchai:hackathon%4069@forestwatchai.kshtlwm.mongodb.net/"


mongo_client = MongoClient(mongo_uri)
db = mongo_client["main_db"]
collection1 = db["human_images"]
collection2 = db["not_human_images"]


emailapi_url = "https://emailapi-lfle.onrender.com"

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    return {"ML API": "Server has started"}


@app.post("/alert/human/{imagedata}")
def alert_human(imagedata: str):
    email_data = {
        "subject": "Human Detected",
        "body": "This is a test email body.",
        "imagedata": imagedata
    }
    response = requests.post(
        f"{emailapi_url}/sendmail/",
        json=email_data
    )


@app.post("/alert/nohuman/{imagedata}")
def alert_no_human(imagedata: str):
    email_data = {
        "subject": "No Human was detected",
        "body": "This is a test email body.",
        "imagedata": imagedata
    }
    response = requests.post(
        f"{emailapi_url}/sendmail/",
        json=email_data
    )


@app.post("/insert_into_human_images/", response_model=dict)
def insert_into_human_images(imagedata):
    inserted_data = collection1.insert_one(
        jsonable_encoder({"imagedata": imagedata}))


@app.post("/insert_into_not_human_images/")
def insert_into_not_human_images(imagedata):
    inserted_data = collection2.insert_one(
        jsonable_encoder({"imagedata": imagedata}))


@app.post("/mldetector")
def mldetector(imagedata: ImageData):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image_data = imagedata.imagedata
    image_decoded_data = base64.b64decode(image_data)
    image_rgb = Image.open(BytesIO(image_decoded_data)).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image_rgb, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:].strip(), end="\n")
    print("Confidence Score:", confidence_score)

    if class_name[2:].strip() == 'Human':
        insert_into_human_images(image_data)
        alert_human(image_data)
    elif class_name[2:].strip() == 'Not Human':
        insert_into_not_human_images(image_data)
        alert_no_human(image_data)

    return class_name[2:].strip()
