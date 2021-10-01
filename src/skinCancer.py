import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
from PIL import Image
import io
from tensorflow.keras.applications import Xception
from pprint import pprint


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def make_model():
    base_model = Xception(
        include_top=False, input_shape=(71, 71, 3), weights=None)

    for layer in base_model.layers:
        layer.trainable = True

    model = base_model.output
    model = Flatten()(model)

    model = Dense(512, kernel_initializer='he_uniform')(model)
    model = Dropout(0.2)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    model = Dense(128, kernel_initializer='he_uniform')(model)
    model = Dropout(0.2)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    model = Dense(52, kernel_initializer='he_uniform')(model)
    model = Dropout(0.2)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    model = Dense(16, kernel_initializer='he_uniform')(model)
    model = Dropout(0.2)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    output = Dense(7, activation='softmax')(model)

    return Model(inputs=base_model.input, outputs=output)


def process_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    img = image.resize((71, 71))
    img = img_to_array(img)
    img = img.astype('float32')
    img /= 255.0
    img = np.array([img])
    return img


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def SCprediction():
    print("""
        Welcome to SkinCancer Classifier
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    """)
    path = input("Enter the image path: ")
    result = {'success': False}
    if(allowed_file(path)):
        image = Image.open(path)

        img = process_image(image)
        prediction = model.predict(img)

        for i in range(len(classes)):
            result[classes[i]] = str(
                round(100*prediction[0][i]/sum(prediction[0]), 3)) + "%"

        result['success'] = True
        result['Predicted_Class'] = f"{classes[np.argmax(prediction[0])]}"
        pprint(result)

    else:
        pprint(result)

    print("\n\n")
    return


model = make_model()
model.load_weights("src\Resources\XceptionNN.h5")

classes = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions',
           'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions']
