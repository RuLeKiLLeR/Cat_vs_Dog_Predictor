import streamlit as st
import io
import tensorflow as tf 
from tensorflow import keras
from keras._tf_keras.keras.preprocessing import image
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('cat_dog_classifier.h5')


def predict_image(img):
    img = img.resize((200, 200))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    if result >= 0.5:
        return "Dog"
    else:
        return "Cat"

# Streamlit app
st.title("Cat and Dog Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Predict button
    if st.button('Predict'):
        result = predict_image(img)
        st.write(f'The uploaded image is a {result}.')
